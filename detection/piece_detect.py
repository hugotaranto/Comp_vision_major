import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

import torch
from segment_anything import sam_model_registry, SamPredictor

# file imports
from detection.plots import *
from detection.util import *
from detection.board_detect import get_board_area
# from .depth import depth_predict, load_model
from detection.depth import depth_predict, load_model


IMAGE_DIRECTORY = './every_place'
# IMAGE_DIRECTORY = './side_test'
# IMAGE_DIRECTORY = '../images'
# DEPTH_MAP_DIRECTORY = './our_depths'
# DEPTH_MAP_DIRECTORY = './margold_depth/depth_npy'

# save directory
MASK_OUTPUT_DIRECTORY = './detection/detection_output'

SAM_MODEL_TYPE = 'vit_l'
SAM_MODEL_PATH = './sam_checkpoints/sam_vit_l_0b3195.pth'


DEPTH_PRO_CHECKPOINT_PATH = './ml-depth-pro/checkpoints/depth_pro.pt'

MAX_DIM = 1600

def depth_to_point_cloud(depth_map, mask=None):
    H, W = depth_map.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    # Simple coordinates, assuming fx=fy=1, cx=W/2, cy=H/2
    x = (xs - W/2) * depth_map
    y = (ys - H/2) * depth_map
    z = depth_map
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    if mask is not None:
        mask_flat = mask.flatten().astype(bool)
        points = points[mask_flat]

    return points

def segment_board_plane(depth_map, board_mask):
    points = depth_to_point_cloud(depth_map, mask=board_mask)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.002,  # adjust if noisy
        ransac_n=3,
        num_iterations=500
    )
    # segment_plane's normal sign is arbitrary and can flip between runs.
    # detect_pieces keeps height_residual < median - k*mad (the negative tail),
    # and pieces sit closer to the camera (smaller depth z) than the board.
    # Forcing the z-component positive makes their residual a*x+b*y+c*z+d land on
    # that negative side, so pieces are always detected (prevents random zero-detection).
    if plane_model[2] < 0:
        plane_model = [-v for v in plane_model]
    [a, b, c, d] = plane_model
    return plane_model, inliers

def subtract_plane(depth_map, plane_model):
    a, b, c, d = plane_model
    H, W = depth_map.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    x = (xs - W/2) * depth_map
    y = (ys - H/2) * depth_map
    z = depth_map
    dist = a*x + b*y + c*z + d
    return dist

def get_central_points(piece_mask, K=3):

    # Compute distance transform
    dist = cv2.distanceTransform(piece_mask, cv2.DIST_L2, 5)
    h, w = dist.shape

    # Split image into K horizontal bands
    y_splits = np.linspace(0, h, K + 1, dtype=int)

    points = []
    for i in range(K):
        y0, y1 = y_splits[i], y_splits[i + 1]
        band = dist[y0:y1, :]

        if band.size == 0:
            continue

        # Find max distance in this band
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(band)
        x_best, y_best_local = max_loc
        y_best = y0 + y_best_local

        # Only keep if it's inside mask
        if piece_mask[y_best, x_best] > 0:
            points.append((x_best, y_best))

    return points

def get_most_central_point(piece_mask):
    if piece_mask.dtype != np.uint8:
        piece_mask = piece_mask.astype(np.uint8)

    # Compute distance transform
    dist = cv2.distanceTransform(piece_mask, cv2.DIST_L1, 3)

    # Find the maximum value location (most central point)
    _, _, _, max_loc = cv2.minMaxLoc(dist)

    x_best, y_best = max_loc

    return (x_best, y_best)


def refine_mask_by_residual(mask, height_residual, corners, keep_fraction=0.5, expand_top_px=50):

    refined_mask = np.zeros_like(mask, dtype=np.uint8)

    # --- Move only the top corners upward ---
    expanded_corners = corners.copy()

    # Find which corners are "top" based on smallest y values
    y_values = corners[:, 1]
    top_indices = np.argsort(y_values)[:2]  # two smallest y-values = top edge

    # Move those points upward (subtract from y)
    expanded_corners[top_indices, 1] -= expand_top_px

    expanded_corners = expanded_corners.astype(np.int32)

    # Create a mask of the polygon defined by the corners
    board_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(board_mask, [expanded_corners.astype(np.int32)], 1)
    board_mask_bool = board_mask.astype(bool)

    num_labels, labels = cv2.connectedComponents(mask)

    for i in range(1, num_labels):  # skip background
        component_mask = (labels == i)
        values = height_residual[component_mask]

        if len(values) == 0:
            continue

        # check if the component intersects the board
        if not np.any(np.logical_and(component_mask, board_mask_bool)):
            continue

        # Find cutoff for the lowest X% values
        cutoff = np.percentile(values, keep_fraction * 100)

        # Keep only pixels below that cutoff
        refined_component = np.logical_and(component_mask, height_residual <= cutoff)

        refined_mask[refined_component] = 1

    return refined_mask

def detect_pieces(image, depth_map, plane_model, corners, show=True):

    height_residual = subtract_plane(depth_map, plane_model)
    # height_residual[board_mask == 0] = 1
    # board_heights = height_residual[board_mask > 0]

    median_h = np.median(height_residual)
    mad = np.median(np.abs(height_residual - median_h))  # median absolute deviation

    # pieces are lower than board by > k*MAD
    k = 6
    threshold = median_h - k*mad
    mask = (height_residual < threshold).astype(np.uint8)

    refined_mask = refine_mask_by_residual(mask, height_residual, corners, keep_fraction=0.5)
    refined_mask = refine_mask_by_residual(refined_mask, height_residual, corners, keep_fraction=0.65)

    kernel = np.ones((6, 6), np.uint8)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=5)

    # erode the mask to separate pieces
    # eroded_mask = cv2.erode(refined_mask, kernel, iterations=2)
    eroded_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=5)
    # eroded_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))


    # Label pieces
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded_mask)
    print(f"Detected {num_labels-1} pieces at threshold={threshold}")

    height, width = image.shape[:2]
    area_reject_percentage = 5 # reject "pieces" detected that are > 5% of the image
    area_reject_threshold = (height * width / 100) * area_reject_percentage

    min_area = 5

    valid_centroids = []

    display_image = image.copy()

    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]

        # reject if the area is too big
        if area > area_reject_threshold or area < min_area:
            continue

        # Extract mask for this component
        piece_mask = (labels[y:y+h, x:x+w] == i).astype(np.uint8)

        # points = get_central_points(piece_mask, K=1)
        point = get_most_central_point(piece_mask)

        # Shift local points to global image coordinates
        # global_points = [(x + px, y + py) for (px, py) in points]
        global_points = [(x + point[0], y + point[1])]

        valid_centroids.append(global_points)

        if show:
            # Draw them for debugging
            for (gx, gy) in global_points:
                cv2.circle(display_image, (int(gx), int(gy)), 10, (0, 0, 255), -1)

    if show:

        # --- Plot 3 panels side-by-side ---
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))

        axes[0, 1].imshow(depth_map, cmap='turbo')
        axes[0, 1].set_title("Monocular Depth Map Estimation")

        im0 = axes[1, 0].imshow(height_residual, cmap='turbo')
        axes[1, 0].set_title("Height residuals (above plane)")
        plt.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)

        axes[1, 1].imshow(mask, cmap='gray')
        axes[1, 1].set_title(f"Binary mask (<{threshold:.4f})")

        axes[1, 2].imshow(refined_mask, cmap='gray')
        axes[1, 2].set_title("Refined Mask")

        # axes[2, 0].imshow(eroded_mask, cmap='gray')
        # axes[2, 0].set_title("Eroded Mask")
        # --- Plot eroded mask with centroids ---
        axes[2, 0].imshow(eroded_mask, cmap='gray')
        axes[2, 0].set_title("Eroded Mask + Centroids")
        for global_points in valid_centroids:
            for (gx, gy) in global_points:
                axes[2, 0].plot(gx, gy, 'ro', markersize=2)

        axes[2, 1].imshow(display_image)
        axes[2, 1].set_title(f"Detected pieces ({num_labels-1})")

        for i in range(len(axes)):
            for j in range(len(axes[i])):
                axes[i, j].axis('off')

        plt.tight_layout()
        plt.show()

    return valid_centroids


def get_local_negative_points(centroids, current_idx, num_negatives=3):

    current_points = np.array(centroids[current_idx])
    
    # Flatten all points from other pieces
    other_points = []
    for i, pts in enumerate(centroids):
        if i == current_idx:
            continue
        other_points.extend(pts)
    if len(other_points) == 0:
        return []

    other_points = np.array(other_points)

    # Compute distance from each other point to the **closest point in the current piece**
    distances = np.min(np.linalg.norm(other_points[:, None, :] - current_points[None, :, :], axis=2), axis=1)

    # Get indices of the closest `num_negatives` points
    if len(distances) <= num_negatives:
        closest_indices = np.arange(len(distances))
    else:
        closest_indices = np.argpartition(distances, num_negatives)[:num_negatives]

    neg_points = other_points[closest_indices].tolist()
    return neg_points


def mask_contains_any_points(mask, points):
    if len(points) == 0:
        return False
    points = np.round(points).astype(int)
    # Clip points to image boundaries
    points[:, 0] = np.clip(points[:, 0], 0, mask.shape[1]-1)
    points[:, 1] = np.clip(points[:, 1], 0, mask.shape[0]-1)

    return mask[points[:, 1], points[:, 0]].any()

def segment_with_sam(image, centroids, predictor, show_each=False, show_final=False):
    predictor.set_image(image)

    height, width = image.shape[:2]
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    piece_masks = []

    piece_index = 1

    for i, pos_points in enumerate(centroids):

        # if i < 17:
        #     continue

        # --- Build positive & negative prompts ---
        pos_points = np.array(pos_points, dtype=np.float32)

        neg_points = get_local_negative_points(centroids, i, num_negatives=0)
        neg_points = np.array(neg_points, dtype=np.float32)

        # Combine both
        all_points = np.concatenate([pos_points, neg_points], axis=0) if len(neg_points) > 0 else pos_points
        all_labels = np.concatenate([
            np.ones(len(pos_points), dtype=np.int32),
            np.zeros(len(neg_points), dtype=np.int32)
        ]) if len(neg_points) > 0 else np.ones(len(pos_points), dtype=np.int32)

        # --- Predict mask ---
        masks, scores, _ = predictor.predict(
            point_coords=all_points,
            point_labels=all_labels,
            multimask_output=True
        )

        # all points for other pieces
        other_piece_points = np.concatenate(
            [np.array(pts, dtype=np.float32) for j, pts in enumerate(centroids) if j != i],
            axis=0
        ) if len(centroids) > 1 else np.empty((0, 2), dtype=np.float32)

        valid_masks = []

        # for m in masks:
        for i in range(len(masks)):
            m = masks[i].astype(np.uint8)
            score = scores[i]

            # print("before morph")
            # plot_segment(image, m, pos_points, neg_points, score)
            # remove speckles
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations = 1)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations = 1)

            if show_each:
                plot_segment(image, m, pos_points, neg_points, score)

            if score < 0.88:
                print("score too low")
                continue

            # check if the mask is too big
            area = m.sum()
            # print(f"area: {area}, height: {height}, width: {width}")

            if area > 0.03 * height * width:
                print("too big")
                continue


            # check for one connected component (1 contour)
            mask_uint8 = (m.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # print(f"num contours: {len(contours)}")

            # if len(contours) != 1:
            #     print("disconnected")
            #     continue
            if len(contours) > 1:
                # Find contour with the largest area that actually contains at least one positive point
                valid_contours = []
                for contour in contours:
                    for point in pos_points:
                        if cv2.pointPolygonTest(contour, tuple(point), False) >= 0:
                            valid_contours.append(contour)
                            break  # no need to check all points for this contour

                if len(valid_contours) == 0:
                    # fallback: no contour contains the positive point — skip this mask
                    print("No contour contains the prompt point, skipping mask")
                    continue

                # choose the largest among the valid ones
                largest_contour = max(valid_contours, key=cv2.contourArea)

                # Create a clean mask with only that contour filled in
                clean_mask = np.zeros_like(mask_uint8)
                cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                m = (clean_mask > 0).astype(np.uint8)


            # check if the mask segments multiple pieces
            if mask_contains_any_points(m, other_piece_points):
                print("masked other pieces")
                continue
            

            # if all filtering passed, this is valid
            valid_masks.append(m)

        if len(valid_masks) == 0:
            print("======= No valid masks ====================================")
            continue
        else:
            areas = [m.sum() for m in valid_masks]
            idx = np.argmax(areas)
            selected_mask = valid_masks[idx]
            selected_index = idx

        selected_mask = selected_mask.astype(bool)

        combined_mask[selected_mask] = piece_index
        piece_index += 1

        piece_masks.append(selected_mask)

        # --- Visualization ---
        if show_final:
            print("Selected mask")
            plot_segment(image, selected_mask, pos_points, neg_points, scores[selected_index])

    predictor.reset_image()

    return combined_mask, piece_masks


def expand_corners(corners, factor=1/16, show=False, image=None) -> np.ndarray:
    center = corners.mean(axis=0)

    # Compute the bounding box width and height
    w = corners[:, 0].max() - corners[:, 0].min()
    h = corners[:, 1].max() - corners[:, 1].min()

    # Expansion factor (1/16th of size)
    scale_x = w * factor
    scale_y = h * factor

    expanded_corners = corners.copy().astype(float)
    for i, (x, y) in enumerate(corners):
        dx = x - center[0]
        dy = y - center[1]
        norm = np.sqrt(dx**2 + dy**2)
        if norm > 0:
            dx /= norm
            dy /= norm
        expanded_corners[i, 0] += dx * scale_x
        expanded_corners[i, 1] += dy * scale_y

    # Clip corners to image bounds
    if image is not None:
        h_img, w_img = image.shape[:2]
        expanded_corners[:, 0] = np.clip(expanded_corners[:, 0], 0, w_img - 1)
        expanded_corners[:, 1] = np.clip(expanded_corners[:, 1], 0, h_img - 1)

    expanded_corners = expanded_corners.astype(np.int32)

    if show and image is not None:
        vis_corners = image.copy()
        for i, (x, y) in enumerate(expanded_corners):
            cv2.circle(vis_corners, (int(x), int(y)), 20, (0,0,255), -1)
            cv2.putText(vis_corners, f"{i+1}", (int(x)+5,int(y)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        plt.figure(figsize=(15, 15))
        plt.imshow(vis_corners)
        plt.title("Board Corners from Intersections")
        plt.axis('off')
        plt.show()

    return expanded_corners


def load_sam(sam_path):

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=sam_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam.to(device)

    predictor = SamPredictor(sam)

    return predictor

def get_piece_segments(image, f_px, name, sam_predictor, depth_model, depth_transform):

    print(f"Finding Pieces in image: {name}")

    print("getting board mask")
    corners = get_board_area(image, show=False, show_detail=False)

    if corners is None:
        print(f"Board not detected in image: {name}")
        return None, None, None

    # expand the corners out slightly
    expanded_corners = expand_corners(corners, show=False, image=image, factor=1/4)

    image_resized, corners = resize_image(image, MAX_DIM, expanded_corners, corners, show=False)

    print("predicting depth")
    depth = depth_predict(image_resized, f_px, depth_model, depth_transform)
    print("depth prediction done")

    print("getting plane model using ransac")
    plane_model, inliers = segment_board_plane(depth, None)

    print("detecting pieces")
    centroids = detect_pieces(image_resized, depth, plane_model, corners, show=True)

    print("Segmenting with SAM")
    segmentation_mask, piece_masks = segment_with_sam(image_resized, centroids, sam_predictor, show_each=False, show_final=False)

    # detect_poses(segmentation_mask, corners, show=True, image=image_resized, show_each=True)

    # plot_segmentation_mask(image_resized, segmentation_mask)

    # pad the images to target before saving
    final_image = pad_to_target(image_resized, MAX_DIM)
    segmentation_mask, corners = pad_to_target(segmentation_mask, MAX_DIM, corners)
    
    save_segmentations_to_file(MASK_OUTPUT_DIRECTORY, name, segmentation_mask, final_image)

    return segmentation_mask, final_image, corners


def main():

    images, f_pxs, names = load_images(IMAGE_DIRECTORY)

    # load depth model
    print("Loading depth model")
    depth_model, depth_transform = load_model(DEPTH_PRO_CHECKPOINT_PATH)
    print("Loading SAM")
    sam_predictor = load_sam(SAM_MODEL_PATH)

    for i in range(len(images)):
        image = images[i]
        f_px = f_pxs[i]
        name = names[i]

        segmentation_mask, image_resized, corners = get_piece_segments(image, f_px, name, sam_predictor, depth_model, depth_transform)

        if segmentation_mask is None:
            continue

        # plot_segmentation_mask(image_resized, segmentation_mask)

if __name__ == "__main__":
    main()
