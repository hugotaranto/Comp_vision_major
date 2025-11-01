import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import os

import torch
from segment_anything import sam_model_registry, SamPredictor

# file imports
from plots import *
from util import *
from board_detect import get_board_area

IMAGE_DIRECTORY = './side_test'
DEPTH_MAP_DIRECTORY = './our_depths'
# DEPTH_MAP_DIRECTORY = './margold_depth/depth_npy'

# save directory
MASK_OUTPUT_DIRECTORY = './detection_output'

SAM_MODEL_TYPE = 'vit_l'
SAM_MODEL_PATH = './sam_checkpoints/sam_vit_l_0b3195.pth'


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
        num_iterations=1000
    )
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

def refine_mask_by_residual(mask, height_residual, keep_fraction=0.5):

    refined_mask = np.zeros_like(mask, dtype=np.uint8)

    num_labels, labels = cv2.connectedComponents(mask)

    for i in range(1, num_labels):  # skip background
        component_mask = (labels == i)
        values = height_residual[component_mask]

        if len(values) == 0:
            continue

        # Find cutoff for the lowest X% values
        cutoff = np.percentile(values, keep_fraction * 100)

        # Keep only pixels below that cutoff
        refined_component = np.logical_and(component_mask, height_residual <= cutoff)

        refined_mask[refined_component] = 1

    return refined_mask



def detect_pieces(image, depth_map, plane_model, board_mask, show=True):

    height_residual = subtract_plane(depth_map, plane_model)
    height_residual[board_mask == 0] = 1
    board_heights = height_residual[board_mask > 0]

    median_h = np.median(board_heights)
    mad = np.median(np.abs(board_heights - median_h))  # median absolute deviation

    # pieces are lower than board by > k*MAD
    k = 4.5
    threshold = median_h - k*mad
    mask = (height_residual < threshold).astype(np.uint8)

    refined_mask = refine_mask_by_residual(mask, height_residual, keep_fraction=0.4)

    kernel = np.ones((3, 3), np.uint8)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # erode the mask to separate pieces
    kernel = np.ones((2,2), np.uint8)
    eroded_mask = cv2.erode(refined_mask, kernel, iterations=2)

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

        points = get_central_points(piece_mask, K=2)

        # Shift local points to global image coordinates
        global_points = [(x + px, y + py) for (px, py) in points]

        valid_centroids.append(global_points)

        if show:
            # Draw them for debugging
            for (gx, gy) in global_points:
                cv2.circle(display_image, (int(gx), int(gy)), 3, (0, 0, 255), -1)

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

        axes[1, 2].imshow(eroded_mask, cmap='gray')
        axes[1, 2].set_title("Eroded Mask")

        axes[2, 1].imshow(display_image[..., ::-1])
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

def segment_with_sam(image, centroids, show_each=False):
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_MODEL_PATH)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam.to(device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)

    height, width = image.shape[:2]
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    piece_masks = []

    piece_index = 1

    for i, pos_points in enumerate(centroids):
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

        print(all_labels)

        # --- Predict mask ---
        masks, scores, _ = predictor.predict(
            point_coords=all_points,
            point_labels=all_labels,
            multimask_output=True
        )

        # choose the highest scoring mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        mask_area = best_mask.sum()

        # all points for other pieces
        other_piece_points = np.concatenate(
            [np.array(pts, dtype=np.float32) for j, pts in enumerate(centroids) if j != i],
            axis=0
        ) if len(centroids) > 1 else np.empty((0, 2), dtype=np.float32)

        valid_masks = []
        for m in masks:

            # check if the mask is too big
            area = m.sum()
            if area > 0.3 * height * width:
                continue

            # check if the mask segments multiple pieces
            if mask_contains_any_points(m, other_piece_points):
                continue
            
            # check for one connected component (1 contour)
            mask_uint8 = (m.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) != 1:
                continue

            # if all filtering passed, this is valid
            valid_masks.append(m)

        if len(valid_masks) == 0:
            selected_mask = best_mask
            print("Highest score mask used")
        else:
            areas = [m.sum() for m in valid_masks]
            idx = np.argmax(areas)
            selected_mask = valid_masks[idx]
            print("Biggest exclusive mask used, score:", scores[idx])

        # selected_mask = biggest_mask


        combined_mask[selected_mask] = piece_index
        piece_index += 1

        piece_masks.append(selected_mask)

        # --- Visualization ---
        if show_each:
            overlay = image.copy().astype(np.float32) / 255.0
            mask_color = np.zeros_like(overlay)
            mask_color[..., 0] = 1.0  # red

            alpha = 0.5
            overlay = np.where(selected_mask[..., None],
                               (1 - alpha) * overlay + alpha * mask_color,
                               overlay)

            plt.figure(figsize=(25, 25))
            plt.imshow(overlay)
            plt.scatter(pos_points[:, 0], pos_points[:, 1],
                        c='cyan', s=50, edgecolors='black', label='Positive')
            if len(neg_points) > 0:
                plt.scatter(neg_points[:, 0], neg_points[:, 1],
                            c='red', s=30, label='Negative', alpha=0.6)
            plt.legend()
            plt.title(f"Piece {i+1} (Score={scores[best_idx]:.3f})")
            plt.axis('off')
            plt.show()

    return combined_mask, piece_masks

def main():

    data = load_image_depth_pairs(IMAGE_DIRECTORY, DEPTH_MAP_DIRECTORY, type="depth-pro")

    for i in range(len(data)):
        image = data[i][0]
        depth = data[i][1]
        name = data[i][2]

        # print(threshold_util(np.array([image])))

        board_mask = get_board_area(image, show=False, show_detail=False)
        # apply the mask to the depth map
        # depth = depth * board_mask

        plane_model, inliers = segment_board_plane(depth, board_mask)
        centroids = detect_pieces(image, depth, plane_model, board_mask, show=True)

        segmentation_mask, piece_masks = segment_with_sam(image, centroids, show_each=False)

        # plot_segmentation_mask(image, segmentation_mask)

        save_segmentations_to_file(MASK_OUTPUT_DIRECTORY, name, segmentation_mask, image)

if __name__ == "__main__":
    main()
