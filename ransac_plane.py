import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

import torch
from segment_anything import sam_model_registry, SamPredictor


from plots import *

# IMAGE_DIRECTORY = './boards'
# DEPTH_MAP_DIRECTORY = './depth-pro_output'
# IMAGE_DIRECTORY = './our_images'

IMAGE_DIRECTORY = './side_test'
DEPTH_MAP_DIRECTORY = './our_depths'

SAM_MODEL_TYPE = 'vit_l'
SAM_MODEL_PATH = './sam_vit_l_0b3195.pth'

# LOWER_GREEN_HSV_THRESHOLD = [0, 155, 40]
LOWER_GREEN_HSV_THRESHOLD = [35, 85, 65]
UPPER_GREEN_HSV_THRESHOLD = [78, 255, 255]


# [35, 85, 65, 179, 255, 255]


def get_board_area(image, show=False, show_detail=False):
    # --- Step 1: create green mask ---
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask_green = cv2.inRange(hsv, np.array(LOWER_GREEN_HSV_THRESHOLD),
                                     np.array(UPPER_GREEN_HSV_THRESHOLD))
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask_gray = (mask_green * 255).astype(np.uint8)

    # Convert to float and blur/dilate
    mask_float = cv2.dilate(mask_gray.astype(np.float32), np.ones((3,3), np.uint8), iterations=1)
    mask_blur = cv2.GaussianBlur(mask_float, (5,5), 0)
    mask_blur_norm = cv2.normalize(mask_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if show_detail:
        plt.imshow(mask_blur_norm, cmap='gray')
        plt.title("Blurred Green Mask")
        plt.axis('off')
        plt.show()

    # --- Step 2: edge detection ---
    edges = cv2.Canny(mask_blur_norm, 50, 100, apertureSize=3)
    if show_detail:
        plt.imshow(edges, cmap='gray')
        plt.title("Canny Edges from Green Mask")
        plt.axis('off')
        plt.show()

    # --- Step 3: Hough line detection ---
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=20, maxLineGap=40)
    if lines is None:
        print("No lines detected.")
        return np.zeros(mask_gray.shape, dtype=np.uint8), None

    if show_detail:
        vis_lines = image.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(vis_lines, (x1, y1), (x2, y2), (0,255,0), 2)
        plt.imshow(vis_lines)
        plt.title("All Hough Lines Detected")
        plt.axis('off')
        plt.show()

    # --- Step 2: separate vertical/horizontal lines ---
    vertical_lines = []
    horizontal_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < abs(y2 - y1):  # vertical-ish
            vertical_lines.append([[x1, y1], [x2, y2]])
        else:  # horizontal-ish
            horizontal_lines.append([[x1, y1], [x2, y2]])

    vertical_lines = np.array(vertical_lines)      # shape: (num_vertical_lines, 2, 2)
    horizontal_lines = np.array(horizontal_lines)  # shape: (num_horizontal_lines, 2, 2)

    if len(vertical_lines) < 2 or len(horizontal_lines) < 2:
        print("Not enough vertical/horizontal lines")
        return None

    # --- Step 3: find extremal lines ---

    # For verticals: use min/max of mean x of each line
    vertical_mean_x = vertical_lines[:,:,0].mean(axis=1)
    left_idx = np.argmin(vertical_mean_x)
    right_idx = np.argmax(vertical_mean_x)
    left_line_pts = vertical_lines[left_idx]   # full 2 points
    right_line_pts = vertical_lines[right_idx]

    # For horizontals: use min/max of mean y of each line
    horizontal_mean_y = horizontal_lines[:,:,1].mean(axis=1)
    top_idx = np.argmin(horizontal_mean_y)
    bottom_idx = np.argmax(horizontal_mean_y)
    top_line_pts = horizontal_lines[top_idx]
    bottom_line_pts = horizontal_lines[bottom_idx]

    if show_detail and image is not None:
        vis_selected = image.copy()
        for pts in [left_line_pts, right_line_pts]:
            cv2.line(vis_selected, tuple(pts[0]), tuple(pts[1]), (255,0,0), 2)
        for pts in [top_line_pts, bottom_line_pts]:
            cv2.line(vis_selected, tuple(pts[0]), tuple(pts[1]), (0,0,255), 2)
        plt.imshow(vis_selected)
        plt.title("Selected Extremal Lines")
        plt.axis('off')
        plt.show()

    # --- Step 4: fit lines ---
    vx_l, vy_l, x0_l, y0_l = cv2.fitLine(left_line_pts, cv2.DIST_L2,0,0.01,0.01)
    vx_r, vy_r, x0_r, y0_r = cv2.fitLine(right_line_pts, cv2.DIST_L2,0,0.01,0.01)
    vx_t, vy_t, x0_t, y0_t = cv2.fitLine(top_line_pts, cv2.DIST_L2,0,0.01,0.01)
    vx_b, vy_b, x0_b, y0_b = cv2.fitLine(bottom_line_pts, cv2.DIST_L2,0,0.01,0.01)

    # Convert to scalars
    vx_l, vy_l, x0_l, y0_l = vx_l.item(), vy_l.item(), x0_l.item(), y0_l.item()
    vx_r, vy_r, x0_r, y0_r = vx_r.item(), vy_r.item(), x0_r.item(), y0_r.item()
    vx_t, vy_t, x0_t, y0_t = vx_t.item(), vy_t.item(), x0_t.item(), y0_t.item()
    vx_b, vy_b, x0_b, y0_b = vx_b.item(), vy_b.item(), x0_b.item(), y0_b.item()

    if show_detail and image is not None:
        # Draw fitted lines extended for visualization
        vis_fit = image.copy()
        h, w = image.shape[:2]
        def draw_line(p, v, color):
            # extend line to image borders
            t0, t1 = -1000, 1000
            pt0 = (int(p[0] + t0*v[0]), int(p[1] + t0*v[1]))
            pt1 = (int(p[0] + t1*v[0]), int(p[1] + t1*v[1]))
            cv2.line(vis_fit, pt0, pt1, color, 2)

        draw_line([x0_l, y0_l], [vx_l, vy_l], (255,0,0))
        draw_line([x0_r, y0_r], [vx_r, vy_r], (255,0,0))
        draw_line([x0_t, y0_t], [vx_t, vy_t], (0,0,255))
        draw_line([x0_b, y0_b], [vx_b, vy_b], (0,0,255))
        plt.imshow(vis_fit)
        plt.title("Fitted Extremal Lines")
        plt.axis('off')
        plt.show()

    # --- Step 5: compute intersections ---
    def line_intersection(p1, v1, p2, v2):
        x0_1, y0_1 = p1
        vx1, vy1 = v1
        x0_2, y0_2 = p2
        vx2, vy2 = v2
        A = np.array([[vx1, -vx2], [vy1, -vy2]])
        b = np.array([x0_2 - x0_1, y0_2 - y0_1])
        if np.linalg.cond(A) > 1e12:
            return np.array([(x0_1+x0_2)/2, (y0_1+y0_2)/2])
        t = np.linalg.lstsq(A, b, rcond=None)[0]
        intersection = np.array([x0_1, y0_1]) + t[0]*np.array([vx1, vy1])
        return intersection

    p_l, v_l = [x0_l, y0_l], [vx_l, vy_l]
    p_r, v_r = [x0_r, y0_r], [vx_r, vy_r]
    p_t, v_t = [x0_t, y0_t], [vx_t, vy_t]
    p_b, v_b = [x0_b, y0_b], [vx_b, vy_b]

    corners = np.array([
        line_intersection(p_l, v_l, p_t, v_t),  # top-left
        line_intersection(p_r, v_r, p_t, v_t),  # top-right
        line_intersection(p_r, v_r, p_b, v_b),  # bottom-right
        line_intersection(p_l, v_l, p_b, v_b),  # bottom-left
    ], dtype=np.int32)

    if show_detail and image is not None:
        vis_corners = image.copy()
        for i, (x, y) in enumerate(corners):
            cv2.circle(vis_corners, (int(x), int(y)), 8, (0,0,255), -1)
            cv2.putText(vis_corners, f"{i+1}", (int(x)+5,int(y)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        plt.imshow(vis_corners)
        plt.title("Board Corners from Intersections")
        plt.axis('off')
        plt.show()

    # now create the mask with the board area
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [corners], 1)

    if (show or show_detail):
        plt.imshow(mask, cmap='gray')
        plt.show()

    return mask


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

    plt.imshow(refined_mask, cmap='gray')
    plt.show()

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


# def detect_pieces(image, depth_map, plane_model, board_mask, show=True):
#     """
#     Detect chess pieces on a board using K-means clustering on height residuals above the plane.
#     """
#
#     # --- Step 1: compute height residuals from the plane ---
#     height_residual = subtract_plane(depth_map, plane_model)
#     height_residual[board_mask == 0] = 1  # ignore areas outside the board
#
#     # --- Step 2: K-means clustering to separate board vs pieces ---
#     board_pixels = height_residual[board_mask > 0].reshape(-1, 1)
#     kmeans = KMeans(n_clusters=2, random_state=42).fit(board_pixels)
#     labels = kmeans.labels_
#     centers = kmeans.cluster_centers_
#
#     # cluster with lower center corresponds to pieces
#     piece_cluster = np.argmin(centers)
#
#     # create full-size mask
#     mask = np.zeros_like(height_residual, dtype=np.uint8)
#     board_indices = np.where(board_mask > 0)
#     mask[board_indices[0], board_indices[1]] = (labels == piece_cluster).astype(np.uint8)
#
#     # --- Step 3: Morphological cleanup ---
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
#
#     # --- Step 4: connected components and area rejection ---
#     num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(mask)
#     print(f"Detected {num_labels-1} pieces via K-means clustering.")
#
#     height, width = image.shape[:2]
#     area_reject_percentage = 5  # reject pieces >5% of image area
#     area_reject_threshold = (height * width / 100) * area_reject_percentage
#     min_area = 5
#
#     display_image = image.copy()
#     for i in range(1, num_labels):  # skip background
#         x, y, w, h, area = stats[i]
#
#         if area > area_reject_threshold or area < min_area:
#             continue
#
#         cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     # --- Step 5: visualization ---
#     if show:
#         fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#
#         im0 = axes[0].imshow(height_residual, cmap='turbo')
#         axes[0].set_title("Height residuals (above plane)")
#         plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
#
#         axes[1].imshow(mask, cmap='gray')
#         axes[1].set_title("K-means binary mask (pieces)")
#
#         axes[2].imshow(display_image[..., ::-1])
#         axes[2].set_title(f"Detected pieces ({num_labels-1})")
#
#         for ax in axes:
#             ax.axis('off')
#
#         plt.tight_layout()
#         plt.show()
#
#     return mask

def load_data(image_dir, depth_dir):

    image_names = os.listdir(image_dir) 

    image_depth_pairs = []

    for name in image_names:
        base_name = os.path.splitext(name)[0]

        depth_name = f"{base_name}.npz"

        image_path = os.path.join(image_dir, name)
        depth_path = os.path.join(depth_dir, depth_name)

        if not os.path.exists(depth_path):
            raise RuntimeError(f"depth map not found {depth_path}")

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path)['depth']

        image_depth_pairs.append((image, depth))

    return image_depth_pairs


# def segment_with_sam(image, centroids, show_each=False):
#      
#     sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_MODEL_PATH)
#
#     device = 'cpu'
#     if torch.cuda.is_available():
#         device = 'cuda'
#
#     sam.to(device)
#     predictor = SamPredictor(sam)
#
#     predictor.set_image(image)
#
#     height, width = image.shape[:2]
#     combined_mask = np.zeros((height, width), dtype=np.uint8)
#
#     leaf_masks = []
#
#     # segment each leaf using the centroids
#     for i, points in enumerate(centroids):
#
#         points = np.array(points, dtype=np.float32)
#
#         # input_point = np.array((points), dtype=np.float32)
#         input_label = np.array([1] * len(points)) # foreground point
#
#         masks, scores, _ = predictor.predict(
#             point_coords=points,
#             point_labels=input_label,
#             multimask_output=True
#         )
#
#         # choose the smallest mask
#         mask_areas = masks.sum(axis=(1, 2))
#         # biggest_idx = np.argmax(mask_areas)
#         # biggest_mask = masks[biggest_idx]
#         best_idx = np.argmax(scores)
#         best_mask = masks[best_idx]
#
#         selected_mask = best_mask
#         selected_idx = best_idx
#
#         # remove large masks that could just be the background or entire plants
#         if mask_areas[selected_idx] > 0.2 * height * width:
#             continue
#
#         # combine the masks
#         combined_mask[selected_mask] = i
#
#         # save the leaf mask
#         leaf_masks.append(selected_mask)
#
#         # --- Visualization for this centroid ---
#         if show_each:
#             overlay = image.copy().astype(np.float32) / 255.0
#             mask_color = np.zeros_like(overlay)
#             mask_color[..., 0] = 1.0  # red overlay
#
#             alpha = 0.5
#             overlay = np.where(selected_mask[..., None], 
#                                (1 - alpha) * overlay + alpha * mask_color, 
#                                overlay)
#
#             plt.figure(figsize=(6, 6))
#             plt.imshow(overlay)
#             plt.scatter(points[:, 0], points[:, 1], c='cyan', s=40, edgecolors='black')
#             plt.title(f"Centroid {i+1}: Score={scores[selected_idx]:.3f}")
#             plt.axis('off')
#             plt.show()
#
#     return combined_mask, leaf_masks
#

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
    leaf_masks = []

    for i, pos_points in enumerate(centroids):
        # --- Build positive & negative prompts ---
        pos_points = np.array(pos_points, dtype=np.float32)


        # neg_points = np.concatenate(
        #     [np.array(pts, dtype=np.float32) for j, pts in enumerate(centroids) if j != i],
        #     axis=0
        # ) if len(centroids) > 1 else np.empty((0, 2), dtype=np.float32)

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

        # choose the biggest mask
        # mask_areas = masks.sum(axis=(1, 2))
        # biggest_idx = np.argmax(mask_areas)
        # biggest_mask = masks[biggest_idx]


        # all points for other pieces
        other_piece_points = np.concatenate(
            [np.array(pts, dtype=np.float32) for j, pts in enumerate(centroids) if j != i],
            axis=0
        ) if len(centroids) > 1 else np.empty((0, 2), dtype=np.float32)

        valid_masks = []
        for m in masks:

            if mask_contains_any_points(m, other_piece_points):
                continue
            
            # check for one connected component (1 contour)
            mask_uint8 = (m.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) != 1:
                continue

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

        # skip huge masks
        if mask_area > 0.2 * height * width:
            continue

        combined_mask[selected_mask] = i + 1
        leaf_masks.append(selected_mask)

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

    return combined_mask, leaf_masks

def main():

    data = load_data(IMAGE_DIRECTORY, DEPTH_MAP_DIRECTORY)

    for i in range(len(data)):
        image = data[i][0]
        depth = data[i][1]

        # print(threshold_util(np.array([image])))

        board_mask = get_board_area(image, show=False, show_detail=False)
        # apply the mask to the depth map
        # depth = depth * board_mask

        plane_model, inliers = segment_board_plane(depth, board_mask)
        centroids = detect_pieces(image, depth, plane_model, board_mask, show=True)

        segmentation_mask, piece_masks = segment_with_sam(image, centroids, show_each=False)

        plot_segmentation_mask(image, segmentation_mask)

if __name__ == "__main__":
    main()
