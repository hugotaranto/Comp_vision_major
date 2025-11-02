import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import List

DPI = 100

def display_depth(depth, cmap="plasma"):
    plt.imshow(depth, cmap=cmap)
    plt.axis("off")
    plt.show()

def plot_image_and_depth(image, depth, title=None, cmap="plasma"):
    """
    Display an RGB image and its depth map side by side.

    Parameters:
        image (np.ndarray): The RGB image (H, W, 3).
        depth (np.ndarray): The depth map (H, W).
        title (str, optional): Optional title for the figure.
        cmap (str, optional): Colormap for depth visualization (default: 'plasma').
    """
    # ensure proper types
    image = np.asarray(image)
    depth = np.asarray(depth)

    # create the figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # plot RGB image
    axes[0].imshow(image)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")

    # plot depth map
    im = axes[1].imshow(depth, cmap=cmap)
    axes[1].set_title("Depth Map")
    axes[1].axis("off")

    # add colorbar for depth
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # optional overall title
    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_depth_with_clusters(depth_map, labels_2d, cluster_centroids, cmap='plasma'):
    """
    Plot depth map with DBSCAN clusters and centroids overlaid.

    Parameters:
        depth_map (np.ndarray): 2D depth map (H, W)
        labels_2d (np.ndarray): 2D array of cluster labels (-1 = noise)
        cluster_centroids (list of tuples): list of (x, y) coordinates for cluster centers
        cmap (str): colormap for depth map
    """
    plt.figure(figsize=(10, 8))
    
    # Show depth map
    plt.imshow(depth_map, cmap=cmap)
    
    # Overlay clusters (semi-transparent mask)
    mask = labels_2d > -1
    plt.imshow(mask, alpha=0.3, cmap='cool')  # highlight all detected clusters
    
    # Overlay centroids
    for x, y in cluster_centroids:
        plt.scatter(x, y, color='cyan', s=50, edgecolors='black', linewidth=1)
    
    plt.title(f"Detected {len(cluster_centroids)} clusters")
    plt.axis('off')
    plt.show()


def show_dbscan_clusters(depth_map, filtered_xy, labels, image, depth, centroids, orig_centroids):
    """
    depth_map: 2D array (for grayscale display)
    filtered_xy: (N, 2) array of (x, y) coords used in clustering
    labels: DBSCAN cluster labels for each filtered point
    image: original RGB image (for context)
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    # --- Left: original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[0].scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, marker='x')

    # --- Right: depth map with colorbar
    im = axes[2].imshow(depth, cmap="plasma")
    axes[2].set_title("Depth Map")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label='Depth')

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])

    axes[1].imshow(depth_map, cmap='gray')
    axes[1].set_title(f"DBSCAN Clusters ({n_clusters} clusters)")
    axes[1].axis('off')

    for label in unique_labels:
        if label == -1:
            cluster_color = 'lightgray'
            size = 4
        else:
            cluster_color = plt.cm.tab10(label % 10)
            size = 15

        cluster_points = filtered_xy[labels == label]
        axes[1].scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=size,
            c=[cluster_color],
            alpha=0.7,
            edgecolors='none'
        )

    axes[1].scatter(orig_centroids[:, 0], orig_centroids[:, 1], c='red', s=50, marker='x')

    plt.tight_layout()
    plt.show()


def plot_segmentation_mask(image, mask):

    width, height = image.shape[:2]

    combined_mask = np.ma.masked_where(mask == 0, mask) # mask out the zero values

    # === Visualise with matplot === This is quite slow (takes majority of the runtime), could definitely be done faster using cv2 (or equivalent) directly
    fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)
    plt.imshow(image)
    plt.imshow(combined_mask, alpha=0.5, cmap='tab10')
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()


def visualise_top_leaves(image, leaf_segmentations, scores):

    vis = image.copy()

    colours = plt.cm.viridis(np.linspace(0, 1, len(scores)))[:, :3] * 255

    for i in range(len(scores)):
        mask = leaf_segmentations[i].astype(np.uint8)
        score = scores[i]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color = tuple(int(c) for c in colours[i])
        cv2.drawContours(vis, contours, -1, color, 2)

        # find centroid for text label
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            label = f"#{i+1}: {score:.2f}"
            cv2.putText(vis, label, (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    plt.figure(figsize=(10, 10))
    plt.imshow(vis)
    plt.title("Top Ranked Leaves by Visibility Score")
    plt.axis("off")
    plt.show()

def nothing(x): pass

def threshold_util(images:np.ndarray, thresh_vals:List[int] = [0, 155, 40, 78, 255, 255]) -> List[int]:

    lh = thresh_vals[0]
    ls = thresh_vals[1]
    lv = thresh_vals[2]
    uh = thresh_vals[3]
    us = thresh_vals[4]
    uv = thresh_vals[5]

    for image in images:

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Create resizable windows
        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Mask", 800, 600)

        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Result", 800, 600)

        cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Trackbars", 400, 300)


        # Create sliders for HSV thresholds
        cv2.createTrackbar("LH", "Trackbars", lh, 179, nothing)
        cv2.createTrackbar("LS", "Trackbars", ls, 255, nothing)
        cv2.createTrackbar("LV", "Trackbars", lv, 255, nothing)
        cv2.createTrackbar("UH", "Trackbars", uh, 179, nothing)
        cv2.createTrackbar("US", "Trackbars", us, 255, nothing)
        cv2.createTrackbar("UV", "Trackbars", uv, 255, nothing)

        while True:
            # Get current positions of sliders
            lh = cv2.getTrackbarPos("LH", "Trackbars")
            ls = cv2.getTrackbarPos("LS", "Trackbars")
            lv = cv2.getTrackbarPos("LV", "Trackbars")
            uh = cv2.getTrackbarPos("UH", "Trackbars")
            us = cv2.getTrackbarPos("US", "Trackbars")
            uv = cv2.getTrackbarPos("UV", "Trackbars")

            lower = (lh, ls, lv)
            upper = (uh, us, uv)

            # Create mask
            mask = cv2.inRange(hsv, lower, upper)

            # Apply mask
            result = cv2.bitwise_and(bgr_img, bgr_img, mask=mask)

            # Downscale for display if needed (optional)
            scale = 0.5
            mask_small = cv2.resize(mask, (0, 0), fx=scale, fy=scale)
            result_small = cv2.resize(result, (0, 0), fx=scale, fy=scale)

            cv2.imshow("Mask", mask_small)
            cv2.imshow("Result", result_small)

            # Press ESC to exit
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()

    return [lh, ls, lv, uh, us, uv]


