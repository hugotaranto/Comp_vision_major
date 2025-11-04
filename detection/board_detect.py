import matplotlib.pyplot as plt
import numpy as np
import cv2

from sklearn.cluster import AgglomerativeClustering

from .util import *
from .plots import *

IMAGE_DIR = '../side_test'

# LOWER_GREEN_HSV_THRESHOLD = [35, 85, 45]
LOWER_GREEN_HSV_THRESHOLD = [35, 55, 40]
UPPER_GREEN_HSV_THRESHOLD = [85, 255, 255]


# [35, 55, 40, 85, 255, 255]

def show_lines_with_midpoints(img, lines_pts, color=(255,0,0), radius=5):
    vis = img.copy()
    num_lines = len(lines_pts) // 2
    for i in range(num_lines):
        pt1 = tuple(lines_pts[2*i])
        pt2 = tuple(lines_pts[2*i+1])
        cv2.line(vis, pt1, pt2, color, 2)
        mid = ((pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2)
        cv2.circle(vis, mid, radius, (0,255,0), -1)  # midpoint in green
    plt.figure(figsize=(10,10))
    plt.imshow(vis)
    plt.title("Lines with Midpoints")
    plt.axis('off')
    plt.show()

def average_extremal_lines(lines, mean_coords, n=4, side='left'):
    if side =='left' or side == 'top':
        idxs = np.argsort(mean_coords)[:n]
    else:
        idxs = np.argsort(mean_coords)[-n:]

    pts = lines[idxs].reshape(-1, 2)

    return pts

def get_board_area(image, show=False, show_detail=False):
    # --- Step 1: create green mask ---
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask_green = cv2.inRange(hsv, np.array(LOWER_GREEN_HSV_THRESHOLD),
                                     np.array(UPPER_GREEN_HSV_THRESHOLD))

    if show_detail:
        plt.imshow(mask_green, cmap='gray')
        plt.title("Green masked area")
        plt.axis("off")
        plt.show()

    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, np.ones((10,10), np.uint8))
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    if show_detail:
        plt.imshow(mask_green, cmap='gray')
        plt.title("Small contours removed")
        plt.axis("off")
        plt.show()

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
    edges = cv2.Canny(mask_blur_norm, 100, 200, apertureSize=3)
    if show_detail:
        plt.imshow(edges, cmap='gray')
        plt.title("Canny Edges from Green Mask")
        plt.axis('off')
        plt.show()

    # --- Step 3: Hough line detection ---
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=70 , minLineLength=150, maxLineGap=1000)
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

    # --- Step 2: cluster lines by angle ---
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2((y2 - y1), (x2 - x1))  # angle in radians
        angles.append(angle)
    angles = np.array(angles)

    points = np.column_stack((np.cos(2*angles), np.sin(2*angles)))

    cluster = AgglomerativeClustering(n_clusters=2)
    labels = cluster.fit_predict(points)

    if show_detail:
        plt.figure(figsize=(6,6))
        plt.scatter(points[:,0], points[:,1], c=labels)
        plt.xlabel('cos(angle)')
        plt.ylabel('sin(angle)')
        plt.title('Clustering lines by angle on the unit circle')
        plt.axis('equal')
        plt.show()

    # Now separate
    lines_group0 = lines[labels == 0]
    lines_group1 = lines[labels == 1]

    # Determine which group is vertical and which is horizontal
    mean_angle0 = np.mean(np.abs(np.sin([np.arctan2(y2-y1, x2-x1) for x1,y1,x2,y2 in lines_group0[:,0]])))
    mean_angle1 = np.mean(np.abs(np.sin([np.arctan2(y2-y1, x2-x1) for x1,y1,x2,y2 in lines_group1[:,0]])))

    if mean_angle0 > mean_angle1:
        vertical_lines = lines_group0
        horizontal_lines = lines_group1
    else:
        vertical_lines = lines_group1
        horizontal_lines = lines_group0

    if show_detail:
        vis_lines = image.copy()

        # Draw group 0 in red
        for line in vertical_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(vis_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw group 1 in blue
        for line in horizontal_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(vis_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

        plt.figure(figsize=(10, 10))
        plt.imshow(vis_lines)
        plt.title("Line Groups after Clustering, Red = vertical, blue = horizontal")
        plt.axis('off')
        plt.show()


    if len(vertical_lines) < 2 or len(horizontal_lines) < 2:
        print("Not enough vertical/horizontal lines")
        return None

    # --- Step 3: find extremal lines ---

    # For verticals
    vertical_mean_x = vertical_lines[:, 0, [0, 2]].mean(axis=1)
    left_line_pts  = average_extremal_lines(vertical_lines, vertical_mean_x, n=1, side='left')
    right_line_pts = average_extremal_lines(vertical_lines, vertical_mean_x, n=1, side='right')

    # For horizontals
    horizontal_mean_y = horizontal_lines[:, 0, [1, 3]].mean(axis=1)
    top_line_pts    = average_extremal_lines(horizontal_lines, horizontal_mean_y, n=1, side='top')
    bottom_line_pts = average_extremal_lines(horizontal_lines, horizontal_mean_y, n=1, side='bottom')

    if show_detail:
        show_lines_with_midpoints(image, right_line_pts, color=(255,0,0))
        # show_lines_with_midpoints(image, horizontal_lines, color=(0,0,255))


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

    if (show_detail or show) and image is not None:
        vis_corners = image.copy()
        for i, (x, y) in enumerate(corners):
            cv2.circle(vis_corners, (int(x), int(y)), 80, (0,0,255), -1)
            cv2.putText(vis_corners, f"{i+1}", (int(x)+5,int(y)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        plt.figure(figsize=(15, 15))
        plt.imshow(vis_corners)
        plt.title("Board Corners from Intersections")
        plt.axis('off')
        plt.show()

    # now create the mask with the board area
    # mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # cv2.fillPoly(mask, [corners], 1)

    # return mask
    return corners


if __name__ == "__main__":

    images, _, _ = load_images(IMAGE_DIR)

    for image in images:
        # get_board_area(image, show_detail=True)
        get_board_area(image, show=True)
        # threshold_util(np.array([image]), [35, 55, 40, 85, 255, 255])


# LOWER_GREEN_HSV_THRESHOLD = [35, 55, 40]
# UPPER_GREEN_HSV_THRESHOLD = [85, 255, 255]


