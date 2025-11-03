import cv2
import numpy as np
import matplotlib.pyplot as plt

import cv2
import numpy as np
import matplotlib.pyplot as plt

from Piece_Labeling_and_Classification.colour_detect import get_piece_colour

def detect_poses(segmentation_mask, corners, image, show=False, show_each=False):
    corners = np.array(corners, dtype=np.float32)

    # Destination points for top-down view
    board_size = 800
    dst_pts = np.array([
        [0, 0],
        [board_size - 1, 0],
        [board_size - 1, board_size - 1],
        [0, board_size - 1]
    ], dtype=np.float32)

    # Perspective transform
    M = cv2.getPerspectiveTransform(corners, dst_pts)
    warped_mask = cv2.warpPerspective(segmentation_mask, M, (board_size, board_size), flags=cv2.INTER_NEAREST)
    square_size = board_size // 8
    board = np.zeros((8, 8), dtype=int)
    board_colours = np.full((8, 8), '', dtype=object)

    # If requested, warp the original image
    warped_image = cv2.warpPerspective(image, M, (board_size, board_size))

    # ---- Process each piece ----
    piece_ids = np.unique(warped_mask)
    piece_ids = piece_ids[piece_ids != 0]

    for piece_id in piece_ids:
        ys, xs = np.where(warped_mask == piece_id)
        if len(xs) == 0:
            continue

        # Bottom 50% of the piece
        y_min, y_max = ys.min(), ys.max()
        bottom_half_mask = ys >= (y_min + (y_max - y_min) // 2)
        xs_bottom = xs[bottom_half_mask]
        ys_bottom = ys[bottom_half_mask]

        # Assign to board
        square_rows = ys_bottom // square_size
        square_cols = xs_bottom // square_size
        squares, counts = np.unique(np.stack([square_rows, square_cols], axis=1), axis=0, return_counts=True)
        r, c = squares[np.argmax(counts)]
        board[r, c] = piece_id

        # detect the piece colour
        piece_mask = (warped_mask == piece_id).astype(np.uint8)
        colour = get_piece_colour(warped_image, piece_mask)
        board_colours[r, c] = colour

        # ---- If show_each, visualize this piece’s bottom half ----
        if show_each and warped_image is not None:
            mask_bottom = np.zeros_like(warped_mask)
            mask_bottom[ys_bottom, xs_bottom] = piece_id

            # Draw gridlines
            image_with_grid = warped_image.copy()
            mask_color = cv2.cvtColor((mask_bottom > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
            for i in range(1, 8):
                cv2.line(image_with_grid, (0, i*square_size), (board_size, i*square_size), (255, 0, 0), 2)
                cv2.line(image_with_grid, (i*square_size, 0), (i*square_size, board_size), (255, 0, 0), 2)
                cv2.line(mask_color, (0, i*square_size), (board_size, i*square_size), (0, 0, 255), 2)
                cv2.line(mask_color, (i*square_size, 0), (i*square_size, board_size), (0, 0, 255), 2)

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title(f"Piece {piece_id} — Bottom Half Mask")
            plt.imshow(mask_color)
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title(f"Warped Image with Grid (Piece {piece_id})")
            plt.imshow(cv2.cvtColor(image_with_grid, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()

    # ---- Print board ----
    if show:
        print("Detected board positions (8x8):")
        for row in board:
            print(" ".join(map(str, row)))


    # ---- Visualization: show warped mask + image side-by-side ----
    if show and warped_image is not None:
        # Draw grid on copies
        image_with_grid = warped_image.copy()
        mask_color = cv2.cvtColor((warped_mask > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        for i in range(1, 8):
            cv2.line(image_with_grid, (0, i*square_size), (board_size, i*square_size), (255, 0, 0), 2)
            cv2.line(image_with_grid, (i*square_size, 0), (i*square_size, board_size), (255, 0, 0), 2)
            cv2.line(mask_color, (0, i*square_size), (board_size, i*square_size), (0, 0, 255), 2)
            cv2.line(mask_color, (i*square_size, 0), (i*square_size, board_size), (0, 0, 255), 2)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Warped Mask with Grid")
        plt.imshow(mask_color)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Warped Image with Grid")
        plt.imshow(cv2.cvtColor(image_with_grid, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return board, board_colours

