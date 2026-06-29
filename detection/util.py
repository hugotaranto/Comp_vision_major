import os
import cv2
import numpy as np
from matplotlib import cm
from depth_pro import load_rgb
import matplotlib.pyplot as plt

def resize_image(image, target_w, expanded_corners, corners, show=False):
    # --- crop the image ---
    x_min = expanded_corners[:,0].min()
    x_max = expanded_corners[:,0].max()
    y_min = expanded_corners[:,1].min()
    y_max = expanded_corners[:,1].max()

    cropped = image[y_min:y_max, x_min:x_max]

    # shift corners to cropped image coordinates
    cropped_board_corners = corners - np.array([x_min, y_min])

    cropped_h, cropped_w = cropped.shape[:2]

    # target height for 4:3 ratio
    target_h = round(target_w * 3 / 4)

    # compute scale to fit inside 4:3 while keeping aspect ratio
    scale = min(target_w / cropped_w, target_h / cropped_h)
    new_w = int(cropped_w * scale)
    new_h = int(cropped_h * scale)

    # resize image
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # scale corners to match resized image
    resized_board_corners = cropped_board_corners * scale

    if show:
        vis_corners = resized.copy()
        for i, (x, y) in enumerate(resized_board_corners):
            cv2.circle(vis_corners, (int(x), int(y)), 20, (0,0,255), -1)
            cv2.putText(vis_corners, f"{i+1}", (int(x)+5,int(y)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        plt.figure(figsize=(15, 15))
        plt.imshow(vis_corners)
        plt.title("Board Corners in Resized Image")
        plt.axis('off')
        plt.show()

    return resized, resized_board_corners


def pad_to_target(image, target_w, corners=None, pad_value=0):

    target_h = round(target_w * 3 / 4)
    h, w = image.shape[:2]

    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left

    # Handle both 2D and 3D arrays
    if image.ndim == 3:
        value = [pad_value] * image.shape[2]
    else:
        value = pad_value

    padded_image = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=value
    )

    if corners is not None:
        padded_board_corners = corners + np.array([pad_left, pad_top])
        return padded_image, padded_board_corners

    return padded_image



def load_images(file_path):
    images = []
    f_pxs = []
    names = []

    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_names = [f for f in os.listdir(file_path)
                   if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]

    for name in image_names:
        path = os.path.join(file_path, name)
        print(f"Loading Image: {path}")

        image, _, f_px = load_rgb(path)

        images.append(image)
        f_pxs.append(f_px)
        names.append(os.path.splitext(name)[0])


    return images, f_pxs, names


def load_image_depth_pairs(image_dir, depth_dir, type="depth-pro"):

    image_names = os.listdir(image_dir) 

    image_depth_pairs = []

    for name in image_names:
        base_name = os.path.splitext(name)[0]

        if type == "depth-pro":
            depth_name = f"{base_name}.npz"
        elif type == "marigold":
            depth_name = f"{base_name}_depth.npy"
        else:
            raise RuntimeError(f"depth model type {type} not supported")

        image_path = os.path.join(image_dir, name)
        depth_path = os.path.join(depth_dir, depth_name)

        if not os.path.exists(depth_path):
            print(f"depth map not found {depth_path}")
            continue

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        if type == "depth-pro":
            depth = np.load(depth_path)['depth']
        elif type == "marigold":
            depth = np.load(depth_path).astype(np.float32)

        image_depth_pairs.append((image, depth, base_name))

    return image_depth_pairs


def save_segmentations_to_file(output_dir, name, masks, image):
    os.makedirs(output_dir, exist_ok=True)

    semantics_dir = os.path.join(output_dir, "semantics")
    os.makedirs(semantics_dir, exist_ok=True)

    display_dir = os.path.join(output_dir, "display")
    os.makedirs(display_dir, exist_ok=True)

    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    image_path = os.path.join(image_dir, f"{name}.png")
    cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Save the raw integer mask
    save_path = os.path.join(semantics_dir, f"{name}_mask.png")
    print(f"saved masks to {save_path}")
    cv2.imwrite(save_path, masks.astype(np.uint8))

    # Create colorized version
    colour_mask = np.zeros((*masks.shape, 3), dtype=np.uint8)  # H x W x 3

    # Apply colormap only to non-zero mask values
    nonzero = masks > 0
    if np.any(nonzero):
        cmap = cm.get_cmap('tab20')
        normalized_vals = masks[nonzero] / (masks.max() + 1e-6)
        colours = (cmap(normalized_vals)[:, :3] * 255).astype(np.uint8)
        colour_mask[nonzero] = colours

    # Blend overlay in RGB
    overlay = image.copy().astype(np.float32) / 255.0
    colour_mask_float = colour_mask.astype(np.float32) / 255.0
    alpha = 0.5
    blended = cv2.addWeighted(overlay, 1 - alpha, colour_mask_float, alpha, 0)

    colour_path = os.path.join(display_dir, f"{name}_mask_coloured.png")
    cv2.imwrite(colour_path, cv2.cvtColor((blended * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


