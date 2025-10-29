import os
import cv2
import numpy as np
from matplotlib import cm

def load_images(file_path):

    images = []

    image_names = os.listdir(file_path)

    for name in image_names:
        path = os.path.join(file_path, name)

        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        images.append(image)

    return images


def load_image_depth_pairs(image_dir, depth_dir):

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

        image_depth_pairs.append((image, depth, base_name))

    return image_depth_pairs


def save_segmentations_to_file(output_dir, name, masks, image):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the raw integer mask
    save_path = os.path.join(output_dir, f"{name}_mask.png")
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

    # Convert RGB to BGR for OpenCV
    colour_mask_bgr = cv2.cvtColor(colour_mask, cv2.COLOR_RGB2BGR)

    # Blend overlay
    overlay = image.copy().astype(np.float32) / 255.0
    colour_mask_float = colour_mask_bgr.astype(np.float32) / 255.0
    alpha = 0.5
    blended = cv2.addWeighted(overlay, 1 - alpha, colour_mask_float, alpha, 0)

    colour_path = os.path.join(output_dir, f"{name}_mask_coloured.png")
    cv2.imwrite(colour_path, (blended * 255).astype(np.uint8))


