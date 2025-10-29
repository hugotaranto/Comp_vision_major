import os
import cv2
import numpy as np

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

        image_depth_pairs.append((image, depth))

    return image_depth_pairs

