

def get_piece_colour(image, piece_mask):
    # Ensure mask is boolean
    mask = piece_mask.astype(bool)

    # Compute mean color in RGB (or BGR if OpenCV)
    mean_color = image[mask].mean(axis=0)

    # Convert to perceived brightness (luminance)
    brightness = 0.2126*mean_color[2] + 0.7152*mean_color[1] + 0.0722*mean_color[0]

    # Threshold to classify
    return "W" if brightness > 128 else "B"
