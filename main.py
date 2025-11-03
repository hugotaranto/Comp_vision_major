from detection.piece_detect import get_piece_segments, load_sam
from detection.util import *
from detection.plots import *
from detection.depth import load_model
from detection.pose_detect import detect_poses

from Piece_Labeling_and_Classification.inference import predict_pieces_from_semantic
from chess_board_reconstructor import draw_chess_board


IMAGE_DIRECTORY = './side_test'
MASK_OUTPUT_DIRECTORY = './detection/detection_output'

SAM_MODEL_TYPE = 'vit_l'
SAM_MODEL_PATH = './sam_checkpoints/sam_vit_l_0b3195.pth'

DEPTH_PRO_CHECKPOINT_PATH = './ml-depth-pro/checkpoints/depth_pro.pt'

CLASSIFIER_MODEL_PATH = './Piece_Labeling_and_Classification/semantic_chess_classifier.pkl'

MAX_DIM = 1600


        # 'WK': 'king.png',
        # 'WQ': 'queen.png', 
        # 'WR': 'rook.png',
        # 'WB': 'bishop.png',
        # 'WN': 'knight.png',
        # 'WP': 'pawn.png',
        # 
        # # Black pieces (with 'b' prefix)
        # 'BK': 'bking.png',
        # 'BQ': 'bqueen.png',
        # 'BR': 'brook.png', 
        # 'BB': 'bbishop.png',
        # 'BN': 'bknight.png',
        # 'BP': 'bpawn.png'

piece_translate = {
    'rook': 'R',
    'pawn': 'P',
    'king': 'K',
    'queen': 'Q',
    'knight': 'N',
    'bishop': 'B'
}

def main():
    images, f_pxs, names = load_images(IMAGE_DIRECTORY)

    print("loading depth model")
    depth_model, depth_transform = load_model(DEPTH_PRO_CHECKPOINT_PATH)

    print("Loading SAM")
    sam_predictor = load_sam(SAM_MODEL_PATH)

    for i in range(len(names)):
        image = images[i]
        f_px = f_pxs[i]
        name = names[i]

        segmentation_mask, image_resized, corners = get_piece_segments(image, f_px, name, sam_predictor, depth_model, depth_transform)

        # plot_segmentation_mask(image_resized, segmentation_mask)

        board, board_colours = detect_poses(segmentation_mask, corners, show=True, show_each=False, image=image_resized)

        print("Classifying pieces")
        pred = predict_pieces_from_semantic(segmentation_mask, CLASSIFIER_MODEL_PATH)

        # Create final labelled board (strings like 'WK', 'BP', etc.)
        labelled_board = np.full((8, 8), '', dtype=object)

        for r in range(8):
            for c in range(8):
                piece_id = board[r, c]
                if piece_id != 0:
                    colour = board_colours[r, c]
                    piece_type = pred[piece_id]['type']  # e.g., 'K', 'Q', etc.
                    labelled_board[r, c] = f"{colour}{piece_translate[piece_type]}"

        draw_chess_board(labelled_board)

        # go through the pieces and 

    

if __name__ == "__main__":
    main()
