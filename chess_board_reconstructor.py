"""
2D Chess Board Reconstructor
===========================

Creates a visual chess board similar to Chess.com layout with green and white squares.
Shows the starting position of all chess pieces.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
from PIL import Image
from scipy import ndimage

def create_chess_board_layout():
    """
    Create the standard chess starting position.
    Returns a 8x8 array with piece codes.
    """
    # Empty board
    board = np.full((8, 8), '', dtype=object)
    
    # Starting position mapping
    starting_position = {
        # White pieces (bottom rows)
        (7, 0): 'WR', (7, 1): 'WN', (7, 2): 'WB', (7, 3): 'WQ',
        (7, 4): 'WK', (7, 5): 'WB', (7, 6): 'WN', (7, 7): 'WR',
        # White pawns
        (6, 0): 'WP', (6, 1): 'WP', (6, 2): 'WP', (6, 3): 'WP',
        (6, 4): 'WP', (6, 5): 'WP', (6, 6): 'WP', (6, 7): 'WP',
        
        # Black pieces (top rows)
        (0, 0): 'BR', (0, 1): 'BN', (0, 2): 'BB', (0, 3): 'BQ',
        (0, 4): 'BK', (0, 5): 'BB', (0, 6): 'BN', (0, 7): 'BR',
        # Black pawns
        (1, 0): 'BP', (1, 1): 'BP', (1, 2): 'BP', (1, 3): 'BP',
        (1, 4): 'BP', (1, 5): 'BP', (1, 6): 'BP', (1, 7): 'BP',
    }
    
    # Fill the board
    for (row, col), piece in starting_position.items():
        board[row, col] = piece
    
    return board

def get_piece_unicode(piece_code):
    """
    Convert piece codes to Unicode chess symbols.
    
    Args:
        piece_code (str): Two character code like 'WK' (White King), 'BP' (Black Pawn)
    
    Returns:
        str: Unicode chess symbol
    """
    piece_symbols = {
        # White pieces
        'WK': '♔',  # White King
        'WQ': '♕',  # White Queen  
        'WR': '♖',  # White Rook
        'WB': '♗',  # White Bishop
        'WN': '♘',  # White Knight
        'WP': '♙',  # White Pawn
        
        # Black pieces
        'BK': '♚',  # Black King
        'BQ': '♛',  # Black Queen
        'BR': '♜',  # Black Rook
        'BB': '♝',  # Black Bishop
        'BN': '♞',  # Black Knight
        'BP': '♟',  # Black Pawn
    }
    
    return piece_symbols.get(piece_code, '')

def load_piece_images(pieces_folder="2d_chess_pieces_images"):
    """
    Load chess piece images from a folder.
    
    Expected file names (user's naming convention):
    - king.png, queen.png, rook.png, bishop.png, knight.png, pawn.png (White pieces)  
    - bking.png, bqueen.png, brook.png, bbishop.png, bknight.png, bpawn.png (Black pieces)
    
    Args:
        pieces_folder (str): Path to folder containing piece images
        
    Returns:
        dict: Dictionary mapping piece codes to loaded images
    """
    piece_images = {}
    
    if not os.path.exists(pieces_folder):
        print(f"⚠️  Piece images folder not found: {pieces_folder}")
        print("Using Unicode symbols as fallback")
        return None
    
    # Mapping from internal codes to user's file names
    piece_file_mapping = {
        # White pieces (no prefix)
        'WK': 'king.png',
        'WQ': 'queen.png', 
        'WR': 'rook.png',
        'WB': 'bishop.png',
        'WN': 'knight.png',
        'WP': 'pawn.png',
        
        # Black pieces (with 'b' prefix)
        'BK': 'bking.png',
        'BQ': 'bqueen.png',
        'BR': 'brook.png', 
        'BB': 'bbishop.png',
        'BN': 'bknight.png',
        'BP': 'bpawn.png'
    }
    
    for piece_code, filename in piece_file_mapping.items():
        image_path = os.path.join(pieces_folder, filename)
        
        if os.path.exists(image_path):
            try:
                # Load image with PIL to handle transparency
                pil_image = Image.open(image_path)
                
                # Convert to RGBA if not already
                if pil_image.mode != 'RGBA':
                    pil_image = pil_image.convert('RGBA')
                
                # Remove checkered/light background 
                img_array = np.array(pil_image)
                
                # More aggressive background removal for checkered patterns
                # Target multiple background patterns common in chess piece images
                
                # Method 1: Remove light colors (checkered squares are usually light)
                light_threshold = 220
                light_mask = (
                    (img_array[:, :, 0] > light_threshold) &
                    (img_array[:, :, 1] > light_threshold) &  
                    (img_array[:, :, 2] > light_threshold)
                )
                
                # Method 2: Remove specific checkered pattern colors
                # Common checkered colors: light gray, white, beige
                checkered_colors = [
                    (255, 255, 255),  # White
                    (240, 240, 240),  # Light gray
                    (220, 220, 220),  # Medium light gray
                    (200, 200, 200),  # Gray
                    (245, 245, 220),  # Beige
                    (255, 248, 220),  # Cornsilk
                ]
                
                color_tolerance = 30
                checkered_mask = np.zeros(img_array.shape[:2], dtype=bool)
                
                for target_color in checkered_colors:
                    color_diff = np.abs(img_array[:, :, :3] - np.array(target_color))
                    close_to_target = np.all(color_diff < color_tolerance, axis=2)
                    checkered_mask |= close_to_target
                
                # Combine masks
                background_mask = light_mask | checkered_mask
                
                # Make background pixels transparent
                img_array[background_mask, 3] = 0
                
                # Create PIL image from modified array
                cleaned_image = Image.fromarray(img_array, 'RGBA')
                
                # Resize to standard size (128x128) while maintaining aspect ratio
                max_size = 128
                cleaned_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Create a new 128x128 image with transparent background
                resized_image = Image.new('RGBA', (max_size, max_size), (0, 0, 0, 0))
                
                # Paste the resized piece in the center
                paste_x = (max_size - cleaned_image.width) // 2
                paste_y = (max_size - cleaned_image.height) // 2
                resized_image.paste(cleaned_image, (paste_x, paste_y), cleaned_image)
                
                # Convert PIL to numpy array
                final_array = np.array(resized_image)
                piece_images[piece_code] = final_array
                
            except Exception as e:
                print(f"⚠️  Could not load {image_path}: {e}")
        else:
            print(f"⚠️  Image not found: {image_path}")
    
    if piece_images:
        print(f"✓ Loaded {len(piece_images)} piece images from {pieces_folder}")
        loaded_pieces = list(piece_images.keys())
        print(f"   Available pieces: {', '.join(sorted(loaded_pieces))}")
    else:
        print("❌ No piece images loaded, using Unicode fallback")
        
    return piece_images if piece_images else None

def draw_chess_board(board=None, title="Chess Starting Position", save_path=None, 
                    pieces_folder="2d_chess_pieces_images", piece_scale=0.8):
    """
    Draw a chess board with Chess.com style colors and custom piece images.
    
    Args:
        board (numpy.ndarray): 8x8 array with piece codes. If None, uses starting position.
        title (str): Title for the plot
        save_path (str): Optional path to save the image
        pieces_folder (str): Path to folder containing piece PNG images
        piece_scale (float): Scale factor for piece images (0.1 to 1.0)
    """
    
    if board is None:
        board = create_chess_board_layout()
    
    # Load piece images
    piece_images = load_piece_images(pieces_folder)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Chess.com style colors
    light_square = '#EEEED2'  # Light cream
    dark_square = '#769656'   # Forest green
    
    # Draw squares
    for row in range(8):
        for col in range(8):
            # Determine square color (alternating pattern)
            is_light = (row + col) % 2 == 0
            color = light_square if is_light else dark_square
            
            # Draw square
            square = Rectangle((col, 7-row), 1, 1, 
                             facecolor=color, 
                             edgecolor='black', 
                             linewidth=1)
            ax.add_patch(square)
            
            # Add piece if present
            piece_code = board[row, col]
            if piece_code:
                if piece_images and piece_code in piece_images:
                    # Use custom piece image (now pre-resized to 128x128)
                    piece_img = piece_images[piece_code]
                    
                    # Create OffsetImage with simple scaling
                    imagebox = OffsetImage(piece_img, zoom=piece_scale * 0.6)
                    
                    # Position at square center
                    ab = AnnotationBbox(imagebox, (col + 0.5, 7-row + 0.5), 
                                      frameon=False, pad=0, box_alignment=(0.5, 0.5))
                    ax.add_artist(ab)
                else:
                    # Fallback to Unicode symbols
                    piece_symbol = get_piece_unicode(piece_code)
                    ax.text(col + 0.5, 7-row + 0.5, piece_symbol, 
                           fontsize=36, ha='center', va='center',
                           color='black' if piece_code.startswith('B') else 'white',
                           weight='bold')
    
    # Add file labels (a-h)
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    for i, file_label in enumerate(files):
        ax.text(i + 0.5, -0.2, file_label, fontsize=14, ha='center', va='center', weight='bold')
        ax.text(i + 0.5, 8.1, file_label, fontsize=14, ha='center', va='center', weight='bold')
    
    # Add rank labels (1-8)
    ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
    for i, rank_label in enumerate(ranks):
        ax.text(-0.2, i + 0.5, rank_label, fontsize=14, ha='center', va='center', weight='bold')
        ax.text(8.1, i + 0.5, rank_label, fontsize=14, ha='center', va='center', weight='bold')
    
    # Set up the plot
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=18, weight='bold', pad=20)
    
    # Add legend
    light_patch = mpatches.Patch(color=light_square, label='Light Squares')
    dark_patch = mpatches.Patch(color=dark_square, label='Dark Squares')
    ax.legend(handles=[light_patch, dark_patch], loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"♟️  Chess board saved to: {save_path}")
    
    plt.show()
    
    return fig, ax

def create_custom_board_from_pieces(piece_positions):
    """
    Create a custom board layout from piece positions.
    
    Args:
        piece_positions (dict): Dictionary with (row, col) keys and piece codes as values
                               Example: {(0, 0): 'BR', (0, 1): 'BN', ...}
    
    Returns:
        numpy.ndarray: 8x8 board array
    """
    board = np.full((8, 8), '', dtype=object)
    
    for (row, col), piece in piece_positions.items():
        if 0 <= row < 8 and 0 <= col < 8:
            board[row, col] = piece
    
    return board

def demo_different_positions():
    """Demonstrate different chess positions"""
    
    # 1. Starting position
    print("🏁 Drawing starting position...")
    draw_chess_board(title="Standard Chess Starting Position")
    
    # 2. Custom position - Scholar's Mate setup
    scholars_mate_pieces = {
        # White pieces
        (7, 0): 'WR', (7, 1): 'WN', (7, 2): 'WB', (7, 4): 'WK',
        (7, 5): 'WB', (7, 6): 'WN', (7, 7): 'WR',
        (6, 0): 'WP', (6, 1): 'WP', (6, 2): 'WP', (6, 3): 'WP',
        (6, 5): 'WP', (6, 6): 'WP', (6, 7): 'WP',
        (5, 4): 'WP',  # Pawn moved
        (4, 7): 'WQ',  # Queen moved for attack
        
        # Black pieces  
        (0, 0): 'BR', (0, 1): 'BN', (0, 2): 'BB', (0, 3): 'BQ',
        (0, 4): 'BK', (0, 5): 'BB', (0, 6): 'BN', (0, 7): 'BR',
        (1, 0): 'BP', (1, 1): 'BP', (1, 2): 'BP', (1, 3): 'BP',
        (1, 4): 'BP', (1, 6): 'BP', (1, 7): 'BP',
        (3, 5): 'BP',  # Pawn moved
    }
    
    custom_board = create_custom_board_from_pieces(scholars_mate_pieces)
    draw_chess_board(custom_board, title="Scholar's Mate Position")
    
    # 3. Endgame position
    endgame_pieces = {
        (7, 4): 'WK',  # White King
        (6, 3): 'WQ',  # White Queen
        (0, 4): 'BK',  # Black King
    }
    
    endgame_board = create_custom_board_from_pieces(endgame_pieces)
    draw_chess_board(endgame_board, title="Queen vs King Endgame")

def print_board_ascii(board=None):
    """Print ASCII representation of the board"""
    
    if board is None:
        board = create_chess_board_layout()
    
    print("\n" + "="*33)
    print("ASCII Chess Board Representation")
    print("="*33)
    
    # Top border with file labels
    print("    a b c d e f g h")
    print("  ┌─┬─┬─┬─┬─┬─┬─┬─┐")
    
    for row in range(8):
        rank = 8 - row
        row_str = f"{rank} │"
        
        for col in range(8):
            piece_code = board[row, col]
            if piece_code:
                symbol = get_piece_unicode(piece_code)
            else:
                # Empty square - show alternating pattern
                is_light = (row + col) % 2 == 0
                symbol = '·' if is_light else '□'
            
            row_str += symbol + "│"
        
        row_str += f" {rank}"
        print(row_str)
        
        if row < 7:
            print("  ├─┼─┼─┼─┼─┼─┼─┼─┤")
    
    print("  └─┴─┴─┴─┴─┴─┴─┴─┘")
    print("    a b c d e f g h")

def setup_piece_images_folder():
    """Check user's existing piece images folder"""
    pieces_folder = "2d_chess_pieces_images"
    
    if not os.path.exists(pieces_folder):
        print(f"❌ Piece images folder not found: {pieces_folder}")
        print(f"\n📋 Expected file structure:")
        print(f"   {pieces_folder}/")
        
        piece_files = {
            'king.png': 'White King',     'bking.png': 'Black King',
            'queen.png': 'White Queen',   'bqueen.png': 'Black Queen', 
            'rook.png': 'White Rook',     'brook.png': 'Black Rook',
            'bishop.png': 'White Bishop', 'bbishop.png': 'Black Bishop',
            'knight.png': 'White Knight', 'bknight.png': 'Black Knight',
            'pawn.png': 'White Pawn',     'bpawn.png': 'Black Pawn'
        }
        
        for filename, description in piece_files.items():
            print(f"   ├── {filename}  ({description})")
        
        return False
    else:
        # Check what images are available using user's naming convention
        available_images = []
        piece_files = ['king.png', 'queen.png', 'rook.png', 'bishop.png', 'knight.png', 'pawn.png',
                      'bking.png', 'bqueen.png', 'brook.png', 'bbishop.png', 'bknight.png', 'bpawn.png']
        
        for filename in piece_files:
            image_path = os.path.join(pieces_folder, filename)
            if os.path.exists(image_path):
                available_images.append(filename)
        
        print(f"📁 Found piece images folder: {pieces_folder}")
        print(f"✓ Available images: {len(available_images)}/12")
        
        if len(available_images) == 12:
            print("🎉 All piece images found!")
            return True
        else:
            missing = set(piece_files) - set(available_images)
            print(f"⚠️  Missing images: {', '.join(sorted(missing))}")
            return len(available_images) > 0

if __name__ == "__main__":
    print("🏁 Chess Board Reconstructor with Custom Piece Images")
    print("=" * 55)
    
    # Check/setup piece images folder
    has_images = setup_piece_images_folder()
    
    if has_images:
        print("\n📋 Creating visual chess board with custom images...")
        draw_chess_board(save_path="chess_starting_position.png", piece_scale=0.7)
        
        # Show demo of different positions
        response = input("\n🎮 Show demo positions? (y/n): ").strip().lower()
        if response == 'y':
            demo_different_positions()
    else:
        print("\n📋 Creating chess board with Unicode symbols (fallback)...")
        print_board_ascii()
        
        # Create board without custom images
        draw_chess_board(save_path="chess_starting_position_unicode.png")
    
    print("\n✅ Done! Add your PNG piece images to use custom graphics.")