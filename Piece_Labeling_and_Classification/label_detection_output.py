#!/usr/bin/env python3
"""
Label the new unlabeled pieces in detection_output folder.
Creates a separate CSV file for these pieces that can be:
1. Used for testing the current model
2. Added to training dataset when retraining
"""

import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from label_semantics import display_semantic_piece, PIECE_CODES, find_corresponding_original_image

def load_existing_detection_labels(labels_file):
    """Load existing labels from detection output CSV file"""
    labels_df = pd.DataFrame()
    labeled_pieces = set()
    
    if os.path.exists(labels_file):
        labels_df = pd.read_csv(labels_file)
        # Create set of (image_name, piece_value) tuples for quick lookup
        labeled_pieces = set(zip(labels_df['image_name'], labels_df['piece_value']))
    
    return labels_df, labeled_pieces

def save_detection_labels(labels_df, labels_file):
    """Save labels to CSV file"""
    labels_df.to_csv(labels_file, index=False)
    print(f"Saved {len(labels_df)} labels to {labels_file}")

def label_detection_output_pieces():
    """
    Label all pieces in the detection_output folder using the same method as label_semantics.py
    Creates detection_output_labels.csv
    """
    detection_output_path = "final_chess_dataset/detection_output"
    semantics_path = os.path.join(detection_output_path, "semantics")
    labels_file = os.path.join(detection_output_path, "detection_output_labels.csv")
    
    # Check if semantics path exists
    if not os.path.exists(semantics_path):
        print(f"Error: Semantics path not found: {semantics_path}")
        return
    
    # Load existing labels
    labels_df, labeled_pieces = load_existing_detection_labels(labels_file)
    print(f"Loaded {len(labeled_pieces)} existing labels")
    
    # Get all semantic mask files
    mask_files = [f for f in os.listdir(semantics_path) if f.lower().endswith('.png')]
    mask_files.sort(key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else 0)
    
    if not mask_files:
        print("No mask files found in semantics folder")
        return
    
    print(f"Found {len(mask_files)} mask files to process")
    
    print("\n" + "="*60)
    print("LABELING DETECTION OUTPUT PIECES")
    print("="*60)
    print("Available piece codes:")
    for code, name in PIECE_CODES.items():
        print(f"  {code} = {name}")
    print("\nControls:")
    print("  Enter piece code (p/r/kn/bi/q/ki) to label")
    print("  's' = skip this piece")
    print("  'x' = quit and save")
    print("  'undo' = remove last label")
    print("="*60)
    
    total_labeled = 0
    total_skipped = 0
    
    for img_file in mask_files:
        img_path = os.path.join(semantics_path, img_file)
        
        # Load semantic image
        semantic_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if semantic_img is None:
            print(f"Could not load {img_file}")
            continue
        
        # Get unique piece values (excluding background)
        unique_values = np.unique(semantic_img)
        piece_values = [v for v in unique_values if v != 0 and v < 50]  # Assuming piece values are < 50
        
        print(f"\nImage: {img_file}")
        print(f"   Found {len(piece_values)} pieces: {piece_values}")
        
        # Find corresponding original image
        original_img_path = find_corresponding_original_image(semantics_path, img_file)
        if original_img_path:
            print(f"   Found original image: {os.path.basename(original_img_path)}")
        else:
            print(f"   No corresponding original image found")
        
        # Process each piece in the image
        for piece_value in piece_values:
            # Check if already labeled
            if (img_file, piece_value) in labeled_pieces:
                total_skipped += 1
                continue
            
            # Display the piece with original image
            try:
                cropped_piece = display_semantic_piece(semantic_img, piece_value, img_file, original_img_path)
                if cropped_piece is None:
                    continue
                
                # Get user input
                print(f"\nPiece value {piece_value} in {img_file}")
                print("Enter piece type: p(awn), r(ook), kn(ight), bi(shop), q(ueen), ki(ng)")
                print("Or: s(kip), x(quit), undo")
                
                while True:
                    user_input = input("Piece type: ").strip().lower()
                    
                    if user_input in ['x', 'quit', 'exit']:  # Quit
                        print("Quitting labeling session...")
                        save_detection_labels(labels_df, labels_file)
                        plt.close('all')
                        return
                    
                    elif user_input == 's':  # Skip
                        print("Skipped")
                        break
                    
                    elif user_input in ['undo', 'u']:  # Undo last label
                        if len(labels_df) > 0:
                            last_label = labels_df.iloc[-1]
                            print(f"Removing last label: {last_label['piece_type']} for {last_label['image_name']} piece {last_label['piece_value']}")
                            labels_df = labels_df.iloc[:-1]  # Remove last row
                            labeled_pieces.discard((last_label['image_name'], last_label['piece_value']))
                            save_detection_labels(labels_df, labels_file)
                            print("Last label removed")
                        else:
                            print("No labels to undo")
                        break
                    
                    elif user_input in PIECE_CODES:
                        # Valid piece code
                        piece_type = PIECE_CODES[user_input]
                        
                        # Get piece info for labels
                        piece_mask = (semantic_img == piece_value).astype(np.uint8)
                        contours, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if len(contours) > 0:
                            main_contour = max(contours, key=cv2.contourArea)
                            x, y, w, h = cv2.boundingRect(main_contour)
                            area = cv2.contourArea(main_contour)
                            
                            # Create new label entry
                            new_label = {
                                'image_name': img_file,
                                'piece_value': piece_value,
                                'piece_type': piece_type,
                                'area': area,
                                'bbox_x': x,
                                'bbox_y': y,
                                'bbox_w': w,
                                'bbox_h': h
                            }
                            
                            # Add to dataframe
                            labels_df = pd.concat([labels_df, pd.DataFrame([new_label])], ignore_index=True)
                            labeled_pieces.add((img_file, piece_value))
                            
                            # Save after each label
                            save_detection_labels(labels_df, labels_file)
                            
                            print(f"Labeled as {piece_type}")
                            total_labeled += 1
                            break
                        else:
                            print("Could not find piece contour")
                    
                    else:
                        print("Invalid input. Use piece codes (p,r,kn,bi,q,ki), 's' to skip, 'x' to quit, or 'undo'")

                # Close the current plot
                plt.close('all')
                
            except Exception as e:
                print(f"Error processing piece {piece_value}: {e}")
                continue

    print(f"\nLabeling complete!")
    print(f"   Total pieces labeled: {total_labeled}")
    print(f"   Total pieces skipped: {total_skipped}")
    print(f"   Final dataset size: {len(labels_df)} pieces")
    plt.close('all')



def add_detection_labels_to_training():
    """
    Add the detection_output labels to the main training dataset.
    This merges detection_output_labels.csv into the main labels.csv
    """
    detection_labels_file = "final_chess_dataset/detection_output/detection_output_labels.csv"
    main_labels_file = "final_chess_dataset/labels.csv"
    backup_file = "final_chess_dataset/labels_before_detection_merge.csv"
    
    if not os.path.exists(detection_labels_file):
        print(f"Error: Detection labels file not found: {detection_labels_file}")
        return
    
    if not os.path.exists(main_labels_file):
        print(f"Error: Main labels file not found: {main_labels_file}")
        return
    
    # Load both datasets
    detection_df = pd.read_csv(detection_labels_file)
    main_df = pd.read_csv(main_labels_file)
    
    print(f"Detection output labels: {len(detection_df)} pieces")
    print(f"Main training labels: {len(main_df)} pieces")
    
    # Create backup
    main_df.to_csv(backup_file, index=False)
    print(f"Created backup: {backup_file}")
    
    # Merge datasets
    combined_df = pd.concat([main_df, detection_df], ignore_index=True)
    
    # Save combined dataset
    combined_df.to_csv(main_labels_file, index=False)
    
    print(f"Successfully merged datasets!")
    print(f"New training dataset: {len(combined_df)} pieces")
    print(f"Added {len(detection_df)} new pieces from detection_output")

def remove_specific_label():
    """Remove a specific label from detection output"""
    labels_file = "final_chess_dataset/detection_output/detection_output_labels.csv"
    
    if not os.path.exists(labels_file):
        print(f"No labels file found: {labels_file}")
        return
    
    df = pd.read_csv(labels_file)
    
    if len(df) == 0:
        print("No labels to remove")
        return
    
    print(f"\nCurrent labels ({len(df)} total):")
    for i, (idx, row) in enumerate(df.iterrows()):
        print(f"   {i+1}. {row['image_name']} piece {row['piece_value']}: {row['piece_type']}")
    
    try:
        choice = input("\nEnter number to remove (or press Enter to cancel): ").strip()
        if choice:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(df):
                row_to_remove = df.iloc[choice_idx]
                print(f"Removing: {row_to_remove['image_name']} piece {row_to_remove['piece_value']} ({row_to_remove['piece_type']})")
                
                confirm = input("Are you sure? (y/N): ").strip().lower()
                if confirm == 'y':
                    # Remove the row
                    df = df.drop(df.index[choice_idx]).reset_index(drop=True)
                    
                    # Save updated file
                    df.to_csv(labels_file, index=False)
                    print(f"Label removed. {len(df)} labels remaining.")
                else:
                    print("Cancelled")
            else:
                print("Invalid choice")
    except ValueError:
        print("Invalid input")

def show_detection_output_stats():
    """Show statistics about the detection output labels"""
    labels_file = "final_chess_dataset/detection_output/detection_output_labels.csv"
    
    if not os.path.exists(labels_file):
        print(f"No labels file found: {labels_file}")
        return
    
    df = pd.read_csv(labels_file)
    print(f"\nDetection Output Statistics")
    print(f"Total pieces labeled: {len(df)}")
    print(f"Piece type distribution:")
    piece_counts = df['piece_type'].value_counts()
    for piece, count in piece_counts.items():
        print(f"  {piece}: {count}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "merge":
            add_detection_labels_to_training()
        elif sys.argv[1] == "stats":
            show_detection_output_stats()
        elif sys.argv[1] == "remove":
            remove_specific_label()
        else:
            print("Usage:")
            print("  python label_detection_output.py          # Label pieces")
            print("  python label_detection_output.py merge    # Merge to training")
            print("  python label_detection_output.py stats    # Show statistics")
            print("  python label_detection_output.py remove   # Remove specific label")
    else:
        label_detection_output_pieces()