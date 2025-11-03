"""
Interactive Semantic Piece Labeling Tool
========================================

This tool scans the final_chess_dataset for unlabeled semantic images
and prompts the user to identify pieces by typing codes:
- p: pawn
- r: rook  
- kn: knight
- b: bishop
- q: queen
- ki: king

The labels are saved to labels.csv for training the classifier.
"""

import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

# Piece type mappings
PIECE_CODES = {
    'p': 'pawn',
    'r': 'rook', 
    'kn': 'knight',
    'bi': 'bishop',  # Changed from 'b' to 'bi' to avoid conflict
    'q': 'queen',
    'ki': 'king'
}

def display_semantic_piece(semantic_img, piece_value, img_name, original_img_path=None):
    """Display a semantic piece alongside the original image for user labeling"""
    # Create binary mask for this piece
    piece_mask = (semantic_img == piece_value).astype(np.uint8) * 255
    
    # Find contours to get bounding box
    contours, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    
    # Get the largest contour (main piece)
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    
    # Add padding
    padding = 20
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(semantic_img.shape[1], x + w + padding)
    y_end = min(semantic_img.shape[0], y + h + padding)
    
    # Crop the piece region
    cropped_semantic = piece_mask[y_start:y_end, x_start:x_end]
    
    # Load and crop the original image if available
    original_img = None
    cropped_original = None
    
    if original_img_path and os.path.exists(original_img_path):
        original_img = cv2.imread(original_img_path)
        if original_img is not None:
            # Convert BGR to RGB for display
            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            # Crop the same region
            cropped_original = original_img_rgb[y_start:y_end, x_start:x_end]
    
    # Create figure with subplots
    if cropped_original is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display original image
        ax1.imshow(cropped_original)
        ax1.set_title(f'Original Chess Piece\n{img_name}')
        ax1.axis('off')
        
        # Display semantic mask
        ax2.imshow(cropped_semantic, cmap='gray')
        ax2.set_title(f'Semantic Mask\nValue: {piece_value}\nArea: {cv2.contourArea(main_contour):.0f} pixels')
        ax2.axis('off')
        
        plt.tight_layout()
    else:
        # Fallback to semantic only if original not available
        plt.figure(figsize=(6, 6))
        plt.imshow(cropped_semantic, cmap='gray')
        plt.title(f'Semantic Only (Original not found)\nImage: {img_name}\nPiece Value: {piece_value}\nArea: {cv2.contourArea(main_contour):.0f} pixels')
        plt.axis('off')
        plt.tight_layout()
    
    plt.show(block=False)
    plt.pause(0.1)  # Allow display to update
    
    return cropped_semantic

def load_existing_labels(labels_file):
    """Load existing labels from CSV file"""
    if os.path.exists(labels_file):
        try:
            df = pd.read_csv(labels_file)
            # Create a set of (image_name, piece_value) tuples for quick lookup
            labeled_pieces = set(zip(df['image_name'], df['piece_value']))
            return df, labeled_pieces
        except Exception as e:
            print(f"Error loading existing labels: {e}")
            return pd.DataFrame(), set()
    else:
        # Create new DataFrame with required columns
        df = pd.DataFrame(columns=['image_name', 'piece_value', 'piece_type', 'area', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'])
        return df, set()

def save_labels(labels_df, labels_file):
    """Save labels to CSV file"""
    labels_df.to_csv(labels_file, index=False)
    print(f"✓ Labels saved to {labels_file}")

def find_corresponding_original_image(semantic_folder, img_file):
    """Find the corresponding original image for a semantic mask"""
    # Get the parent directory of the semantics folder
    parent_dir = os.path.dirname(semantic_folder)
    
    # Look for 'images' folder in the same parent directory
    images_folder = os.path.join(parent_dir, 'images')
    
    if not os.path.exists(images_folder):
        return None
    
    # Remove common semantic suffixes from filename
    base_name = os.path.splitext(img_file)[0]
    
    # Common semantic suffixes to remove
    semantic_suffixes = ['_mask', '_semantic', '_seg', '_segmentation']
    original_base = base_name
    
    for suffix in semantic_suffixes:
        if base_name.endswith(suffix):
            original_base = base_name[:-len(suffix)]
            break
    
    # Common image extensions to check
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # First try: exact match with original base name
    for ext in extensions:
        potential_path = os.path.join(images_folder, original_base + ext)
        if os.path.exists(potential_path):
            return potential_path
    
    # Second try: fuzzy matching - look for files that start with the base name
    try:
        image_files = os.listdir(images_folder)
        for image_file in image_files:
            image_base = os.path.splitext(image_file)[0]
            # Check if the image base name matches our original base (case insensitive)
            if image_base.lower() == original_base.lower():
                return os.path.join(images_folder, image_file)
    except:
        pass
    
    return None

def organize_training_data(dataset_path):
    """Move files from training_example folders into main training folders"""
    training_path = os.path.join(dataset_path, 'training')
    training_example_path = os.path.join(training_path, 'training_example')
    
    if not os.path.exists(training_example_path):
        print(f"No training_example folder found at {training_example_path}")
        return
    
    print(f"📁 Organizing training data from {training_example_path}")
    
    # Subfolders to merge
    subfolders = ['display', 'images', 'semantics']
    
    for subfolder in subfolders:
        source_folder = os.path.join(training_example_path, subfolder)
        target_folder = os.path.join(training_path, subfolder)
        
        if not os.path.exists(source_folder):
            print(f"   ⚠️  Source folder not found: {source_folder}")
            continue
        
        # Create target folder if it doesn't exist
        os.makedirs(target_folder, exist_ok=True)
        
        # Get files in source folder
        files = [f for f in os.listdir(source_folder) 
                if os.path.isfile(os.path.join(source_folder, f))]
        
        if not files:
            print(f"   📂 {subfolder}/: No files to move")
            continue
        
        print(f"   📂 {subfolder}/: Moving {len(files)} files...")
        
        # Move each file
        moved_count = 0
        for file in files:
            source_file = os.path.join(source_folder, file)
            target_file = os.path.join(target_folder, file)
            
            # Check if target file already exists
            if os.path.exists(target_file):
                # Generate unique name if file exists
                base_name, ext = os.path.splitext(file)
                counter = 1
                while os.path.exists(target_file):
                    new_name = f"{base_name}_{counter}{ext}"
                    target_file = os.path.join(target_folder, new_name)
                    counter += 1
                print(f"      ⚠️  Renamed {file} to {os.path.basename(target_file)} (file existed)")
            
            try:
                shutil.move(source_file, target_file)
                moved_count += 1
            except Exception as e:
                print(f"      ❌ Error moving {file}: {e}")
        
        print(f"      ✅ Moved {moved_count}/{len(files)} files")
    
    # Remove empty training_example folder structure if all files moved
    try:
        # Check if all subfolders are empty
        all_empty = True
        for subfolder in subfolders:
            source_folder = os.path.join(training_example_path, subfolder)
            if os.path.exists(source_folder) and os.listdir(source_folder):
                all_empty = False
                break
        
        if all_empty:
            shutil.rmtree(training_example_path)
            print(f"   🗑️  Removed empty training_example folder")
        else:
            print(f"   📁 Kept training_example folder (some files remain)")
    except Exception as e:
        print(f"   ⚠️  Could not remove training_example folder: {e}")
    
    print("✅ Training data organization complete!")

def scan_and_label_semantics(dataset_path):
    """Main function to scan and label semantic pieces"""
    
    # Create labels file path
    labels_file = os.path.join(dataset_path, 'labels.csv')
    
    # Load existing labels
    labels_df, labeled_pieces = load_existing_labels(labels_file)
    print(f"Loaded {len(labeled_pieces)} existing labels")
    
    # Scan for semantic files
    semantic_folders = []
    for root, dirs, files in os.walk(dataset_path):
        if 'semantics' in os.path.basename(root):
            semantic_folders.append(root)
    
    if not semantic_folders:
        print(f"No 'semantics' folders found in {dataset_path}")
        return
    
    print(f"Found {len(semantic_folders)} semantics folders")
    
    total_labeled = 0
    total_skipped = 0
    
    for folder in semantic_folders:
        print(f"\n--- Processing folder: {folder} ---")
        
        # Get all semantic image files
        image_files = [f for f in os.listdir(folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            img_path = os.path.join(folder, img_file)
            
            # Load semantic image
            semantic_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if semantic_img is None:
                print(f"⚠️  Could not load {img_file}")
                continue
            
            # Get unique piece values (excluding background)
            unique_values = np.unique(semantic_img)
            piece_values = [v for v in unique_values if v != 0 and v < 50]  # Assuming piece values are < 50
            
            print(f"\n📁 Image: {img_file}")
            print(f"   Found {len(piece_values)} pieces: {piece_values}")
            
            # Find corresponding original image
            original_img_path = find_corresponding_original_image(folder, img_file)
            if original_img_path:
                print(f"   Found original image: {original_img_path}")
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
                    print(f"\n🎯 Piece value {piece_value} in {img_file}")
                    print("Enter piece type: p(awn), r(ook), kn(ight), bi(shop), q(ueen), ki(ng)")
                    print("Or: s(kip), x(quit), undo, edit")
                    
                    while True:
                        user_input = input("Piece type: ").strip().lower()
                        
                        if user_input in ['x', 'quit', 'exit']:  # Quit
                            print("Quitting labeling session...")
                            save_labels(labels_df, labels_file)
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
                                save_labels(labels_df, labels_file)
                                print("✅ Last label removed")
                            else:
                                print("No labels to undo")
                            break
                        
                        elif user_input in ['edit', 'e']:  # Edit existing labels
                            print("\n📝 Recent labels:")
                            if len(labels_df) >= 5:
                                recent = labels_df.tail(5)
                            else:
                                recent = labels_df
                            
                            for i, (idx, row) in enumerate(recent.iterrows()):
                                print(f"   {i+1}. {row['image_name']} piece {row['piece_value']}: {row['piece_type']}")
                            
                            try:
                                choice = input("Enter number to edit (or press Enter to cancel): ").strip()
                                if choice:
                                    choice_idx = int(choice) - 1
                                    if 0 <= choice_idx < len(recent):
                                        edit_row = recent.iloc[choice_idx]
                                        original_idx = edit_row.name
                                        print(f"Editing: {edit_row['image_name']} piece {edit_row['piece_value']} (currently: {edit_row['piece_type']})")
                                        
                                        new_type = input("Enter new piece type (p,r,kn,bi,q,ki): ").strip().lower()
                                        if new_type in PIECE_CODES:
                                            labels_df.at[original_idx, 'piece_type'] = PIECE_CODES[new_type]
                                            save_labels(labels_df, labels_file)
                                            print(f"✅ Updated to {PIECE_CODES[new_type]}")
                                        else:
                                            print("Invalid piece type")
                                    else:
                                        print("Invalid choice")
                            except ValueError:
                                print("Invalid input")
                            break
                        
                        elif user_input in PIECE_CODES:
                            # Valid piece code
                            piece_type = PIECE_CODES[user_input]
                            
                            # Get piece info for labels
                            piece_mask = (semantic_img == piece_value).astype(np.uint8)
                            contours, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                main_contour = max(contours, key=cv2.contourArea)
                                area = cv2.contourArea(main_contour)
                                x, y, w, h = cv2.boundingRect(main_contour)
                                
                                # Add to labels DataFrame
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
                                
                                labels_df = pd.concat([labels_df, pd.DataFrame([new_label])], ignore_index=True)
                                labeled_pieces.add((img_file, piece_value))
                                total_labeled += 1
                                
                                print(f"✓ Labeled as {piece_type}")
                                break
                            else:
                                print("Error: Could not analyze piece contour")
                                break
                        
                        else:
                            print("Invalid input. Use: p, r, kn, bi, q, ki, s, x, undo, edit")
                    
                    # Close the current plot
                    plt.close()
                    
                    # Save periodically (every 10 labels)
                    if total_labeled % 10 == 0:
                        save_labels(labels_df, labels_file)
                        
                except KeyboardInterrupt:
                    print("\nInterrupted by user. Saving labels...")
                    save_labels(labels_df, labels_file)
                    plt.close('all')
                    return
                except Exception as e:
                    print(f"Error processing piece {piece_value}: {e}")
                    continue
    
    # Final save
    save_labels(labels_df, labels_file)
    plt.close('all')
    
    print(f"\n🎉 Labeling complete!")
    print(f"   Total labeled: {total_labeled}")
    print(f"   Total skipped: {total_skipped}")
    
    # Show summary
    if len(labels_df) > 0:
        print(f"\n📊 Label Summary:")
        summary = labels_df['piece_type'].value_counts()
        for piece_type, count in summary.items():
            print(f"   {piece_type}: {count}")

def edit_existing_labels(dataset_path):
    """Standalone function to edit existing labels"""
    labels_file = os.path.join(dataset_path, 'labels.csv')
    
    if not os.path.exists(labels_file):
        print(f"❌ No labels file found: {labels_file}")
        return
    
    labels_df, _ = load_existing_labels(labels_file)
    
    if len(labels_df) == 0:
        print("❌ No labels found to edit")
        return
    
    print(f"\n📝 Edit Mode - Found {len(labels_df)} existing labels")
    
    while True:
        print(f"\n📊 Current label summary:")
        summary = labels_df['piece_type'].value_counts()
        for piece_type, count in summary.items():
            print(f"   {piece_type}: {count}")
        
        print(f"\n📝 Recent labels (last 10):")
        recent = labels_df.tail(10)
        for i, (idx, row) in enumerate(recent.iterrows()):
            print(f"   {i+1:2d}. {row['image_name'][:30]:<30} piece {row['piece_value']:2d}: {row['piece_type']}")
        
        print(f"\nEdit options:")
        print(f"  1-{len(recent)}: Edit specific label from list above")
        print(f"  row X: Edit specific row number (e.g., 'row 52')")
        print(f"  undo: Remove last label")
        print(f"  all: Show all labels")
        print(f"  search: Find labels by image name")
        print(f"  x: Exit edit mode")
        
        choice = input(f"\nChoice: ").strip().lower()
        
        if choice in ['x', 'quit', 'exit']:
            break
        
        elif choice == 'undo':
            if len(labels_df) > 0:
                last_label = labels_df.iloc[-1]
                print(f"Removing: {last_label['piece_type']} for {last_label['image_name']} piece {last_label['piece_value']}")
                confirm = input("Confirm removal? (y/n): ").strip().lower()
                if confirm == 'y':
                    labels_df = labels_df.iloc[:-1]
                    save_labels(labels_df, labels_file)
                    print("✅ Label removed")
                else:
                    print("Cancelled")
            else:
                print("No labels to remove")
        
        elif choice == 'all':
            print(f"\n📋 All {len(labels_df)} labels:")
            for i, (idx, row) in enumerate(labels_df.iterrows()):
                print(f"   {i+1:3d}. {row['image_name'][:25]:<25} piece {row['piece_value']:2d}: {row['piece_type']}")
            
            # Allow immediate editing from the full list
            edit_choice = input(f"\nEnter row number to edit (1-{len(labels_df)}) or press Enter to go back: ").strip()
            if edit_choice.isdigit():
                row_idx = int(edit_choice) - 1
                if 0 <= row_idx < len(labels_df):
                    edit_row = labels_df.iloc[row_idx]
                    
                    print(f"\n✏️  Editing row {edit_choice}: {edit_row['image_name']} piece {edit_row['piece_value']}")
                    print(f"   Current: {edit_row['piece_type']}")
                    print(f"   Area: {edit_row['area']:.1f} pixels")
                    
                    # Ask if user wants to see the images
                    show_images = input("Show images? (y/n): ").strip().lower()
                    if show_images == 'y':
                        # Find the semantic image file
                        semantic_path = None
                        for root, dirs, files in os.walk(dataset_path):
                            if 'semantics' in os.path.basename(root) and edit_row['image_name'] in files:
                                semantic_path = os.path.join(root, edit_row['image_name'])
                                break
                        
                        if semantic_path and os.path.exists(semantic_path):
                            # Load semantic image
                            semantic_img = cv2.imread(semantic_path, cv2.IMREAD_GRAYSCALE)
                            if semantic_img is not None:
                                # Find corresponding original image
                                original_img_path = find_corresponding_original_image(os.path.dirname(semantic_path), edit_row['image_name'])
                                
                                # Display the images
                                display_semantic_piece(semantic_img, edit_row['piece_value'], edit_row['image_name'], original_img_path)
                                input("Press Enter to continue after viewing images...")
                                plt.close('all')  # Close the display
                            else:
                                print("Could not load semantic image")
                        else:
                            print(f"Semantic image not found: {edit_row['image_name']}")
                    
                    new_type = input("Enter new piece type (p,r,kn,bi,q,ki) or press Enter to cancel: ").strip().lower()
                    if new_type in PIECE_CODES:
                        old_type = edit_row['piece_type']
                        labels_df.iloc[row_idx, labels_df.columns.get_loc('piece_type')] = PIECE_CODES[new_type]
                        save_labels(labels_df, labels_file)
                        print(f"✅ Updated row {edit_choice} from {old_type} to {PIECE_CODES[new_type]}")
                        # Reload the dataframe to reflect changes
                        labels_df, _ = load_existing_labels(labels_file)
                    elif new_type == "":
                        print("Cancelled")
                    else:
                        print("Invalid piece type")
                else:
                    print("Invalid row number")
        
        elif choice == 'search':
            search_term = input("Enter part of image name to search: ").strip()
            if search_term:
                matches = labels_df[labels_df['image_name'].str.contains(search_term, case=False)]
                if len(matches) > 0:
                    print(f"\n🔍 Found {len(matches)} matches:")
                    match_indices = []
                    for i, (idx, row) in enumerate(matches.iterrows()):
                        original_row = idx + 1  # Convert to 1-based row number
                        match_indices.append(idx)
                        print(f"   {i+1}. {row['image_name']} piece {row['piece_value']}: {row['piece_type']} (row {original_row})")
                    
                    # Allow immediate editing from search results
                    edit_choice = input(f"\nEnter number to edit (1-{len(matches)}) or press Enter to go back: ").strip()
                    if edit_choice.isdigit():
                        match_idx = int(edit_choice) - 1
                        if 0 <= match_idx < len(matches):
                            row_idx = match_indices[match_idx]
                            edit_row = labels_df.iloc[row_idx]
                            
                            print(f"\n✏️  Editing: {edit_row['image_name']} piece {edit_row['piece_value']}")
                            print(f"   Current: {edit_row['piece_type']}")
                            print(f"   Area: {edit_row['area']:.1f} pixels")
                            
                            # Ask if user wants to see the images
                            show_images = input("Show images? (y/n): ").strip().lower()
                            if show_images == 'y':
                                # Find the semantic image file
                                semantic_path = None
                                for root, dirs, files in os.walk(dataset_path):
                                    if 'semantics' in os.path.basename(root) and edit_row['image_name'] in files:
                                        semantic_path = os.path.join(root, edit_row['image_name'])
                                        break
                                
                                if semantic_path and os.path.exists(semantic_path):
                                    # Load semantic image
                                    semantic_img = cv2.imread(semantic_path, cv2.IMREAD_GRAYSCALE)
                                    if semantic_img is not None:
                                        # Find corresponding original image
                                        original_img_path = find_corresponding_original_image(os.path.dirname(semantic_path), edit_row['image_name'])
                                        
                                        # Display the images
                                        display_semantic_piece(semantic_img, edit_row['piece_value'], edit_row['image_name'], original_img_path)
                                        input("Press Enter to continue after viewing images...")
                                        plt.close('all')  # Close the display
                                    else:
                                        print("Could not load semantic image")
                                else:
                                    print(f"Semantic image not found: {edit_row['image_name']}")
                            
                            new_type = input("Enter new piece type (p,r,kn,bi,q,ki) or press Enter to cancel: ").strip().lower()
                            if new_type in PIECE_CODES:
                                old_type = edit_row['piece_type']
                                labels_df.iloc[row_idx, labels_df.columns.get_loc('piece_type')] = PIECE_CODES[new_type]
                                save_labels(labels_df, labels_file)
                                print(f"✅ Updated from {old_type} to {PIECE_CODES[new_type]}")
                                # Reload the dataframe to reflect changes
                                labels_df, _ = load_existing_labels(labels_file)
                            elif new_type == "":
                                print("Cancelled")
                            else:
                                print("Invalid piece type")
                        else:
                            print("Invalid choice")
                else:
                    print("No matches found")
        
        elif choice.startswith('row '):
            try:
                row_num = int(choice.split()[1])
                row_idx = row_num - 1  # Convert to 0-based index
                if 0 <= row_idx < len(labels_df):
                    edit_row = labels_df.iloc[row_idx]
                    
                    print(f"\n✏️  Editing row {row_num}: {edit_row['image_name']} piece {edit_row['piece_value']}")
                    print(f"   Current: {edit_row['piece_type']}")
                    print(f"   Area: {edit_row['area']:.1f} pixels")
                    
                    # Ask if user wants to see the images
                    show_images = input("Show images? (y/n): ").strip().lower()
                    if show_images == 'y':
                        # Find the semantic image file
                        semantic_path = None
                        for root, dirs, files in os.walk(dataset_path):
                            if 'semantics' in os.path.basename(root) and edit_row['image_name'] in files:
                                semantic_path = os.path.join(root, edit_row['image_name'])
                                break
                        
                        if semantic_path and os.path.exists(semantic_path):
                            # Load semantic image
                            semantic_img = cv2.imread(semantic_path, cv2.IMREAD_GRAYSCALE)
                            if semantic_img is not None:
                                # Find corresponding original image
                                original_img_path = find_corresponding_original_image(os.path.dirname(semantic_path), edit_row['image_name'])
                                
                                # Display the images
                                display_semantic_piece(semantic_img, edit_row['piece_value'], edit_row['image_name'], original_img_path)
                                input("Press Enter to continue after viewing images...")
                                plt.close('all')  # Close the display
                            else:
                                print("Could not load semantic image")
                        else:
                            print(f"Semantic image not found: {edit_row['image_name']}")
                    
                    new_type = input("Enter new piece type (p,r,kn,bi,q,ki) or press Enter to cancel: ").strip().lower()
                    if new_type in PIECE_CODES:
                        old_type = edit_row['piece_type']
                        labels_df.iloc[row_idx, labels_df.columns.get_loc('piece_type')] = PIECE_CODES[new_type]
                        save_labels(labels_df, labels_file)
                        print(f"✅ Updated row {row_num} from {old_type} to {PIECE_CODES[new_type]}")
                        # Reload the dataframe to reflect changes
                        labels_df, _ = load_existing_labels(labels_file)
                    elif new_type == "":
                        print("Cancelled")
                    else:
                        print("Invalid piece type")
                else:
                    print(f"Invalid row number. Valid range: 1-{len(labels_df)}")
            except (ValueError, IndexError):
                print("Invalid format. Use 'row X' where X is the row number")
        
        elif choice.isdigit():
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(recent):
                edit_row = recent.iloc[choice_idx]
                original_idx = edit_row.name
                
                print(f"\n✏️  Editing: {edit_row['image_name']} piece {edit_row['piece_value']}")
                print(f"   Current: {edit_row['piece_type']}")
                print(f"   Area: {edit_row['area']:.1f} pixels")
                
                new_type = input("Enter new piece type (p,r,kn,bi,q,ki) or press Enter to cancel: ").strip().lower()
                if new_type in PIECE_CODES:
                    old_type = edit_row['piece_type']
                    labels_df.iloc[original_idx, labels_df.columns.get_loc('piece_type')] = PIECE_CODES[new_type]
                    save_labels(labels_df, labels_file)
                    print(f"✅ Updated from {old_type} to {PIECE_CODES[new_type]}")
                    # Reload the dataframe to reflect changes
                    labels_df, _ = load_existing_labels(labels_file)
                elif new_type == "":
                    print("Cancelled")
                else:
                    print("Invalid piece type")
            else:
                print("Invalid choice")
        
        else:
            print("Invalid option")

def main():
    """Main entry point"""
    # Default dataset path
    dataset_path = "final_chess_dataset"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset folder not found: {dataset_path}")
        print("Please ensure the final_chess_dataset folder exists with training/ and testing/ subdirectories")
        return
    
    print("🏷️  Interactive Semantic Piece Labeling Tool")
    print("=" * 50)
    print(f"Scanning dataset: {dataset_path}")
    
    # First, organize training data by moving files from training_example
    organize_training_data(dataset_path)
    
    # Check if there are existing labels
    labels_file = os.path.join(dataset_path, 'labels.csv')
    if os.path.exists(labels_file):
        labels_df, _ = load_existing_labels(labels_file)
        print(f"\n📊 Found {len(labels_df)} existing labels")
        
        # Check if user wants to edit existing labels or continue labeling
        print("\nOptions:")
        print("  1: Continue labeling new pieces")
        print("  2: Edit existing labels")
        choice = input("Choose (1 or 2): ").strip()
        
        if choice == '2':
            edit_existing_labels(dataset_path)
            return
    
    print("\nPiece codes:")
    for code, name in PIECE_CODES.items():
        print(f"  {code}: {name}")
    print("\nControls:")
    print("  s: skip current piece")
    print("  x: quit and save")
    print("  undo: remove last label")
    print("  edit: modify recent labels")
    
    input("\nPress Enter to start labeling...")
    
    try:
        scan_and_label_semantics(dataset_path)
        
        # After labeling is complete, offer edit mode
        print("\n🎉 All pieces labeled!")
        edit_choice = input("Want to edit any labels? (y/n): ").strip().lower()
        if edit_choice == 'y':
            edit_existing_labels(dataset_path)
            
    except Exception as e:
        print(f"Error during labeling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
