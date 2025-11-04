#!/usr/bin/env python3
"""
Test Current Model on Detection Output
=====================================

Test the existing trained model on the newly labeled detection_output pieces.
This lets you evaluate performance before adding them to training.
"""

import pandas as pd
import numpy as np
import cv2
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from inference import predict_pieces_from_semantic

def test_model_on_detection_output():
    """Test the current model on detection output labeled data"""
    
    detection_path = "final_chess_dataset/detection_output"
    labels_file = os.path.join(detection_path, "detection_output_labels.csv")
    semantics_path = os.path.join(detection_path, "semantics")
    
    if not os.path.exists(labels_file):
        print(f"Error: Labels file not found: {labels_file}")
        print("Run 'python label_detection_output.py' first to label the pieces")
        return
    
    # Load labels
    labels_df = pd.read_csv(labels_file)
    print(f"Loaded {len(labels_df)} labeled pieces from detection output")
    
    # Group by image to process each semantic mask
    images = labels_df['image_name'].unique()
    print(f"Found {len(images)} images to test")
    
    all_predictions = []
    all_ground_truth = []
    detailed_results = []
    
    for image_name in images:
        print(f"\nProcessing {image_name}...")
        
        # Load semantic mask
        mask_path = os.path.join(semantics_path, image_name)
        semantic_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if semantic_mask is None:
            print(f"Error loading mask: {mask_path}")
            continue
        
        # Run inference
        try:
            predictions = predict_pieces_from_semantic(semantic_mask)
            print(f"  Model detected {len(predictions)} pieces")
        except Exception as e:
            print(f"  Error during inference: {e}")
            continue
        
        # Get ground truth for this image
        image_labels = labels_df[labels_df['image_name'] == image_name]
        ground_truth = {int(row['piece_value']): row['piece_type'] for _, row in image_labels.iterrows()}
        
        print(f"  Ground truth has {len(ground_truth)} pieces")
        
        # Match predictions to ground truth
        for piece_value, gt_type in ground_truth.items():
            if piece_value in predictions:
                pred_info = predictions[piece_value]
                pred_type = pred_info['type']
                confidence = pred_info['confidence']
                
                all_predictions.append(pred_type)
                all_ground_truth.append(gt_type)
                
                detailed_results.append({
                    'image': image_name,
                    'piece_value': piece_value,
                    'ground_truth': gt_type,
                    'prediction': pred_type,
                    'confidence': confidence,
                    'correct': pred_type == gt_type
                })
                
                status = "✓" if pred_type == gt_type else "✗"
                print(f"    Piece {piece_value}: {gt_type} → {pred_type} ({confidence:.3f}) {status}")
            else:
                print(f"    Piece {piece_value}: {gt_type} → NOT DETECTED ✗")
                detailed_results.append({
                    'image': image_name,
                    'piece_value': piece_value,
                    'ground_truth': gt_type,
                    'prediction': 'NOT_DETECTED',
                    'confidence': 0.0,
                    'correct': False
                })
    
    # Calculate metrics
    if not all_predictions:
        print("\nNo predictions to evaluate!")
        return
    
    accuracy = accuracy_score(all_ground_truth, all_predictions)
    detection_rate = len(all_predictions) / len(labels_df)
    
    print("\n" + "="*60)
    print("DETECTION OUTPUT TEST RESULTS")
    print("="*60)
    print(f"Total pieces in ground truth: {len(labels_df)}")
    print(f"Pieces detected by model: {len(all_predictions)}")
    print(f"Detection rate: {detection_rate:.1%}")
    print(f"Classification accuracy (detected pieces): {accuracy:.1%}")
    
    # Per-class performance
    print(f"\nGround truth distribution:")
    gt_counts = labels_df['piece_type'].value_counts()
    for piece, count in gt_counts.items():
        print(f"  {piece}: {count}")
    
    # Detailed classification report
    if len(set(all_ground_truth)) > 1:  # Need multiple classes for report
        print(f"\nDetailed classification report:")
        print(classification_report(all_ground_truth, all_predictions))
    
    # Show worst performing pieces
    results_df = pd.DataFrame(detailed_results)
    incorrect = results_df[~results_df['correct']]
    if len(incorrect) > 0:
        print(f"\nIncorrect predictions ({len(incorrect)} total):")
        for _, row in incorrect.head(10).iterrows():
            print(f"  {row['image']}, piece {row['piece_value']}: {row['ground_truth']} → {row['prediction']} (conf: {row['confidence']:.3f})")
    
    # Save detailed results
    results_file = os.path.join(detection_path, "model_test_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")
    
    return results_df

if __name__ == "__main__":
    test_model_on_detection_output()
