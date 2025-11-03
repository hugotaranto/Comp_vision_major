#!/usr/bin/env python3
"""
Test the trained semantic chess classifier on detection_output data.
This script evaluates the classifier performance on new labeled test data.
"""

import pandas as pd
import numpy as np
import cv2
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from inference import predict_pieces_from_semantic

def load_detection_output_data():
    """Load the detection output test data and semantic masks."""
    base_path = "final_chess_dataset/detection_output"
    labels_path = os.path.join(base_path, "labels.csv")
    semantics_path = os.path.join(base_path, "semantics")
    
    # Load ground truth labels
    labels_df = pd.read_csv(labels_path)
    print(f"Loaded {len(labels_df)} labeled pieces from {labels_path}")
    
    # Load semantic mask image
    mask_files = [f for f in os.listdir(semantics_path) if f.endswith('.png')]
    if not mask_files:
        raise FileNotFoundError("No semantic mask files found in semantics folder")
    
    mask_path = os.path.join(semantics_path, mask_files[0])
    semantic_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if semantic_mask is None:
        raise FileNotFoundError(f"Could not load semantic mask from {mask_path}")
    
    print(f"Loaded semantic mask: {mask_path}")
    print(f"Mask dimensions: {semantic_mask.shape}")
    print(f"Unique values in mask: {np.unique(semantic_mask)}")
    
    return labels_df, semantic_mask, mask_path

def run_classifier_test():
    """Run the trained classifier on detection output data and evaluate performance."""
    print("="*60)
    print("TESTING TRAINED SEMANTIC CHESS CLASSIFIER")
    print("="*60)
    
    # Load test data
    labels_df, semantic_mask, mask_path = load_detection_output_data()
    
    # Run inference
    print("\nRunning classifier inference...")
    try:
        predictions = predict_pieces_from_semantic(semantic_mask)
        print(f"Classifier found {len(predictions)} pieces")
    except Exception as e:
        print(f"Error during inference: {e}")
        return
    
    # Prepare ground truth data
    ground_truth = {}
    for _, row in labels_df.iterrows():
        piece_id = int(row['piece_value'])
        piece_type = row['piece_type']
        ground_truth[piece_id] = piece_type
    
    print(f"\nGround truth contains {len(ground_truth)} pieces")
    print("Ground truth piece distribution:")
    piece_counts = labels_df['piece_type'].value_counts()
    for piece, count in piece_counts.items():
        print(f"  {piece}: {count}")
    
    # Match predictions to ground truth
    matched_predictions = []
    matched_ground_truth = []
    unmatched_predictions = []
    missing_pieces = []
    
    # predictions is a dictionary with piece_id as key
    for piece_id, pred_info in predictions.items():
        predicted_type = pred_info['type']
        
        if piece_id in ground_truth:
            true_type = ground_truth[piece_id]
            matched_predictions.append(predicted_type)
            matched_ground_truth.append(true_type)
            
            confidence = pred_info['confidence']
            bbox = pred_info['bbox']
            print(f"Piece {piece_id}: True={true_type}, Pred={predicted_type}, "
                  f"Conf={confidence:.3f}, BBox={bbox}")
        else:
            unmatched_predictions.append((piece_id, pred_info))
    
    # Find pieces in ground truth that weren't detected
    detected_ids = set(predictions.keys())
    for piece_id in ground_truth:
        if piece_id not in detected_ids:
            missing_pieces.append((piece_id, ground_truth[piece_id]))
    
    print(f"\nMATCHING RESULTS:")
    print(f"Successfully matched: {len(matched_predictions)}/{len(ground_truth)} pieces")
    print(f"Unmatched predictions: {len(unmatched_predictions)}")
    print(f"Missing pieces: {len(missing_pieces)}")
    
    if unmatched_predictions:
        print("\nUnmatched predictions (extra detections):")
        for piece_id, pred_info in unmatched_predictions:
            print(f"  Piece {piece_id}: {pred_info['type']} (conf={pred_info['confidence']:.3f})")
    
    if missing_pieces:
        print("\nMissing pieces (not detected):")
        for piece_id, piece_type in missing_pieces:
            print(f"  Piece {piece_id}: {piece_type}")
    
    # Calculate performance metrics
    if matched_predictions:
        accuracy = accuracy_score(matched_ground_truth, matched_predictions)
        print(f"\nCLASSIFICATION ACCURACY: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Detailed classification report
        print("\nDETAILED CLASSIFICATION REPORT:")
        print(classification_report(matched_ground_truth, matched_predictions))
        
        # Confusion matrix
        create_confusion_matrix(matched_ground_truth, matched_predictions)
        
        # Per-class accuracy
        print("\nPER-CLASS PERFORMANCE:")
        unique_classes = sorted(set(matched_ground_truth + matched_predictions))
        for class_name in unique_classes:
            true_positives = sum(1 for i, true_class in enumerate(matched_ground_truth) 
                               if true_class == class_name and matched_predictions[i] == class_name)
            total_true = sum(1 for true_class in matched_ground_truth if true_class == class_name)
            total_pred = sum(1 for pred_class in matched_predictions if pred_class == class_name)
            
            if total_true > 0:
                recall = true_positives / total_true
                precision = true_positives / total_pred if total_pred > 0 else 0
                print(f"  {class_name}: Precision={precision:.3f}, Recall={recall:.3f}, "
                      f"Count={total_true}")
    
    # Detection rate
    detection_rate = len(matched_predictions) / len(ground_truth)
    print(f"\nDETECTION RATE: {detection_rate:.3f} ({detection_rate*100:.1f}%)")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

def create_confusion_matrix(true_labels, predicted_labels):
    """Create and display confusion matrix."""
    unique_labels = sorted(set(true_labels + predicted_labels))
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Confusion Matrix - Semantic Chess Piece Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save confusion matrix
    output_path = 'confusion_matrix_detection_output.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    # Change to the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        run_classifier_test()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()