"""
Simple function to run inference on masks.
Easy to integrate with other detection/tracking code.
"""

import cv2
import numpy as np
import pickle
import os

# Import the feature extraction function
from .train_semantic_from_labeled import extract_geometric_features

def predict_pieces_from_semantic(semantic_mask, model_path='semantic_chess_classifier.pkl', piece_value_range=(1, 50), return_details=False):
    # Load model once
    if not hasattr(predict_pieces_from_semantic, '_cached_model'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        predict_pieces_from_semantic._cached_model = {
            'classifier': model_data['classifier'],
            'scaler': model_data['scaler'],
            'piece_types': model_data['piece_types']
        }
    
    model = predict_pieces_from_semantic._cached_model
    classifier = model['classifier']
    scaler = model['scaler']
    
    # Find all unique piece values in the semantic mask
    unique_values = np.unique(semantic_mask)
    piece_values = [v for v in unique_values 
                   if piece_value_range[0] <= v <= piece_value_range[1]]
    
    results = {}
    
    for piece_value in piece_values:
        try:
            # Extract geometric features
            features = extract_geometric_features(semantic_mask, piece_value)
            
            if features is None:
                continue
            
            # Scale features and predict
            features_scaled = scaler.transform(features.reshape(1, -1))
            prediction = classifier.predict(features_scaled)[0]
            probabilities = classifier.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
            
            # Get piece info (bounding box, area, centroid)
            piece_mask = (semantic_mask == piece_value).astype(np.uint8)
            contours, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                main_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(main_contour)
                area = cv2.contourArea(main_contour)
                
                # Calculate centroid
                M = cv2.moments(main_contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                else:
                    cx, cy = x + w//2, y + h//2
                
                # Build result dictionary
                piece_info = {
                    'type': prediction,
                    'confidence': float(confidence),
                    'bbox': (int(x), int(y), int(w), int(h)),
                    'area': float(area),
                    'centroid': (int(cx), int(cy))
                }
                
                # Add detailed info if requested
                if return_details:
                    piece_info['features'] = features
                    piece_info['probabilities'] = {
                        model['piece_types'][i]: float(prob) 
                        for i, prob in enumerate(probabilities)
                    }
                
                results[int(piece_value)] = piece_info
                
        except Exception as e:
            # Skip pieces that cause errors
            print(f"Warning: Could not process piece {piece_value}: {e}")
            continue
    
    return results

def clear_model_cache():
    """Clear the cached model (useful if you want to load a different model)"""
    if hasattr(predict_pieces_from_semantic, '_cached_model'):
        delattr(predict_pieces_from_semantic, '_cached_model')

# Example usage function
def example_usage():
    """Example of how to use the inference function"""
    
    # Load a semantic mask
    semantic_path = "final_chess_dataset/semantics/1.png_mask.png"
    
    if os.path.exists(semantic_path):
        semantic_mask = cv2.imread(semantic_path, cv2.IMREAD_GRAYSCALE)
        
        print("Running inference on semantic mask...")
        results = predict_pieces_from_semantic(semantic_mask, return_details=True)
        
        print(f"\nFound {len(results)} pieces:")
        for piece_id, info in results.items():
            print(f"  Piece {piece_id}: {info['type']} (confidence: {info['confidence']:.3f})")
            print(f"    Bbox: {info['bbox']}, Area: {info['area']:.0f}, Center: {info['centroid']}")
            
            if 'probabilities' in info:
                print(f"    All probabilities: {info['probabilities']}")
    else:
        print(f"Example semantic mask not found: {semantic_path}")

if __name__ == "__main__":
    example_usage()
