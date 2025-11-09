"""
Trains a classifier using geometric features extracted from labeled semantic masks.
Focuses purely on shape/geometric patterns, ignoring color information.
"""

import cv2
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def extract_geometric_features(semantic_img, piece_value):
    """
    Extract geometric features from a semantic piece region.
    Focus on shape, size, and geometric patterns only.
    """
    # Create binary mask for this piece
    piece_mask = (semantic_img == piece_value).astype(np.uint8)
    
    if np.sum(piece_mask) == 0:
        return None
    
    # Find contours
    contours, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    
    # Get the largest contour (main piece)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Initialize feature vector
    features = []
    
    # === BASIC GEOMETRIC FEATURES ===
    
    # 1. Area and Perimeter
    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, True)
    features.extend([area, perimeter])
    
    # 2. Bounding Rectangle Features
    x, y, w, h = cv2.boundingRect(main_contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0
    features.extend([w, h, aspect_ratio, extent])
    
    # 3. Circularity and Compactness
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        compactness = area / (perimeter * perimeter) if perimeter > 0 else 0
    else:
        circularity = 0
        compactness = 0
    features.extend([circularity, compactness])
    
    # 4. Solidity (Convexity)
    hull = cv2.convexHull(main_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    features.append(solidity)
    
    # === ADVANCED SHAPE FEATURES ===
    
    # 5. Convexity Defects (important for knight vs other pieces)
    hull_indices = cv2.convexHull(main_contour, returnPoints=False)
    if len(hull_indices) > 3:
        defects = cv2.convexityDefects(main_contour, hull_indices)
        if defects is not None and len(defects) > 0:
            num_defects = len(defects)
            max_defect = np.max(defects[:, 0, 3])
            avg_defect = np.mean(defects[:, 0, 3])
            defect_ratio = max_defect / area if area > 0 else 0
        else:
            num_defects = 0
            max_defect = 0
            avg_defect = 0
            defect_ratio = 0
    else:
        num_defects = 0
        max_defect = 0
        avg_defect = 0
        defect_ratio = 0
    features.extend([num_defects, max_defect, avg_defect, defect_ratio])
    
    # 6. Moment-based Shape Descriptors
    moments = cv2.moments(main_contour)
    if moments['m00'] > 0:
        # Hu Moments (7 invariant shape descriptors)
        hu_moments = cv2.HuMoments(moments)
        # Convert to log scale for better numerical stability
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments + 1e-10))
        features.extend(hu_moments.flatten())
        
        # Centroid
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        
        # Normalized central moments (scale invariant)
        mu20 = moments['mu20'] / moments['m00']
        mu02 = moments['mu02'] / moments['m00']
        mu11 = moments['mu11'] / moments['m00']
        features.extend([mu20, mu02, mu11])
    else:
        # Add zeros if moments can't be computed
        features.extend([0] * 7)  # Hu moments
        cx, cy = x + w//2, y + h//2  # Fallback centroid
        features.extend([0, 0, 0])  # Central moments
    
    # 7. Eccentricity and Orientation
    if len(main_contour) >= 5:  # Need at least 5 points for fitEllipse
        try:
            ellipse = cv2.fitEllipse(main_contour)
            (center_x, center_y), (minor_axis, major_axis), angle = ellipse
            eccentricity = np.sqrt(1 - (minor_axis/major_axis)**2) if major_axis > 0 else 0
            orientation = angle
            axis_ratio = minor_axis / major_axis if major_axis > 0 else 0
        except:
            eccentricity = 0
            orientation = 0
            axis_ratio = 1
    else:
        eccentricity = 0
        orientation = 0
        axis_ratio = 1
    features.extend([eccentricity, orientation, axis_ratio])
    
    # 8. Contour Approximation Features
    # Douglas-Peucker approximation with different epsilons
    epsilon_ratios = [0.01, 0.02, 0.05]
    for eps_ratio in epsilon_ratios:
        epsilon = eps_ratio * perimeter
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        num_vertices = len(approx)
        features.append(num_vertices)
    
    # 9. Distance Transform Features
    # Create distance transform of the piece
    dist_transform = cv2.distanceTransform(piece_mask, cv2.DIST_L2, 5)
    max_distance = np.max(dist_transform)
    mean_distance = np.mean(dist_transform[piece_mask > 0])
    std_distance = np.std(dist_transform[piece_mask > 0])
    features.extend([max_distance, mean_distance, std_distance])
    
    # 10. Skeleton/Thinning Features
    # Simple skeleton approximation
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skeleton = cv2.morphologyEx(piece_mask, cv2.MORPH_HITMISS, kernel)
    skeleton_area = np.sum(skeleton > 0)
    skeleton_ratio = skeleton_area / area if area > 0 else 0
    features.append(skeleton_ratio)
    
    # 11. Fourier Descriptors (simplified)
    # Get contour points and compute basic frequency features
    contour_points = main_contour[:, 0, :]  # Remove extra dimension
    if len(contour_points) > 10:
        # Resample to fixed number of points
        num_points = min(64, len(contour_points))
        indices = np.linspace(0, len(contour_points)-1, num_points, dtype=int)
        resampled = contour_points[indices]
        
        # Compute centroid-relative coordinates
        centroid = np.mean(resampled, axis=0)
        relative_coords = resampled - centroid
        
        # Convert to complex numbers and apply FFT
        complex_coords = relative_coords[:, 0] + 1j * relative_coords[:, 1]
        fft_coeffs = np.fft.fft(complex_coords)
        
        # Use magnitude of first few coefficients (excluding DC component)
        fourier_features = np.abs(fft_coeffs[1:6])  # First 5 AC coefficients
        features.extend(fourier_features)
    else:
        features.extend([0] * 5)  # Fallback if too few points
    
    # 12. Radial Distribution Features
    # Analyze distance distribution from centroid
    if moments['m00'] > 0:
        # Sample points on contour
        contour_points = main_contour[:, 0, :]
        distances = np.sqrt((contour_points[:, 0] - cx)**2 + (contour_points[:, 1] - cy)**2)
        
        if len(distances) > 0:
            radial_mean = np.mean(distances)
            radial_std = np.std(distances)
            radial_max = np.max(distances)
            radial_min = np.min(distances)
            radial_range = radial_max - radial_min
            radial_cv = radial_std / radial_mean if radial_mean > 0 else 0
        else:
            radial_mean = radial_std = radial_max = radial_min = radial_range = radial_cv = 0
    else:
        radial_mean = radial_std = radial_max = radial_min = radial_range = radial_cv = 0
    
    features.extend([radial_mean, radial_std, radial_max, radial_min, radial_range, radial_cv])
    
    return np.array(features)

def load_labeled_data(dataset_path):
    "Load labeled semantic data and extract features"
    labels_file = os.path.join(dataset_path, 'labels_final.csv')
    
    if not os.path.exists(labels_file):
        print(f"Labels file not found: {labels_file}")
        print("Please run label_semantics.py first to create labeled data.")
        return None, None, None
    
    # Load labels
    labels_df = pd.read_csv(labels_file)
    print(f"Loaded {len(labels_df)} labeled pieces")
    
    # Group by piece type
    piece_counts = labels_df['piece_type'].value_counts()
    print("\nLabeled Data Distribution:")
    for piece_type, count in piece_counts.items():
        print(f"   {piece_type}: {count}")
    
    # Extract features for each labeled piece
    X = []
    y = []
    image_info = []
    
    # Process each labeled piece
    for idx, row in labels_df.iterrows():
        image_name = row['image_name']
        piece_value = row['piece_value']
        piece_type = row['piece_type']
        
        # Find the semantic image file
        semantic_path = None
        for root, dirs, files in os.walk(dataset_path):
            if 'semantics' in os.path.basename(root) and image_name in files:
                semantic_path = os.path.join(root, image_name)
                break
        
        if semantic_path is None:
            print(f"Semantic file not found: {image_name}")
            continue
        
        # Load semantic image
        semantic_img = cv2.imread(semantic_path, cv2.IMREAD_GRAYSCALE)
        if semantic_img is None:
            print(f"Could not load: {semantic_path}")
            continue
        
        # Extract features
        features = extract_geometric_features(semantic_img, piece_value)
        if features is not None:
            X.append(features)
            y.append(piece_type)
            image_info.append({
                'image_name': image_name,
                'piece_value': piece_value,
                'semantic_path': semantic_path
            })
        else:
            print(f"Could not extract features for {image_name}, piece {piece_value}")
    
    if len(X) == 0:
        print("No valid features extracted!")
        return None, None, None
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n✓ Extracted {len(X)} feature vectors with {X.shape[1]} features each")
    
    return X, y, image_info

def train_classifiers(X, y, image_info):
    "Train multiple classifiers and return the best one"

    print(f"\nTraining classifiers on {len(X)} samples...")
    
    # Separate original data from detection_output data
    original_indices = []
    detection_indices = []
    
    for i, info in enumerate(image_info):
        if 'detection_output' in info['semantic_path']:
            detection_indices.append(i)
        else:
            original_indices.append(i)
    
    print(f"   Original data: {len(original_indices)} samples")
    print(f"   Detection output data: {len(detection_indices)} samples")
    
    # Convert to numpy arrays for indexing
    X = np.array(X)
    y = np.array(y)
    
    if len(original_indices) > 0:
        # Split only the original data 80/20
        X_original = X[original_indices]
        y_original = y[original_indices]
        
        X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
            X_original, y_original, test_size=0.2, random_state=42, stratify=y_original
        )
        
        # Add ALL detection_output data to training set
        if len(detection_indices) > 0:
            X_detection = X[detection_indices]
            y_detection = y[detection_indices]
            
            X_train = np.vstack([X_orig_train, X_detection])
            y_train = np.hstack([y_orig_train, y_detection])
        else:
            X_train = X_orig_train
            y_train = y_orig_train
        
        # Test set contains only original data
        X_test = X_orig_test
        y_test = y_orig_test
        
    else:
        # If no original data, use detection_output for both (shouldn't happen)
        print("Warning: No original data found, using detection_output for train/test split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
    print(f"   Final training set: {len(X_train)} samples (includes {len(detection_indices)} detection_output)")
    print(f"   Final test set: {len(X_test)} samples (original data only)")
    
    # Verify detection_output data is only in training
    detection_in_test = sum(1 for i in range(len(X_test)) if any(
        np.array_equal(X_test[i], X[j]) for j in detection_indices
    ))
    print(f"   Detection_output samples in test set: {detection_in_test} (should be 0)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, 
            max_depth=20,
            min_samples_split=5,
            random_state=42, 
            class_weight='balanced'
        ),
        'SVM (RBF)': SVC(
            kernel='rbf', 
            C=10, 
            gamma='scale', 
            class_weight='balanced', 
            probability=True,
            random_state=42
        ),
        'SVM (Linear)': SVC(
            kernel='linear', 
            C=1, 
            class_weight='balanced', 
            probability=True,
            random_state=42
        ),
        'K-NN': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        )
    }
    
    best_accuracy = 0
    best_classifier = None
    best_name = None
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\n   Training {name}...")
        
        # Show algorithm specifications
        if name == 'Random Forest':
            print(f"      • Trees: {clf.n_estimators}, Max depth: {clf.max_depth}, Min samples split: {clf.min_samples_split}")
        elif name == 'SVM (RBF)':
            print(f"      • Kernel: RBF, C: {clf.C}, Gamma: {clf.gamma}")
        elif name == 'SVM (Linear)':
            print(f"      • Kernel: Linear, C: {clf.C}")
        elif name == 'K-NN':
            print(f"      • Neighbors: {clf.n_neighbors}, Weights: {clf.weights}")
        
        # Train
        clf.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'classifier': clf,
            'predictions': y_pred,
            'true_labels': y_test
        }
        
        print(f"      ✓ Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_classifier = clf
            best_name = name
    
    print(f"\nBest classifier: {best_name} with accuracy {best_accuracy:.4f}")
    
    # Show detailed results for best classifier
    print(f"\nDetailed Results for {best_name}:")
    print(classification_report(results[best_name]['true_labels'], results[best_name]['predictions']))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(results[best_name]['true_labels'], results[best_name]['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f'Confusion Matrix - {best_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return best_classifier, scaler, best_name, best_accuracy

def save_model(classifier, scaler, model_name, accuracy, feature_count):
    "Save the trained model"
    model_data = {
        'classifier': classifier,
        'scaler': scaler,
        'model_name': model_name,
        'accuracy': accuracy,
        'feature_count': feature_count,
        'piece_types': ['pawn', 'rook', 'knight', 'bishop', 'queen', 'king']
    }
    
    model_path = 'semantic_chess_classifier.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Model saved to {model_path}")
    return model_path

def main():
    "Main training function"
    dataset_path = "final_chess_dataset"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset folder not found: {dataset_path}")
        return

    print("Semantic Chess Piece Classifier Training")
    print("=" * 50)
    
    # Load labeled data
    print("Loading labeled data...")
    X, y, image_info = load_labeled_data(dataset_path)
    
    if X is None:
        return
    
    # Check if we have enough data
    unique_classes = np.unique(y)
    min_samples = min([np.sum(y == cls) for cls in unique_classes])
    
    if min_samples < 2:
        print(f"Warning: Some classes have very few samples (min: {min_samples})")
        print("Consider labeling more data for better results.")
    
    if len(X) < 10:
        print(f"Warning: Very few total samples ({len(X)})")
        print("Consider labeling more data for reliable training.")
    
    # Train classifiers
    classifier, scaler, model_name, accuracy = train_classifiers(X, y, image_info)
    
    # Save model
    if classifier is not None:
        model_path = save_model(classifier, scaler, model_name, accuracy, X.shape[1])
        
        print(f"\nTraining Complete!")
        print(f"   Best Model: {model_name}")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Model File: {model_path}")
        print(f"   Feature Count: {X.shape[1]}")
        
        # Show detailed best model specifications
        print(f"\nBest Model Specifications:")
        if model_name == 'Random Forest':
            print(f"   • Algorithm: Random Forest Classifier")
            print(f"   • Number of trees (n_estimators): {classifier.n_estimators}")
            print(f"   • Maximum depth: {classifier.max_depth}")
            print(f"   • Minimum samples per split: {classifier.min_samples_split}")
            print(f"   • Class weighting: balanced")
            print(f"   • Random state: {classifier.random_state}")
        elif model_name == 'SVM (RBF)':
            print(f"   • Algorithm: Support Vector Machine")
            print(f"   • Kernel: Radial Basis Function (RBF)")
            print(f"   • Regularization parameter (C): {classifier.C}")
            print(f"   • Gamma: {classifier.gamma}")
            print(f"   • Class weighting: balanced")
            print(f"   • Random state: {classifier.random_state}")
        elif model_name == 'SVM (Linear)':
            print(f"   • Algorithm: Support Vector Machine") 
            print(f"   • Kernel: Linear")
            print(f"   • Regularization parameter (C): {classifier.C}")
            print(f"   • Class weighting: balanced")
            print(f"   • Random state: {classifier.random_state}")
        elif model_name == 'K-NN':
            print(f"   • Algorithm: K-Nearest Neighbors")
            print(f"   • Number of neighbors (K): {classifier.n_neighbors}")
            print(f"   • Distance weighting: {classifier.weights}")
            print(f"   • Distance metric: {getattr(classifier, 'metric', 'minkowski')}")
        
        # Show feature importance for Random Forest
        if hasattr(classifier, 'feature_importances_'):
            print(f"\nTop 10 Most Important Features:")
            feature_importance = classifier.feature_importances_
            top_indices = np.argsort(feature_importance)[-10:][::-1]
            for i, idx in enumerate(top_indices):
                print(f"   {i+1}. Feature {idx}: {feature_importance[idx]:.4f}")
        
        # Show data preprocessing info
        print(f"\nData Processing:")
        print(f"   • Feature scaling: StandardScaler (mean=0, std=1)")
        print(f"   • Train/test split: 80/20 (stratified)")
        print(f"   • Cross-validation: None (single split)")
        print(f"   • Class balancing: Automatic via class_weight='balanced'")
    
    else:
        print("Training failed!")

if __name__ == "__main__":
    main()