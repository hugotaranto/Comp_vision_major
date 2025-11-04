"""
Semantic Chess Piece Classifier Trainer - Full Dataset Version
==============================================================

Trains a classifier using 100% of available data for live demo purposes.
No train/test split - uses all labeled data to create the best possible model.
"""

import cv2
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

# Import feature extraction from main training script
from train_semantic_from_labeled import extract_geometric_features

def train_full_dataset_classifier():
    """
    Train classifier on 100% of available data for live demo.
    Uses cross-validation to estimate performance without holding out test data.
    """
    
    print("=" * 70)
    print("🚀 SEMANTIC CHESS CLASSIFIER - FULL DATASET TRAINING")
    print("=" * 70)
    print("Training on 100% of available data for live demo...")
    
    # Set paths
    dataset_path = "./new_dataset"
    labels_file = os.path.join(dataset_path, 'labels.csv')
    
    if not os.path.exists(labels_file):
        print(f"❌ Labels file not found: {labels_file}")
        return
    
    # Load labels
    labels_df = pd.read_csv(labels_file)
    print(f"✓ Loaded {len(labels_df)} labeled pieces")
    
    # Identify data sources
    original_count = len(labels_df[~labels_df['image_name'].str.contains('detection_output', na=False)])
    detection_count = len(labels_df[labels_df['image_name'].str.contains('detection_output', na=False)])
    
    print(f"   Original training data: {original_count} pieces")
    print(f"   Detection output data: {detection_count} pieces")
    
    # Group by piece type
    piece_counts = labels_df['piece_type'].value_counts()
    print("\n📊 Full Dataset Distribution:")
    for piece_type, count in piece_counts.items():
        print(f"   {piece_type}: {count}")
    
    # Extract features for each labeled piece
    X = []
    y = []
    failed_extractions = 0
    
    print(f"\n🔄 Extracting features from {len(labels_df)} pieces...")
    
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
            print(f"⚠️  Semantic file not found: {image_name}")
            failed_extractions += 1
            continue
        
        # Load semantic image
        semantic_img = cv2.imread(semantic_path, cv2.IMREAD_GRAYSCALE)
        if semantic_img is None:
            print(f"⚠️  Could not load: {semantic_path}")
            failed_extractions += 1
            continue
        
        # Extract features
        features = extract_geometric_features(semantic_img, piece_value)
        if features is not None:
            X.append(features)
            y.append(piece_type)
        else:
            failed_extractions += 1
    
    if failed_extractions > 0:
        print(f"⚠️  Failed to extract features from {failed_extractions} pieces")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✓ Successfully extracted features from {len(X)} pieces")
    print(f"   Feature dimension: {X.shape[1]}")
    
    if len(X) == 0:
        print("❌ No features extracted! Check your data.")
        return
    
    # Scale features
    print(f"\n⚙️  Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define classifiers for evaluation
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
            C=20, 
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
        'K-NN (k=5)': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        )
    }
    
    # Evaluate classifiers using cross-validation
    print(f"\n🔬 Evaluating classifiers using 5-fold cross-validation...")
    cv_scores = {}
    
    for name, classifier in classifiers.items():
        print(f"   Testing {name}...")
        scores = cross_val_score(classifier, X_scaled, y, cv=5, scoring='accuracy')
        cv_scores[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        print(f"      CV Accuracy: {scores.mean():.4f} (±{scores.std()*2:.4f})")
    
    # Select best classifier (prioritize accuracy, then lower uncertainty)
    def model_selection_score(name):
        mean_acc = cv_scores[name]['mean']
        std_acc = cv_scores[name]['std']
        # Primary: accuracy, Secondary: lower uncertainty (negative std for max function)
        score = (mean_acc, -std_acc)
        print(f"      {name}: score = {score} (acc={mean_acc:.4f}, std={std_acc:.4f})")
        return score
    
    print(f"\n🔍 Model Selection Scores:")
    best_name = max(cv_scores.keys(), key=model_selection_score)
    best_classifier = classifiers[best_name]
    best_score = cv_scores[best_name]['mean']
    
    print(f"\n🏆 Best Classifier: {best_name}")
    print(f"   CV Accuracy: {best_score:.4f} (±{cv_scores[best_name]['std']*2:.4f})")
    
    # Train final model on ALL data
    print(f"\n🚀 Training final model on 100% of data...")
    final_classifier = classifiers[best_name]
    final_classifier.fit(X_scaled, y)
    
    # Get unique piece types for model metadata
    unique_pieces = sorted(set(y))
    
    print(f"✓ Final model trained on {len(X)} samples")
    print(f"   Piece types: {unique_pieces}")
    
    # Save the full model
    model_filename = f'classifier_on_new_data.pkl'
    
    model_data = {
        'classifier': final_classifier,
        'scaler': scaler,
        'piece_types': unique_pieces,
        'model_name': best_name,
        'cv_accuracy': best_score,
        'cv_std': cv_scores[best_name]['std'],
        'training_samples': len(X),
        'feature_count': X.shape[1],
        'data_distribution': dict(piece_counts),
        'original_data_count': original_count,
        'detection_data_count': detection_count
    }
    
    with open(model_filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n💾 Model saved: {model_filename}")
    
    # Show detailed model specifications
    print(f"\n⚙️  Final Model Specifications:")
    if hasattr(final_classifier, 'n_estimators'):
        print(f"   Algorithm: Random Forest")
        print(f"   Trees: {final_classifier.n_estimators}")
        print(f"   Max Depth: {final_classifier.max_depth}")
        print(f"   Min Samples Split: {final_classifier.min_samples_split}")
        print(f"   Class Weight: {final_classifier.class_weight}")
    elif hasattr(final_classifier, 'kernel'):
        print(f"   Algorithm: Support Vector Machine")
        print(f"   Kernel: {final_classifier.kernel}")
        if hasattr(final_classifier, 'C'):
            print(f"   C Parameter: {final_classifier.C}")
        if hasattr(final_classifier, 'gamma'):
            print(f"   Gamma: {final_classifier.gamma}")
        print(f"   Class Weight: {final_classifier.class_weight}")
    elif hasattr(final_classifier, 'n_neighbors'):
        print(f"   Algorithm: K-Nearest Neighbors")
        print(f"   K: {final_classifier.n_neighbors}")
        print(f"   Weights: {final_classifier.weights}")
    
    # Show cross-validation results for all models
    print(f"\n📊 All Model Performance (Cross-Validation):")
    sorted_models = sorted(cv_scores.items(), key=lambda x: x[1]['mean'], reverse=True)
    for name, scores in sorted_models:
        print(f"   {name}: {scores['mean']:.4f} (±{scores['std']*2:.4f})")
    
    # Performance analysis
    print(f"\n📈 Performance Analysis:")
    print(f"   Total training samples: {len(X)}")
    print(f"   Cross-validation accuracy: {best_score:.4f}")
    if best_score >= 0.95:
        print(f"   🟢 Excellent performance (≥95%)")
    elif best_score >= 0.90:
        print(f"   🟡 Good performance (≥90%)")
    elif best_score >= 0.80:
        print(f"   🟠 Fair performance (≥80%)")
    else:
        print(f"   🔴 Poor performance (<80%)")
    
    print(f"\n✅ Full dataset training complete!")
    print(f"   Use {model_filename} for live demo")
    print(f"   Expected accuracy: {best_score:.1%}")
    
    return model_filename, best_score

if __name__ == "__main__":
    # Change to the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir:
        os.chdir(script_dir)
    
    try:
        model_file, accuracy = train_full_dataset_classifier()
        
        print(f"\n🎯 DEMO-READY MODEL CREATED!")
        print(f"   File: {model_file}")
        print(f"   Expected Accuracy: {accuracy:.1%}")
        print(f"   Ready for live inference!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
