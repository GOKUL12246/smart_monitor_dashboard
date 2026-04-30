"""
Face Recognition Training for Missing Person Detection
========================================================
Trains and improves face recognition models for better matching
"""

import cv2
import numpy as np
import os
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import threading

# Try to import face_recognition
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠️ face_recognition not available - Installing required packages...")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    print("⚠️ scikit-learn not available")


class FaceRecognitionTrainer:
    """Train and improve face recognition models for missing persons"""
    
    def __init__(self, 
                 data_dir: str = "missing_persons_yolo",
                 model_dir: str = "models",
                 train_ratio: float = 0.8):
        """
        Initialize trainer
        
        Args:
            data_dir: Directory with missing person images
            model_dir: Directory to save trained models
            train_ratio: Train/test split ratio
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.train_ratio = train_ratio
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Training data
        self.face_encodings = []
        self.person_labels = []
        self.image_paths = []
        self.training_data = {}  # person_id -> {name, encodings, paths}
        
        # Model components
        self.scaler = None
        self.model = None
        self.label_encoder = {}  # Maps person_id to numeric labels
        
        # Training metadata
        self.training_history = {
            'start_time': None,
            'end_time': None,
            'images_processed': 0,
            'faces_detected': 0,
            'encoding_dimension': 128,
            'model_type': 'KNN + FaceEncoding',
            'accuracy': None,
            'metrics': {}
        }
        
        print(f"✅ FaceRecognitionTrainer initialized")
        print(f"   - Data directory: {data_dir}")
        print(f"   - Model directory: {model_dir}")
        print(f"   - Train/Test ratio: {train_ratio}:{1-train_ratio}")
    
    def load_training_data(self) -> bool:
        """Load images from missing persons database"""
        print("\n📂 Loading training data...")
        
        database_file = os.path.join(self.data_dir, "database.json")
        if not os.path.exists(database_file):
            print("❌ database.json not found!")
            return False
        
        try:
            with open(database_file, 'r') as f:
                database = json.load(f)
            
            processed_count = 0
            for person_id, person_data in database.items():
                person_name = person_data.get('name', person_id)
                image_path = person_data.get('image_path')
                
                if image_path and os.path.exists(image_path):
                    self.image_paths.append((person_id, person_name, image_path))
                    processed_count += 1
            
            print(f"✅ Loaded {processed_count} records from database")
            self.training_history['images_processed'] = processed_count
            return True
            
        except Exception as e:
            print(f"❌ Error loading database: {e}")
            return False
    
    def extract_face_encodings(self) -> bool:
        """Extract face encodings from training images"""
        print("\n🔍 Extracting face encodings...")
        
        if not FACE_RECOGNITION_AVAILABLE:
            print("❌ face_recognition library not available")
            return False
        
        faces_detected = 0
        failed_images = []
        
        for idx, (person_id, person_name, image_path) in enumerate(self.image_paths, 1):
            try:
                # Load image
                image = face_recognition.load_image_file(image_path)
                
                # Find faces in image
                face_locations = face_recognition.face_locations(image, model='hog')
                if not face_locations:
                    print(f"   ⚠️ [{idx}/{len(self.image_paths)}] No faces found: {person_name}")
                    failed_images.append(image_path)
                    continue
                
                # Extract encodings for all faces in image
                face_encodings = face_recognition.face_encodings(image, face_locations)
                
                # Store encodings
                if person_id not in self.training_data:
                    self.training_data[person_id] = {
                        'name': person_name,
                        'encodings': [],
                        'paths': []
                    }
                
                for encoding in face_encodings:
                    self.training_data[person_id]['encodings'].append(encoding)
                    self.training_data[person_id]['paths'].append(image_path)
                    self.face_encodings.append(encoding)
                    self.person_labels.append(person_id)
                    faces_detected += 1
                
                print(f"   ✅ [{idx}/{len(self.image_paths)}] {person_name}: {len(face_encodings)} face(s)")
                
            except Exception as e:
                print(f"   ❌ Error processing {person_name}: {e}")
                failed_images.append(image_path)
        
        self.training_history['faces_detected'] = faces_detected
        
        if failed_images:
            print(f"\n⚠️ Failed to process {len(failed_images)} images")
        
        if faces_detected == 0:
            print("❌ No faces detected in any images!")
            return False
        
        print(f"✅ Extracted {faces_detected} face encodings")
        return True
    
    def augment_encodings(self) -> None:
        """Augment encodings with slight variations for better generalization"""
        print("\n🔄 Augmenting encodings...")
        
        original_count = len(self.face_encodings)
        augmentation_factor = 3
        
        for person_id, data in self.training_data.items():
            original_encodings = data['encodings'].copy()
            
            for encoding in original_encodings:
                # Add slight noise variations
                for _ in range(augmentation_factor - 1):
                    noise = np.random.normal(0, 0.01, encoding.shape)
                    augmented = np.clip(encoding + noise, -1, 1)
                    self.training_data[person_id]['encodings'].append(augmented)
                    self.face_encodings.append(augmented)
                    self.person_labels.append(person_id)
        
        print(f"✅ Augmented encodings: {original_count} -> {len(self.face_encodings)}")
    
    def create_label_encoder(self) -> None:
        """Create mapping between person IDs and numeric labels"""
        print("\n🏷️ Creating label encoder...")
        
        unique_persons = list(set(self.person_labels))
        self.label_encoder = {person_id: idx for idx, person_id in enumerate(unique_persons)}
        
        print(f"✅ Encoded {len(unique_persons)} unique persons:")
        for person_id, label in self.label_encoder.items():
            person_name = self.training_data[person_id]['name']
            print(f"   - {person_name}: {label}")
    
    def train_model(self) -> bool:
        """Train KNN classifier on face encodings"""
        print("\n🤖 Training KNN model...")
        
        try:
            # Convert to numpy arrays
            X = np.array(self.face_encodings)
            y = np.array([self.label_encoder[pid] for pid in self.person_labels])
            
            # Standardize features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train KNN
            self.model = KNeighborsClassifier(n_neighbors=3, weights='distance')
            self.model.fit(X_scaled, y)
            
            # Calculate accuracy on training data
            train_accuracy = self.model.score(X_scaled, y)
            self.training_history['accuracy'] = train_accuracy
            
            print(f"✅ Model trained successfully")
            print(f"   - Training samples: {X.shape[0]}")
            print(f"   - Feature dimension: {X.shape[1]}")
            print(f"   - Training accuracy: {train_accuracy:.2%}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            return False
    
    def save_models(self) -> bool:
        """Save trained models and encodings"""
        print("\n💾 Saving models...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model
            model_path = os.path.join(self.model_dir, f"face_model_{timestamp}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"   ✅ Model saved: {model_path}")
            
            # Save scaler
            scaler_path = os.path.join(self.model_dir, f"scaler_{timestamp}.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"   ✅ Scaler saved: {scaler_path}")
            
            # Save label encoder
            encoder_path = os.path.join(self.model_dir, f"label_encoder_{timestamp}.json")
            
            # Create reverse mapping for encoder (numeric -> person_id)
            reverse_encoder = {v: k for k, v in self.label_encoder.items()}
            encoder_data = {
                'label_encoder': self.label_encoder,
                'reverse_encoder': reverse_encoder,
                'person_names': {k: self.training_data[k]['name'] 
                                for k in self.training_data.keys()}
            }
            
            with open(encoder_path, 'w') as f:
                json.dump(encoder_data, f, indent=2)
            print(f"   ✅ Encoder saved: {encoder_path}")
            
            # Save training data
            data_path = os.path.join(self.model_dir, f"training_data_{timestamp}.pkl")
            with open(data_path, 'wb') as f:
                pickle.dump(self.training_data, f)
            print(f"   ✅ Training data saved: {data_path}")
            
            # Save training history
            history_path = os.path.join(self.model_dir, f"training_history_{timestamp}.json")
            self.training_history['end_time'] = datetime.now().isoformat()
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            print(f"   ✅ History saved: {history_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving models: {e}")
            return False
    
    def generate_report(self) -> None:
        """Generate training report"""
        print("\n" + "="*60)
        print("📊 TRAINING REPORT")
        print("="*60)
        
        print(f"\n🎯 Model Performance:")
        print(f"   - Total images processed: {self.training_history['images_processed']}")
        print(f"   - Faces detected: {self.training_history['faces_detected']}")
        print(f"   - Training samples (with augmentation): {len(self.face_encodings)}")
        print(f"   - Unique persons: {len(self.training_data)}")
        print(f"   - Training accuracy: {self.training_history['accuracy']:.2%}")
        
        print(f"\n👥 Person Details:")
        for person_id, data in sorted(self.training_data.items()):
            num_encoded = len(data['encodings'])
            print(f"   - {data['name']}: {num_encoded} encoded samples")
        
        print(f"\n⚙️ Model Specification:")
        print(f"   - Model type: {self.training_history['model_type']}")
        print(f"   - Encoding dimension: {self.training_history['encoding_dimension']}")
        print(f"   - Training time: {self.training_history['start_time']} to {self.training_history['end_time']}")
        
        print("\n" + "="*60 + "\n")
    
    def train_complete_pipeline(self) -> bool:
        """Run complete training pipeline"""
        print("\n🚀 Starting Face Recognition Training Pipeline...")
        self.training_history['start_time'] = datetime.now().isoformat()
        
        # Step 1: Load data
        if not self.load_training_data():
            return False
        
        # Step 2: Extract encodings
        if not self.extract_face_encodings():
            return False
        
        # Step 3: Augment encodings
        self.augment_encodings()
        
        # Step 4: Create label encoder
        self.create_label_encoder()
        
        # Step 5: Train model
        if not self.train_model():
            return False
        
        # Step 6: Save models
        if not self.save_models():
            return False
        
        # Step 7: Generate report
        self.generate_report()
        
        print("✅ Training pipeline completed successfully!\n")
        return True


def main():
    """Main training script"""
    trainer = FaceRecognitionTrainer(
        data_dir="missing_persons_yolo",
        model_dir="models",
        train_ratio=0.8
    )
    
    success = trainer.train_complete_pipeline()
    
    if success:
        print("🎉 Face recognition model trained and saved!")
        print(f"📁 Models location: {trainer.model_dir}/")
    else:
        print("❌ Training failed!")


if __name__ == "__main__":
    main()
