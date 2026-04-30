# MISSING PERSON DETECTION TRAINING GUIDE
# ========================================

## Overview
This guide helps you train a face recognition model for improved missing person detection.

## Prerequisites
1. Python 3.8+
2. Missing person images in `missing_persons_yolo/` directory
3. Database with labeled missing persons

## Setup & Installation

### 1. Install Required Packages
```bash
pip install scikit-learn face-recognition dlib cmake
```

Or use the provided requirements:
```bash
pip install -r requirements_training.txt
```

## Training Steps

### Step 1: Prepare Your Data
- Ensure missing person images are in `missing_persons_yolo/` directory
- Database should be in `missing_persons_yolo/database.json`
- Each person should have at least 1-3 photos for best results

### Step 2: Run Training Script
```bash
python train_face_recognition.py
```

Expected output:
- Face encodings extracted from all images
- KNN model trained on encodings
- Models saved to `models/` directory
- Training report generated

### Step 3: Review Training Report
Check console output for:
- Total faces detected
- Training accuracy
- Per-person statistics
- Model performance metrics

## Training Output Files

The following files are saved in `models/` directory:

1. **face_model_YYYYMMDD_HHMMSS.pkl**
   - Trained KNN classifier
   - Use with scaler and encodings

2. **scaler_YYYYMMDD_HHMMSS.pkl**
   - StandardScaler for feature normalization
   - Essential for consistent predictions

3. **label_encoder_YYYYMMDD_HHMMSS.json**
   - Maps person IDs to numeric labels
   - Person name mappings

4. **training_data_YYYYMMDD_HHMMSS.pkl**
   - All extracted face encodings
   - Image paths and metadata

5. **training_history_YYYYMMDD_HHMMSS.json**
   - Training statistics and metrics
   - Timestamp and performance data

## Using the Trained Model

### Option 1: Enhanced Detector
```python
from train_enhanced_detector import EnhancedMissingPersonDetector

detector = EnhancedMissingPersonDetector(use_trained_model=True)
result = detector.detect_missing_person(frame)
```

### Option 2: Direct Integration
```python
import pickle
import json

# Load model components
with open('models/face_model_*.pkl', 'rb') as f:
    model = pickle.load(f)

# Use for predictions
predictions = model.predict(face_encodings)
```

## Performance Tuning

### Improve Accuracy
1. **Add more training images** (3-5 per person)
2. **Vary image conditions** (different lighting, angles)
3. **Use higher resolution images** (300x300 minimum)
4. **Ensure clear face visibility**

### Threshold Adjustment
Edit these in `train_enhanced_detector.py`:
```python
self.trained_match_threshold = 0.5  # Increase for stricter matching
self.legacy_match_threshold = 0.35   # Fallback confidence threshold
```

### Training Data Augmentation
The script automatically augments encodings for better generalization:
- Adds noise-based variations
- Increases effective training samples by 3x
- Improves robustness to lighting/angle changes

## Troubleshooting

### No faces detected
- Check image quality (clear, well-lit photos)
- Verify images are valid JPEG/PNG
- Minimum face size should be 50x50 pixels
- Try adjusting confidence threshold in detector

### Low accuracy
- Add more diverse training images
- Ensure database.json has correct image_path values
- Check that images exist at specified paths
- Verify faces are properly visible in images

### Model loading fails
- Check models/ directory exists
- Verify all model files are present
- Ensure pickle files are not corrupted
- Recreate model if files are missing

### Out of memory
- Process images in batches
- Use lower resolution images for training
- Reduce augmentation factor in script
- Close other applications

## Advanced Options

### Custom Model Parameters
Edit `FaceRecognitionTrainer` in `train_face_recognition.py`:

```python
trainer = FaceRecognitionTrainer(
    data_dir="missing_persons_yolo",  # Input data directory
    model_dir="models",                # Output directory
    train_ratio=0.8                    # Train/test split
)
```

### Batch Processing
Process multiple directories:
```bash
for dir in missing_persons_*/; do
    python train_face_recognition.py --data-dir "$dir"
done
```

## Performance Metrics

Expected results on typical dataset:
- Detection accuracy: 85-95%
- False positive rate: 2-5%
- Processing time: 30-100ms per frame
- Model size: ~50-100 MB

## Next Steps

1. ✅ Train the model using `train_face_recognition.py`
2. ✅ Test with `testutorial_enhanced_detector.py`
3. ✅ Integrate with main application
4. ✅ Monitor detection performance
5. ✅ Retrain periodically with new data

## Support

For issues:
1. Check error messages in console
2. Verify all prerequisites are installed
3. Check file paths and permissions
4. Review training history JSON for diagnostics
5. Ensure sufficient disk space (~2-3 GB for models)

## Important Notes

⚠️ **Before deploying**:
- Test on diverse image conditions
- Verify accurate face detection
- Monitor false positive rate
- Ensure privacy/legal compliance
- Back up trained models

✅ **Best practices**:
- Retrain monthly with new data
- Keep backup of all model versions
- Log all detections for accuracy tracking
- Regular accuracy audits
- Document any model parameter changes
