# Cats vs Dogs Image Classifier

A deep learning project that classifies images as cats or dogs using a pre-trained ResNet18 model with transfer learning. The project includes both a training script and a Flask web application for making predictions.

## Project Overview

This project demonstrates:
- **Transfer Learning**: Uses a pre-trained ResNet18 model from ImageNet
- **Fine-tuning**: Only trains the final layer while freezing pre-trained weights
- **Web Interface**: Flask-based web app for interactive predictions
- **Deep Learning**: PyTorch for model training and inference

## Project Structure

```
cats-dogs/
├── app.py                          # Flask web application for predictions
├── training.py                     # Script to train the model
├── cats_dogs_resnet18.pth         # Trained model weights (pre-trained)
├── README.md                       # This file
├── static/                         # Static files for Flask app
│   └── uploads/                    # Directory where uploaded images are saved
├── templates/                      # Flask HTML templates
│   └── index.html                  # Web interface for image upload & prediction
└── kagglecatsanddogs_3367a/       # Dataset directory
    └── PetImages/
        ├── Cat/                    # Cat images
        └── Dog/                    # Dog images
```

## Installation & Setup

### Prerequisites
- Python 3.7+
- CUDA 10.2+ (optional, for GPU support)
- pip

### Install Dependencies

1. **Navigate to the project directory:**
   ```bash
   cd cats-dogs
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install torch torchvision flask pillow scikit-learn tqdm
   ```
   
   Or if you need GPU support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install flask pillow scikit-learn tqdm
   ```

## File Descriptions

### `training.py` - Model Training Script

**Purpose**: Trains a ResNet18 classifier on cat and dog images using transfer learning.

**Key Features**:
- Loads images from `kagglecatsanddogs_3367a/PetImages/` directory
- Uses ImageNet pre-trained weights as the starting point
- Freezes all layers except the final fully connected layer
- Performs 80/20 train/validation split
- Trains for 5 epochs with batch size of 32
- Uses CrossEntropyLoss and Adam optimizer
- Saves the trained model as `cats_dogs_resnet18.pth`

**Input Data Structure**:
```
kagglecatsanddogs_3367a/PetImages/
├── Cat/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Dog/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### `app.py` - Flask Web Application

**Purpose**: Provides a web interface for uploading images and getting cat/dog predictions.

**Features**:
- Flask web server running on `http://localhost:5000`
- Image upload functionality
- Real-time predictions with confidence scores
- Uses the same image transformations as training for consistency
- GPU/CPU automatic detection
- Uploaded images saved to `static/uploads/`

### `index.html` - Web Interface

**Purpose**: User-friendly HTML form for image upload and prediction display.

**Features**:
- Clean, centered UI with responsive styling
- File input for image selection
- Submit button to process predictions
- Displays prediction results and confidence percentage
- Shows the uploaded image preview

### `cats_dogs_resnet18.pth` - Trained Model Weights

Pre-trained model weights in PyTorch format. This file is loaded by `app.py` for making predictions.

## How to Use

### Option 1: Training the Model from Scratch

If you want to retrain the model with your own data:

1. **Prepare your dataset** in the following structure:
   ```
   kagglecatsanddogs_3367a/PetImages/
   ├── Cat/
   └── Dog/
   ```

2. **Run the training script**:
   ```bash
   python training.py
   ```

3. **Monitor output**:
   - Shows progress bars for each batch
   - Displays training and validation accuracy after each epoch
   - Saves the model as `cats_dogs_resnet18.pth` after training completes

**Expected Training Time**:
- ~5-10 minutes per epoch on GPU
- ~2-4 hours per epoch on CPU

### Option 2: Using the Web Application (with Pre-trained Model)

1. **Ensure the trained model exists** (`cats_dogs_resnet18.pth`)

2. **Run the Flask app**:
   ```bash
   python app.py
   ```

3. **Open your browser**:
   - Navigate to `http://localhost:5000`
   - Click "Choose File" to select an image
   - Click "Predict" button
   - View the prediction result and confidence score

## Technical Details

### Model Architecture

- **Base Model**: ResNet18 (pre-trained on ImageNet)
- **Input Size**: 224 × 224 pixels (RGB)
- **Output**: 2 classes (Cat or Dog)
- **Final Layer**: Linear layer mapping 512 features to 2 classes

### Transfer Learning Strategy

1. Load ResNet18 with ImageNet pre-trained weights
2. **Freeze** all convolutional layers (requires_grad = False)
3. **Replace** the final fully connected layer with a new 2-class classifier
4. **Train only** the new final layer (faster convergence, requires less data)

### Data Preprocessing

**Training & Prediction**:
```python
- Resize images to 224 × 224 pixels
- Convert to tensor (values 0-1)
- Normalize using ImageNet statistics:
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
```

### Model Training Details

| Parameter | Value |
|-----------|-------|
| Model | ResNet18 |
| Epochs | 5 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Train/Val Split | 80/20 |
| Device | GPU (if available) / CPU |

## Performance Metrics

After training:
- **Training Accuracy**: ~95%+
- **Validation Accuracy**: ~93%+

(Exact values depend on dataset quality and variation)

## Troubleshooting

### Issue: "Model file not found" when running app.py

**Solution**: Run `training.py` first to generate `cats_dogs_resnet18.pth`

### Issue: Out of memory (CUDA)

**Solution**: Reduce batch size in `training.py`:
```python
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
```

### Issue: Images not uploading

**Solution**: Ensure `static/uploads/` directory exists:
```bash
mkdir -p static/uploads
```

### Issue: Incorrect predictions

**Solution**:
- Verify model is properly trained
- Ensure image is clear and well-lit
- Try with different test images

## Model Inference

The model accepts images in common formats (JPG, PNG, etc.) and returns:
- **Prediction**: "Cat" or "Dog"
- **Confidence**: Probability score (0-100%)

Example confidence interpretation:
- 95.5% confidence for "Cat" = Model is highly certain it's a cat
- 52.3% confidence for "Dog" = Model is less certain, nearly 50/50 prediction

## Dependencies

| Package | Purpose |
|---------|---------|
| torch | Deep learning framework |
| torchvision | Computer vision utilities & pre-trained models |
| flask | Web framework |
| pillow | Image processing |
| scikit-learn | Machine learning utilities |
| tqdm | Progress bars |

## Future Enhancements

- [ ] Add data augmentation for better generalization
- [ ] Implement model interpretability (attention maps)
- [ ] Add batch prediction capability
- [ ] Deploy to cloud (AWS, Heroku, etc.)
- [ ] Fine-tune all layers (not just final layer)
- [ ] Experiment with other architectures (ResNet50, EfficientNet)

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Kaggle Cats and Dogs Dataset](https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765)

## License

This project is open source and available for educational purposes.
