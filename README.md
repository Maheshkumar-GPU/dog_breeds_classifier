Dog Breed Classifier (TensorFlow)

This project is a Deep Learning image classification model that predicts dog breeds from images.
It uses Transfer Learning with MobileNetV2 and TensorFlow/Keras.

Features

- Classifies multiple dog breeds
- Uses Transfer Learning (MobileNetV2)
- Custom training on Kaggle dog breed dataset
- Predict breed from new unseen image
- Saves trained model and label mapping

Project Structure

dog-breed-classifier/
│
├── src/                # Dataset loading and model files
├── outputs/            # Saved model and results
├── predict.py          # Predict breed from image
├── train.py            # Train model
├── config.py           # Configuration settings
├── requirements.txt
└── README.md

Dataset

- Dog breed image dataset downloaded from Kaggle
- Images organized by breed folders
- Total classes: ~150+ dog breeds
- Total images: ~17k+

Dataset format:

dataset/
├── beagle/
├── husky/
├── labrador/
├── pug/

Model

- Transfer Learning using MobileNetV2
- Image size: 224x224
- Batch size: 32
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy

Training

Run the following command:

python train.py

Training will:

- Load dataset
- Train model
- Save model to outputs folder
- Save label map JSON

Prediction

Place test image in project folder and run:

python predict.py

Output example:

Predicted Breed: golden_retriever

Outputs

- model.h5 → trained model
- label_map.json → class index mapping
- results.png → training accuracy graph

Notes

- First run downloads MobileNetV2 pretrained weights
- Training may be slow on CPU
- Accuracy depends on dataset quality

Requirements

- Python 3.9+
- TensorFlow
- NumPy
- Matplotlib

Install dependencies:

pip install -r requirements.txt

Future Improvements

- Add FastAPI backend
- Add web UI
- Improve accuracy
- Add confidence score
- Support batch predictions

Author

Mahesh Kumar
