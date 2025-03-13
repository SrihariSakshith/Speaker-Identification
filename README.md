# Speaker Identification

This project implements a deep learning-based speaker identification system. It processes audio recordings, extracts features using MFCC, and classifies speakers using an LSTM-based neural network. The system is designed for real-time speaker identification using pre-recorded audio inputs.

## Features
- **Speaker Identification:** Classifies and identifies speakers based on their unique voice characteristics.
- **MFCC Feature Extraction:** Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio for feature representation.
- **LSTM-Based Model:** Uses a Long Short-Term Memory (LSTM) network for accurate speaker classification.
- **Real-Time Prediction:** Supports real-time audio processing for live speaker identification.
- **Dataset Handling:** Allows training with custom datasets containing multiple speakers.

## Prerequisites
- Python 3.7+
- TensorFlow/Keras
- NumPy
- Librosa (for audio processing)
- Pandas
- Matplotlib (for visualization)
- Scikit-learn

## Installation
Clone the repository:
```bash
git clone <your_repository_url>
cd <your_repository_directory>
```
Install dependencies:
```bash
python -m pip install -r requirements.txt
```

## Usage
### Training the Model
Run the following command to train the model:
```bash
python train.py --data_path <dataset_directory> --epochs 50 --batch_size 32
```
### Testing the Model
To test the model on a new audio file:
```bash
python predict.py --audio_file <audio_path>
```
### Real-Time Prediction
To perform real-time speaker identification using a microphone:
```bash
python real_time_prediction.py
```

## Code Structure
- **`SpeakerIdentification.ipynb`**: This file, contains code for speaker identification.
- **`Speaker_Identification.zip`**: This file, contains the required dataset.
- **`README.md`**: This file, providing an overview of the project.

## Key Functions
- `extract_mfcc(audio_file, max_length=500)`: Extracts MFCC features from an audio file.
- `build_lstm_model(input_shape, num_classes)`: Builds an LSTM-based classification model.
- `train_model(data_path, epochs, batch_size)`: Trains the speaker identification model.
- `predict_speaker(audio_file)`: Predicts the speaker from an input audio file.
- `real_time_identification()`: Performs live speaker identification using a microphone.

## Technologies Used
- TensorFlow/Keras
- Librosa
- OpenCV (for real-time processing)
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

