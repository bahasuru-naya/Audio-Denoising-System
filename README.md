# Audio-Denoising-System
Advanced audio denoising system using a Convolutional Neural Network (CNN) with a U-Net architecture and a ResNet101 backbone. This system effectively removes background noises from audio recordings, preserving the integrity of the original signal. It features a user-friendly GUI built with customtkinter for recording, uploading, denoising, and playing audio, integrated with a server-based backend for processing

<img src="https://github.com/user-attachments/assets/1f5f91a7-0141-4131-b795-6ed09557293d" width="600"/>

## Key Features
- Audio Recording and Upload: Capture audio via microphone or upload files.
- Denoising: Process audio using a pretrained U-Net model to remove noise.
- Playback: Compare original and denoised audio with seamless playback controls.
- Visualization: Display waveforms and spectrograms for both original and denoised audio.
- Save Functionality: Export denoised audio.
- Client-Server Architecture: Scalable backend for denoising with a TCP socket-based server.
- Multi-Threaded Processing: Efficient handling of recording, playback, and denoising tasks.

## Model Training

The model was trained on the LibriSpeech Noise Dataset with the following configuration:

- Architecture: U-Net with ResNet101 backbone, pretrained on ImageNet.
- Input: Log-magnitude spectrograms (544x320x1).
- Preprocessing: Audio chunking, STFT, normalization, and padding
- Training: 100 epochs, batch size 6, Adam optimizer (lr=0.0001), MSE loss.
  
The training achieved a validation loss of 0.0083, indicating strong denoising performance

## System Requirements
### Hardware
- Microphone (built-in or external) (OPTIONAL)
- Minimum 4GB RAM (8GB recommended)

### Software
- Python 3.9.10
- Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019

## Installation
### 1. Create new python project and virtual environment in your IDE (use Python 3.9.10)
### 2. Download or clone repository into your project
```
git clone https://github.com/bahasuru-naya/Audio-Denoising-System.git
```
### 3. Go to the project directory in terminal then run following command to install dependencies in project
```
cd Audio-Denoising-System
pip install -r requirements.txt
 ```
### 4. Download Pretrained Model from following link and add it into `Model` folder
[https://drive.google.com/drive/folders/1MwlU6GL3yRqDf0leQ2_6Y3TbC6avY8M2?usp=sharing](https://drive.google.com/drive/folders/1MwlU6GL3yRqDf0leQ2_6Y3TbC6avY8M2?usp=sharing)

## Run Project
To run the project, First execute the following command in your terminal to run server:
```
python server.py  
 ```
Then open new terminal and run following command to run GUI
```
python Denoise_app.py  
 ```


