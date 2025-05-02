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
## Run Project
To run the project, First execute the following command in your terminal to run server:
```
python server.py  
 ```
Then open new terminal and run following command to run GUI
```
python denoise_app.py  
 ```


