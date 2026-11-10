# AI-Based-Road-Analysis-and-Traffic-Simulator

## Road Vision Studio – AI-Based Road Analysis and Traffic Simulator
Road Vision Studio is an image-based computer vision application built with Python, OpenCV, and YOLO. It provides a desktop interface for analysing road images through multiple computer vision modules and displaying the original and processed images side by side.

## Features
- Modern dark-themed Tkinter desktop interface
- Upload road images in common image formats
- Select and run individual computer vision modules
- View original and processed images side by side
- Display module results and processing time
- Save processed output images locally
Computer Vision Modules
1. Lane Detection – Detects lane-line candidates using Canny edge detection and the Probabilistic Hough Line Transform.
2. Vehicle Detection and Counting – Uses a pretrained YOLOv8 model to detect and count cars, motorcycles, buses, and trucks.
3. Traffic Sign Detection – Identifies potential traffic-sign regions using colour masks and contour analysis. This module detects candidates; it does not classify the meaning of each sign.
4. Pedestrian Detection – Uses the pretrained YOLOv8 model to detect people in the image.
5. Edge Detection – Applies Gaussian smoothing and Canny edge detection to highlight image boundaries.
6. Road Segmentation – Produces an approximate road-region mask using colour thresholding and a lower-image region of interest.
7. Bird's-Eye View Transformation – Applies a perspective transformation to create a top-down-style road view.
Note: This is an image-based simulator. It analyses one still image at a time and does not track vehicles across video frames. Traditional OpenCV modules are heuristic demonstrations, not specialist trained models. Results depend on image quality, lighting, camera angle, occlusion, and road markings.

## Technology Stack
- Python
- OpenCV
- NumPy
- Pillow
- Tkinter
- Ultralytics YOLOv8 (pretrained object detection model)

## Installation and Setup
1. Install Python
Use a supported Python version compatible with the installed packages. Python 3.11 or 3.12 is a practical choice for this project.
2. Clone the repository
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
Replace the URL with your actual GitHub repository URL.
3. Create and activate a virtual environment (recommended)
Windows:
py -3.11 -m venv .venv
.venv\Scripts\activate
If Python 3.11 is not installed, use another Python version supported by your installed PyTorch and Ultralytics packages.
4. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
5. Run the application
python main.py


## YOLO Model
Vehicle Detection and Counting and Pedestrian Detection use the pretrained yolov8s.pt model.
- On the first run, Ultralytics may download the model automatically if it is not already available.
- An internet connection may be required for the first download.
- For offline use, place yolov8s.pt in the application folder and ensure the model-loading path points to that file.
- The model runs on CPU in the current configuration. Inference may take longer on systems without a supported GPU.
The model weights are not included in this repository by default because model files can be large. Follow the Ultralytics model and licensing terms when downloading or redistributing weights.

## How to Use
1. Launch the application.
2. Click Upload Image and choose a road image.
3. Select one of the seven analysis modules.
4. Click Run Simulation.
5. Compare the original image with the processed output.
6. Review the result summary and processing time.
7. Click Save Output to save the processed image.


## Acknowledgements
- OpenCV for computer vision and image-processing functions.
- Ultralytics for the YOLO object-detection framework and pretrained model.
- Python and Tkinter for the desktop application.
