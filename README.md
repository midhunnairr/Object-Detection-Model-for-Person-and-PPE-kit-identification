
YOLOv8 Person and PPE Detection

This repository provides a framework for real-time person and Personal Protective Equipment (PPE) detection using the state-of-the-art YOLOv8 object detection model. It leverages the pre-trained YOLOv8 model and facilitates customization for specific safety scenarios.

Features:

Multi-Object Detection: Simultaneously detects both people and various PPE items within an image or video frame.
Customizable: Easily adapt the model to detect additional PPE types or adjust detection parameters for your specific use case.
Modular Design: The code is well-structured for clarity and maintainability.
Getting Started

Prerequisites:

Python (version 3.6 or later)
PyTorch (version 1.8 or later)
ultralytics (install using pip install ultralytics)
Installation:

Pre-trained Model:

The repository includes a pre-trained YOLOv8 model (person_ppe.pt) that detects people and commonly used PPE items (e.g., helmets, safety vests, goggles). To use it:

Run real-time detection on webcam:

The model can be customized for different scenarios:

Modify classes.py: Update the classNames list to include additional PPE types you want to detect. Refer to the YOLOv8 documentation for class name conventions.
Fine-tuning the model: For more specific use cases, you can fine-tune the pre-trained model on your own dataset. This involves collecting and labeling images containing people and the desired PPE items. Refer to the YOLOv8 documentation for advanced usage and customization options.
Results

The model outputs bounding boxes around detected objects, along with their class labels (e.g., "person," "helmet," "vest"). You can visualize the results in real-time or on individual images.

The results are present in the result folder inside each model folder and the additional report on how the approach is done and how the model is made is also inserted in the repository.

