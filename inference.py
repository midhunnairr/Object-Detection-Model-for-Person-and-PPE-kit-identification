import argparse
from ultralytics import YOLO
import cv2
import os

def perform_inference(input_directory, output_directory, person_model_path, ppe_model_path):
    # Load YOLO models
    person_model = YOLO(person_model_path)
    ppe_model = YOLO(ppe_model_path)
    
    # Iterate over all images in the input directory
    for image_name in os.listdir(input_directory):
        image_path = os.path.join(input_directory, image_name)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error reading image {image_name}. Skipping.")
            continue
        
        # Perform inference with the person model
        person_results = person_model(image)[0]
        # Perform inference with the PPE model
        ppe_results = ppe_model(image)[0]
        
        # Copy the original image for each model's output
        person_out = image.copy()
        ppe_out = image.copy()
        
        # Draw bounding boxes and labels for person detections
        for i in range(len(person_results.boxes.xyxy)):
            x1, y1, x2, y2 = map(int, person_results.boxes.xyxy[i])
            class_id = int(person_results.boxes.cls[i])
            label = person_results.names.get(class_id, "Unknown")
            confidence = person_results.boxes.conf[i]
            cv2.rectangle(person_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(person_out, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw bounding boxes and labels for PPE detections
        for i in range(len(ppe_results.boxes.xyxy)):
            x1, y1, x2, y2 = map(int, ppe_results.boxes.xyxy[i])
            class_id = int(ppe_results.boxes.cls[i])
            label = ppe_results.names.get(class_id, "Unknown")
            confidence = ppe_results.boxes.conf[i]
            cv2.rectangle(ppe_out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(ppe_out, f'{label} {confidence:.2f}', (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Save the output images
        person_output_path = os.path.join(output_directory, 'person_' + image_name)
        ppe_output_path = os.path.join(output_directory, 'ppe_' + image_name)
        cv2.imwrite(person_output_path, person_out)
        cv2.imwrite(ppe_output_path, ppe_out)

    print("Inference completed and results saved.")

#the code is made in such a way that the input to be recieved is from the CLI and this is why argparser is initiated
#python inference.py --input_dir "C:/Users/midiy/OneDrive/Desktop/input" --output_dir "C:/Users/midiy/OneDrive/Desktop/Final_Results" --person_model "C:/Users/midiy/OneDrive/Desktop/new/runs/detect/20/weights/best.pt" --ppe_model "C:/Users/midiy/OneDrive/Desktop/newmodel/runs/detect/train8/weights/best.pt"
#this is the sample usage that i've used

def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference for person and PPE detection.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing input images.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save output images.")
    parser.add_argument('--person_model', type=str, required=True, help="Path to the person detection model.")
    parser.add_argument('--ppe_model', type=str, required=True, help="Path to the PPE detection model.")

    args = parser.parse_args()

    perform_inference(args.input_dir, args.output_dir, args.person_model, args.ppe_model)

if __name__ == "__main__":
    main()
