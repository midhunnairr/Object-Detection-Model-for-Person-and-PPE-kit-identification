import os
import xml.etree.ElementTree as ET

def convert_voc_to_yolo(voc_dir, output_dir, classes):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for xml_file in os.listdir(voc_dir):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(voc_dir, xml_file))
            root = tree.getroot()
            
            image_width = int(root.find('size/width').text)
            image_height = int(root.find('size/height').text)
            
            with open(os.path.join(output_dir, os.path.splitext(xml_file)[0] + '.txt'), 'w') as yolo_file:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name not in classes:
                        continue
                    class_id = classes.index(class_name)
                    
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    
                    # Calculate YOLO format values
                    center_x = (xmin + xmax) / 2 / image_width
                    center_y = (ymin + ymax) / 2 / image_height
                    width = (xmax - xmin) / image_width
                    height = (ymax - ymin) / image_height
                    
                    # Write to YOLO format file
                    yolo_file.write(f"{class_id} {center_x} {center_y} {width} {height}\n")


voc_dir = 'C:/Users/midiy/OneDrive/Desktop/labels'
output_dir = 'C:/Users/midiy/OneDrive/Desktop/datafornemodel/labels_YOLO'
classes = ['hard-hat','gloves','mask','glasses','boots','vest','ppe-suit','ear-protector','safety-harness']  # Replace with your actual class names

convert_voc_to_yolo(voc_dir, output_dir, classes)
