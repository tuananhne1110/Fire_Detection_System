import os
import xml.etree.ElementTree as ET
import pandas as pd
from PIL import Image

def process_xml_annotations(data_dir):
    data_list = []

    image_filenames = sorted([f for f in os.listdir(os.path.join(data_dir, 'images')) if f.endswith('.jpg')])

    index = 1  # Biến đếm cho cột index

    for i, xml_file in enumerate(sorted(os.listdir(os.path.join(data_dir, 'Annotations')))):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(data_dir, 'Annotations', xml_file)

            tree = ET.parse(xml_path)
            root = tree.getroot()

            image_filename = image_filenames[i]
            image_path = os.path.join(data_dir, 'images', image_filename)

            for neighbor in root.iter('object'):
                name = neighbor.find('name').text
                xtl = float(neighbor.find('bndbox/xmin').text)
                ytl = float(neighbor.find('bndbox/ymin').text)
                xbr = float(neighbor.find('bndbox/xmax').text)
                ybr = float(neighbor.find('bndbox/ymax').text)
                bbox_info = (name, (xtl, ytl, xbr, ybr))
                img = Image.open(image_path)
                img_width, img_height = img.size

                # Tính toán chiều rộng và chiều cao của bounding box
                bbox_width = xbr - xtl
                bbox_height = ybr - ytl

                data_list.append({
                    'Index': index,  # Cột Index để đếm số lượng dòng
                    'Name': image_filename,  # Tên ảnh
                    'Object_Name': name,
                    'BoundingBox_Info': bbox_info,
                    'Image_Width': img_width,
                    'Image_Height': img_height,
                    'BoundingBox_Width': bbox_width,  # Chiều rộng của bounding box
                    'BoundingBox_Height': bbox_height  # Chiều cao của bounding box
                })

                index += 1  # Tăng giá trị biến đếm

    return data_list

def export_csv(data_list, output_path):
    df = pd.DataFrame(data_list)
    df.to_csv(output_path, index=False)

# Example usage
if __name__ == "__main__":
    data_dir = 'C:\\Users\\sktkc\\Downloads\\fire&smoke'
    output_csv_path = os.path.join(data_dir, 'f.csv')
    data_list = process_xml_annotations(data_dir)
    export_csv(data_list, output_csv_path)
