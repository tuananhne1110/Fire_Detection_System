import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def process_xml_annotations(data_dir):
    all_annotations = []

    for xml_file in os.listdir(os.path.join(data_dir, 'Annotations')):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(data_dir, 'Annotations', xml_file)
            image_filename = xml_file.replace('.xml', '.jpg')
            image_path = os.path.join(data_dir, 'images', image_filename)

            tree = ET.parse(xml_path)
            root = tree.getroot()
            annotations = []

            for neighbor in root.iter('object'):
                object_name = neighbor.find('name').text
                xtl = float(neighbor.find('bndbox/xmin').text)
                ytl = float(neighbor.find('bndbox/ymin').text)
                xbr = float(neighbor.find('bndbox/xmax').text)
                ybr = float(neighbor.find('bndbox/ymax').text)
                annotations.append({'object_name': object_name, 'bbox': [xtl, ytl, xbr, ybr]})

            all_annotations.append({'image_path': image_path, 'annotations': annotations})

    return all_annotations

def visualize_annotations(all_annotations, max_images_to_show=10):
    image_count = 0

    for image_info in all_annotations:
        image_path = image_info['image_path']
        annotations = image_info['annotations']

        sample_image = Image.open(image_path)
        sample_image_annotated = sample_image.copy()
        img_box = ImageDraw.Draw(sample_image_annotated)

        for annotation in annotations:
            object_name = annotation['object_name']
            bbox = annotation['bbox']
            img_box.rectangle(bbox, outline='green')
            img_box.text((bbox[0], bbox[1]), object_name, fill='green')

        plt.imshow(sample_image_annotated)
        plt.show()

        image_count += 1
        if image_count >= max_images_to_show:
            break

# # Gọi hàm process_xml_annotations để lấy danh sách thông tin
# data_dir = 'C:\\Users\\sktkc\\Downloads\\fire&smoke'
# all_annotations = process_xml_annotations(data_dir)

# # Sử dụng hàm visualize_annotations với số lượng hình ảnh tối đa cần hiển thị (ví dụ: 10)
# visualize_annotations(all_annotations, max_images_to_show=10)
