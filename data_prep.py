import numpy as np
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
from shutil import copyfile
import os
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

classes = ['helmet', 'head', 'person']

def convert_annot(size, box):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    dw = np.float32(1. / int(size[0]))
    dh = np.float32(1. / int(size[1]))

    w = x2 - x1
    h = y2 - y1
    x = x1 + (w / 2)
    y = y1 + (h / 2)

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]

def save_txt_file(img_jpg_file_name, size, img_box):
    save_file_name = 'Dataset/labels/' + img_jpg_file_name + '.txt'

    with open(save_file_name, 'a+') as file_path:
        for box in img_box:
            cls_num = classes.index(box[0])
            new_box = convert_annot(size, box[1:])
            file_path.write(f"{cls_num} {new_box[0]} {new_box[1]} {new_box[2]} {new_box[3]}\n")
        file_path.flush()
        file_path.close()

def get_xml_data(file_path, img_xml_file):
    img_path = file_path + '/' + img_xml_file + '.xml'

    tree = ET.parse(img_path)
    root = tree.getroot()

    img_name = root.find("filename").text
    img_size = root.find("size")
    img_w = int(img_size.find("width").text)
    img_h = int(img_size.find("height").text)
    img_c = int(img_size.find("depth").text)

    img_box = []
    for box in root.findall("object"):
        cls_name = box.find("name").text
        x1 = int(box.find("bndbox").find("xmin").text)
        y1 = int(box.find("bndbox").find("ymin").text)
        x2 = int(box.find("bndbox").find("xmax").text)
        y2 = int(box.find("bndbox").find("ymax").text)

        img_box.append([cls_name, x1, y1, x2, y2])

    img_jpg_file_name = img_xml_file + '.' + img_name.split('.')[-1]
    save_txt_file(img_xml_file, [img_w, img_h], img_box)

if __name__ == "__main__":
    os.makedirs('Dataset/labels', exist_ok=True)
    os.makedirs('Dataset/images', exist_ok=True)
    os.makedirs('Dataset/images/train', exist_ok=True)
    os.makedirs('Dataset/images/val', exist_ok=True)
    os.makedirs('Dataset/images/test', exist_ok=True)

    annotation_path = 'archive/annotations'
    images_path = 'archive/images'

    files = os.listdir(annotation_path)
    files = sorted([f for f in files if f.endswith('.xml')])
    random.seed(42)
    files = random.sample(files, 1000)

    for file in tqdm(files, total=len(files)):
        file_xml = file.split(".")
        get_xml_data(annotation_path, file_xml[0])

    # Copy images
    image_list = [f.replace('.xml', '.png') for f in files]
    for file in image_list:
        copyfile(f'{images_path}/{file}', f'Dataset/images/{file}')

    # Split
    train_list, test_list = train_test_split(image_list, test_size=0.2, random_state=42)
    val_list, test_list = train_test_split(test_list, test_size=0.5, random_state=42)

    # Split
    image_list = [f.replace('.xml', '.png') for f in files]
    train_list, test_list = train_test_split(image_list, test_size=0.2, random_state=42)
    val_list, test_list = train_test_split(test_list, test_size=0.5, random_state=42)

    for mode, flist in [('train', train_list), ('val', val_list), ('test', test_list)]:
        for f in flist:
            if os.path.exists(f'Dataset/images/{f}'):
                copyfile(f'Dataset/images/{f}', f'Dataset/images/{mode}/{f}')
            if os.path.exists(f'Dataset/labels/{f.replace(".png", ".txt")}'):
                copyfile(f'Dataset/labels/{f.replace(".png", ".txt")}', f'Dataset/images/{mode}/{f.replace(".png", ".txt")}')
