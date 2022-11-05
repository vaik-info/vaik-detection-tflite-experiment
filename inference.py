import argparse
import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from vaik_detection_tflite_inference.tflite_model import TfliteModel
from vaik_pascal_voc_rw_ex import pascal_voc_rw_ex


def main(input_saved_model_path, input_classes_path, input_image_dir_path, output_xml_dir_path, score_th, nms_th):
    os.makedirs(output_xml_dir_path, exist_ok=True)
    classes = []
    with open(input_classes_path) as f:
        for line in f:
            classes.append(line.strip())
    model = TfliteModel(input_saved_model_path, tuple(classes))

    types = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    image_path_list = []
    for files in types:
        image_path_list.extend(glob.glob(os.path.join(input_image_dir_path, files), recursive=True))

    import time
    total_time = 0
    for image_path in tqdm(image_path_list):
        image = np.asarray(Image.open(image_path).convert('RGB'))
        start = time.time()
        objects_dict_list, raw_pred = model.inference(image, score_th=score_th, nms_th=nms_th)
        end = time.time() - start
        total_time += end
        output_xml_path = os.path.join(output_xml_dir_path, os.path.splitext(os.path.basename(image_path))[0] + '.xml')
        pascal_voc_rw_ex.write_pascal_voc_xml_dict(output_xml_path, image_path,
                                                   object_extend_dict_list=objects_dict_list)
    print(f'{len(image_path_list)/total_time}[images/sec]')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--input_saved_model_path', type=str, default='~/export_model/efficientdet-lite0_edgetpu.tflite')
    parser.add_argument('--input_classes_path', type=str, default='~/.vaik-mnist-detection-dataset/classes.txt')
    parser.add_argument('--input_image_dir_path', type=str, default='~/.vaik-mnist-detection-dataset/valid')
    parser.add_argument('--output_xml_dir_path', type=str, default='~/.vaik-mnist-detection-dataset/valid_inference')
    parser.add_argument('--score_th', type=float, default=0.2)
    parser.add_argument('--nms_th', type=float, default=None)
    args = parser.parse_args()

    args.input_saved_model_path = os.path.expanduser(args.input_saved_model_path)
    args.input_classes_path = os.path.expanduser(args.input_classes_path)
    args.input_image_dir_path = os.path.expanduser(args.input_image_dir_path)
    args.output_xml_dir_path = os.path.expanduser(args.output_xml_dir_path)

    main(**args.__dict__)
