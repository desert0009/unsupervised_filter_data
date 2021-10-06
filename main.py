import argparse
import cv2
import os
from util import ob_detect

SAVE_PEOPLE = 'people_patches'

def crop_save_people(ob_model_path, data_path, save_path):
    # load object detect model
    ob_detector = ob_detect.ObjectDetect(ob_model_path)

    check_dir(save_path)

    # load img
    for dirPath, dirNames, fileNames in os.walk(data_path):
        for f in fileNames:
            fn, ext = os.path.splitext(f)
            if ext not in ['.jpg', '.jpeg', '.png', '.JPG']:
                continue
            img_path = os.path.join(dirPath, f)
            img = read_img(img_path)
            img_h, img_w, _ = img.shape

            # object detection
            boxes, scores, classes = ob_detector.detect(img)
            people_boxes = ob_detector.get_people_boxes(boxes, scores, classes, conf_thr=0.65)

            # crop and save people patches
            for i, box in enumerate(people_boxes):
                y1, x1, y2, x2 = box
                y1, x1, y2, x2 = (int)(y1 * img_h), (int)(x1 * img_w), (int)(y2 * img_h), (int)(x2 * img_w)
                crop_img = img[y1:y2, x1:x2]
                save_file_name = '{}_{}_{}{}'.format(dirPath.replace('/', '_'), fn, i, ext)
                cv2.imwrite(os.path.join(save_path, save_file_name), \
                            cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

def read_img(path):
    raw_img = cv2.imread(path)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    return raw_img

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter Data')
    parser.add_argument('--ob_model_path', type=str, default='./model_zoo/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model/')
    parser.add_argument('--data_path', type=str, default='./dataset/yoga/dataset2/final_test/yoga_set1')
    parser.add_argument('--save_path', type=str, default='./result/')
    args = parser.parse_args()

    # crop and save people patch
    crop_save_people(args.ob_model_path, args.data_path, os.path.join(args.save_path, SAVE_PEOPLE))
    
            

    