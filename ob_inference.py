'''
random pick data 1000
object detection to crop peoples
extract embedding 
t-SNE
k-means
--------------------------------
util_ob_inference.py
    load_model
    detect
    crop_patch
    save_patch

util_extract_embedding
    load_model
    extract_embedding
    save_embedding
util_dimension_reduction.py
    t-SNE
util_cluster.py
    k-means
main.py
----------------------------------
'''
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import matplotlib.pyplot
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE

def read_img(path):
    raw_img = cv2.imread(path)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    return raw_img


if __name__ == '__main__':
    tf.keras.backend.clear_session()
    model = tf.saved_model.load('/Users/i_chiao/myJob/my_github/unsupervised_filter_data/model_zoo/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model/')
    base_model = keras.applications.MobileNetV2(weights='imagenet')
    base_model.summary()
    embedding_model = keras.applications.MobileNetV2(weights='imagenet', include_top=False, pooling="avg")
    embedding_model.summary()
    for dirPath, dirNames, fileNames in os.walk('/Users/i_chiao/myJob/my_github/unsupervised_filter_data/dataset/yoga/dataset2/final_test/yoga_set1'):
        for f in fileNames:
            fn, ext = os.path.splitext(f)
            if ext not in ['.jpg', '.jpeg', '.png', '.JPG']:
                continue
            path = os.path.join(dirPath, f)
            img = read_img(path)

            # detect
            input_tensor = np.expand_dims(img, 0)
            detections = model(input_tensor)
            boxes = detections['detection_boxes'][0].numpy()
            scores = detections['detection_scores'][0].numpy()
            classes = detections['detection_classes'][0].numpy().astype(np.int32)
            for boxe, score, cla in zip(boxes, scores, classes):
                if score > 0.65 and cla == 1:
                    print(boxe, score, cla)

                    img = cv2.resize(img, (224, 224))
                    x = np.expand_dims(img, axis=0) # (224, 224, 3) -> (1, 224, 224, 3)

                    preds = base_model.predict(x)
                    pred_label = np.argmax(preds, axis=1)[0]
                    print(pred_label, preds[0][pred_label])

                    preds = embedding_model.predict(x)
                    print(preds.shape)
                    print(preds[0])


