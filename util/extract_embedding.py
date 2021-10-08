import tensorflow as tf
from tensorflow import keras
from keras.models import Model
import numpy as np
import cv2

class FeatureExtractor:
    def __init__(self):
        self.base_model = keras.applications.MobileNetV2(weights='imagenet')
        #self.summary_model(self.base_model)
        #self.embedding_model = keras.applications.MobileNetV2(weights='imagenet', include_top=False, pooling="avg")
        self.embedding_model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer("global_average_pooling2d").output)
        #self.summary_model(self.embedding_model)

    def data_preprocess(self, img):
        img = cv2.resize(img, (224, 224))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(img, data_format=None)
        x = np.expand_dims(x, axis=0) # (224, 224, 3) -> (1, 224, 224, 3)
        return x

    def detect_cla(self, x):
        preds = self.base_model.predict(x)
        pred_label = np.argmax(preds, axis=1)[0]
        return pred_label, preds[0][pred_label]

    def extract_embedding(self, x):
        return self.embedding_model.predict(x)
    
    def reshape(self, x):
        return np.reshape(x, (1280))

    def summary_model(self, model):
        model.summary()