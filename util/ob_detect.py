import tensorflow as tf
import numpy as np

class ObjectDetect:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        tf.keras.backend.clear_session()
        return tf.saved_model.load(model_path)

    def detect(self, img):
        input_tensor = np.expand_dims(img, 0)
        detections = self.model(input_tensor)
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int32)
        return boxes, scores, classes

    def get_people_boxes(self, boxes, scores, classes, conf_thr=0.65):
        res = []
        for box, score, cla in zip(boxes, scores, classes):
            if self.__is_people_cla(cla) and score >= conf_thr:
                res.append(box)
        return res
    
    def __is_people_cla(self, cla):
        return cla == 1