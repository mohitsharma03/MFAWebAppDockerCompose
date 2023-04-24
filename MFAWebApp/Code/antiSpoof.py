
import os
import cv2
import numpy as np
import warnings
import math
from AntiSpoof.src.anti_spoof_predict import AntiSpoofPredict
from AntiSpoof.src.generate_patches import CropImage
from AntiSpoof.src.utility import parse_model_name
model_dir = "Code/AntiSpoof/resources/anti_spoof_models"
warnings.filterwarnings('ignore')


class Antispoof: 
    def __init__(self):
        self.model_test = AntiSpoofPredict(0)
        self.image_cropper = CropImage()
    def predict_spoof(self,frame):
        prediction = np.zeros((1, 3))
        image_bbox = self.model_test.get_bbox(frame)
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": frame,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = self.image_cropper.crop(**param)
            prediction += self.model_test.predict(img, os.path.join(model_dir, model_name))
            label = np.argmax(prediction)
            value = prediction[0][label]
            return (label,value)


# img1 = cv2.imread("/home/siddharth/Downloads/lbj1.jpeg")
# sp = Antispoof()
# print(sp.predict_spoof(img1))
