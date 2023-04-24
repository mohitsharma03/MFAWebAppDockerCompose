from deepface import DeepFace
import cv2
import re
import numpy as np
class FaceRecog:
    def __init__(self,model="VGG-Face",metric="euclidean_l2",threshold = 0.86): #"VGG-Face", "Facenet", "OpenFace", "DeepFace",  "Dlib"
        self.model = model
        self.metric = metric
        self.threshold = threshold
    def verify(self,frame,src):
        img = self.shadow_remove(frame, 2)
        # src = cv2.addWeighted(src,2,src,0,10)
        # cv2.imwrite("norm.png",frame)
        # cv2.imwrite("brigth.png",img)
        res = DeepFace.verify(img,src,model_name=self.model,distance_metric = self.metric, enforce_detection = False, detector_backend = 'skip')
        print(res['distance'])
        if res["distance"]<=self.threshold:
            return (True,res["distance"])
        return (False,res["distance"])
    def shadow_remove(self,src,gamma):
        invGamma = 1 / gamma
        table = [((i / 255) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)
        return cv2.LUT(src, table)
    

# fr = FaceRecog()
# img1 = cv2.imread("/home/siddharth/Downloads/lbj1.jpeg")
# img2 = cv2.imread("/home/siddharth/Downloads/lbj2.jpeg")
# print(fr.verify(img1,img2))