import os
import cv2
import numpy as np
import time
from frameDiff import FrameDiff
from faceDetect import FaceDetect
from faceRecog import FaceRecog
from antiSpoof import Antispoof

class Pipeline:
    def __init__(self):
        self.antiSpoof = Antispoof()
        self.diff = FrameDiff(0.92)
        self.detect = FaceDetect(0.50,"cpu")
        self.rec = FaceRecog(threshold = 0.25, metric = "cosine")
        #self.user = cv2.resize(cv2.imread("../Data/user.png"),(1000,1000))

    def pipeline2(self,prev,curr, flag, user):
        prev_frame = cv2.resize(prev,(600, 400))
        curr_frame = cv2.resize(curr,(600, 400))
        #user = cv2.resize(cv2.imread(user),(1000,1000))
        user = cv2.imread(user)
        retCode =  ""


        try:
            #frame_input_time = time.time()
            print("flag ",flag)
            diff_val, diff_score = self.diff.ssim(prev_frame, curr_frame)
            if diff_val or flag == '1':
                if diff_val:
                    cv2.putText(curr_frame, 'Frame Change', (0,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    #print("diff frame", diff_score)
                if flag == '1': 
                     cv2.putText(curr_frame, 'Error in Previous Frame. Present Correct Face or be Logged Out', (10,370),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                numFaces, conf ,boxes = self.detect.detect(curr_frame)

                if numFaces==1:
                    #print("1 face detected")
                    (x,y,w,h) = boxes[0]
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    curr= curr_frame.copy()
                    text = f"{conf[0]*100:.2f}%"
                    cv2.putText(curr_frame,text,(x, y - 20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),1)
                    cv2.rectangle(curr_frame, (x, y), (w, h), (0, 255, 0), 1)
                    ymin = max(y-25,0)
                    hmax = min(h+30,400)
                    xmin = max(x-15,0)
                    wmax = min(w+15,600)
                    isSame, distance = self.rec.verify(curr[ymin:hmax,xmin:wmax],user)
                    if(isSame):
                        cv2.putText(curr_frame, 'Face Verified', (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        #print("face is verified", distance)
                        #####antispoof
                        spoof_result = self.antiSpoof.predict_spoof(curr_frame)
                        if spoof_result[1]>=0.9999:
                            #print("face is Fake", spoof_result[1])
                            cv2.putText(curr_frame, 'Spoofed Face Detected', (0,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            #file.write(outputStr.format("Spoofed Face Detected",frameCount,time.time()-start_time))
                            retCode = "SPOOF"
                        else:
                            retCode = "VERIFIED"
                    else:
                        cv2.putText(curr_frame, 'Face Not Verified', (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        #print("face is not verified", distance)
                        #file.write(outputStr.format("Face Not Verified",frameCount,time.time()-start_time))
                        retCode = "NOTVERIFIED"

                    
                elif numFaces==0:
                    #print("0 face detected")
                    cv2.putText(curr_frame, 'No Faces', (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    #file.write(outputStr.format("No Faces Detected",frameCount,time.time()-start_time))
                    retCode = "NOFACES"
                else:
                    #print("multiple faces detected")
                    cv2.putText(curr_frame, 'Multiple Faces', (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    #file.write(outputStr.format("Multiple Faces Detected",frameCount,time.time()-start_time))
                    retCode = "MULTIPLEFACES"

            else:
                #print("No difference")
                cv2.putText(curr_frame, 'No Frame Change', (0,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            ret, buffer = cv2.imencode('.jpg', curr_frame)
            frame = buffer.tobytes()
            return [frame, retCode]
            
        except cv2.error:
            print(cv2.error)
        # if cv2.waitKey(25) & 0xFF == ord("q"):
        #     break
        # file.write("Summary\n")
        # file.write("1. Missing Face: "+str(noFaceCount)+"\n")
        # file.write("2. Multiple Faces in Frame: "+str(multipleFaceCount)+"\n")
        # file.write("3. Unverified User: "+ str(unverifiedFaceCount)+"\n")
        # file.write("4. Spoofed Face Count: "+ str(spoofedFaceCount)+"\n")
        # file.close()
    


        
   #def runPipeline(self,video=0,source = "../Data/user.png"):
    def runPipeline(self,video=0,source = "Data/user.png"):
	
        ###############################
        cap = cv2.VideoCapture(video)
        ###############################

        user = cv2.resize(cv2.imread(source),(1000,1000))

        ##################################
        ret, curr_frame = cap.read()
        prev_frame = cv2.resize(curr_frame,(600, 400))
        print("read first frame")
        ret, curr_frame = cap.read()
        ###################################

        curr_frame = cv2.resize(curr_frame,(600, 400))
        frameCount = 2
        frameSkip = 10
        noFaceCount = 0
        multipleFaceCount = 0
        unverifiedFaceCount = 0
        spoofedFaceCount=0
        outputStr = "{}, {}, {}\n"
        file = open("pipeline_output.txt","w")
        start_time =time.time()
        while True:
            try:
                frame_input_time = time.time()
                diff_val, diff_score = self.diff.ssim(prev_frame, curr_frame)
                if diff_val:
                    cv2.putText(curr_frame, 'Frame Change', (0,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    print("diff frame", diff_score)
                    numFaces, conf ,boxes = self.detect.detect(curr_frame)

                    if numFaces==1:
                        print("1 face detected")
                        (x,y,w,h) = boxes[0]
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        text = f"{conf[0]*100:.2f}%"
                        cv2.putText(curr_frame,text,(x, y - 20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),1)
                        cv2.rectangle(curr_frame, (x, y), (w, h), (0, 255, 0), 1)
                        isSame, distance = self.rec.verify(curr_frame,user)
                        if(isSame):
                            cv2.putText(curr_frame, 'Face Verified', (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            print("face is verified", distance)
                            #antispoof
                            spoof_result = self.antiSpoof.predict_spoof(curr_frame)
                            if spoof_result[1]>=0.999:
                                print("face is Fake", spoof_result[1])
                                cv2.putText(curr_frame, 'Spoofed Face Detected', (0,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                                file.write(outputStr.format("Spoofed Face Detected",frameCount,time.time()-start_time))
                                spoofedFaceCount+=1

                        else:
                            cv2.putText(curr_frame, 'Face Not Verified', (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            print("face is not verified", distance)
                            file.write(outputStr.format("Face Not Verified",frameCount,time.time()-start_time))
                            unverifiedFaceCount+=1

                        
                    elif numFaces==0:
                        print("0 face detected")
                        cv2.putText(curr_frame, 'No Faces', (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        file.write(outputStr.format("No Faces Detected",frameCount,time.time()-start_time))
                        noFaceCount+=1
                    else:
                        print("multiple faces detected")
                        cv2.putText(curr_frame, 'Multiple Faces', (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        file.write(outputStr.format("Multiple Faces Detected",frameCount,time.time()-start_time))
                        multipleFaceCount+=1
                ret, buffer = cv2.imencode('.jpg', curr_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame+ b'\r\n')
                frameCount+=frameSkip
                cap.set(cv2.CAP_PROP_POS_FRAMES,frameCount)
                ###########################################
                prev_frame = curr_frame.copy()
                ret, curr_frame = cap.read()
                ###########################################
                curr_frame = cv2.resize(curr_frame,(600,400))
                
            except cv2.error:
                break
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        file.write("Summary\n")
        file.write("1. Missing Face: "+str(noFaceCount)+"\n")
        file.write("2. Multiple Faces in Frame: "+str(multipleFaceCount)+"\n")
        file.write("3. Unverified User: "+ str(unverifiedFaceCount)+"\n")
        file.write("4. Spoofed Face Count: "+ str(spoofedFaceCount)+"\n")
        file.close()
        cap.release()
        cv2.destroyAllWindows()


