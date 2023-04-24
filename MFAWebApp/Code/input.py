import cv2
from faceDetect import FaceDetect
cam = cv2.VideoCapture(0)
cv2.namedWindow("User Input")
detect = FaceDetect(0.50,"cpu")
def clickInput():
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("User Input", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            #img_name = "../Data/user.png"
            img_name = "Data/user.png"
            numFaces, conf ,boxes = detect.detect(frame)
            if numFaces!=1:
                print("Please present a single face to the camera")
                continue
            (x,y,w,h) = boxes[0]
            dim = frame.shape
            x, y, w, h = int(x), int(y), int(w), int(h)
            if y-50<0 or x-50<0 or h+50>dim[1] or  w+50 > dim[0]:
                print("Please center yourself in the frame to the camera")
                continue
        
            face_frame = cv2.resize(frame[y-50:h+50,x-50:w+50],(1000,1000))
            cv2.imshow("User Input", frame[y-50:h+50,x-50:w+50])
            cv2.imwrite(img_name, face_frame)
            print("{} written!".format(img_name))
            print("press ENTER to exit or SPACE to retake")
            k2 = cv2.waitKey(0)
            if k2%256 == 13:
                break
            elif k%256 == 32:
                continue

    cam.release()

    cv2.destroyAllWindows()
