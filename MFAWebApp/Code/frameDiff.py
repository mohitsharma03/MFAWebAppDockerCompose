from skimage.metrics import structural_similarity as compare_ssim
import cv2

class FrameDiff:
    def __init__(self,threshold):
        self.threshold = threshold
    def ssim(self,curr_frame,prev_frame):
        current_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        previous_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        (score,frame_diff) = compare_ssim(previous_frame_gray,current_frame_gray,full = True)
        if(score<self.threshold):
            return (True, score)
        return (False, score)

# fd = FrameDiff(0.9)
# img1 = cv2.resize(cv2.imread("/home/siddharth/Downloads/Sunflower_from_Silesia2.jpg"),(300,300))
# img2 = cv2.resize(cv2.imread("/home/siddharth/Downloads/images.jpeg"),(300,300))
# print(fd.ssim(img1,img1))




