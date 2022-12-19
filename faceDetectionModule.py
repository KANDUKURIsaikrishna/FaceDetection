import cv2 as cv
import mediapipe as mp
import time


class faceDetection():
    def __init__(self,min_detection_confidence=0.5,model_selection=0):
        self.min_detection_confidence=min_detection_confidence
        self.model_selection=model_selection
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(0.5)

    def findFaces(self,img,draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        #print(results)
        bboxs=[]

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                #mpDraw.draw_detection(img, detection)
           
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id,bbox,detection.score])
                if draw:
                    img=self.fancyDraw(img,bbox)
                    cv.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20),
                            cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2) 
        return img,bboxs
    
    def fancyDraw(self,img,bbox,l=30,t=5):
        x, y, w, h=bbox
        x1,y1=x+w,y+h

        cv.rectangle(img, bbox, (255, 0, 255), 1)

        cv.line(img,(x,y),(x+l,y),(255,0,255),t)
        cv.line(img,(x,y),(x,y+l),(255,0,255),t)

        cv.line(img,(x1,y),(x1-l,y),(255,0,255),t)
        cv.line(img,(x1,y),(x1,y+l),(255,0,255),t)

        cv.line(img,(x,y1),(x+l,y1),(255,0,255),t)
        cv.line(img,(x,y1),(x,y1-l),(255,0,255),t)

        cv.line(img,(x1,y1),(x1-l,y1),(255,0,255),t)
        cv.line(img,(x1,y1),(x1,y1-l),(255,0,255),t)
        return img

def main():
    cap = cv.VideoCapture(0)
    pTime = 0
    while True:
        success, img = cap.read()
        detector = faceDetection()
        img, bboxs = detector.findFaces(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv.imshow("Image", img)
        key = cv.waitKey(1) & 0XFF
        if key == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()