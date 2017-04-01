import numpy as np
from matplotlib import pyplot as plt
import cv2

cap = cv2.VideoCapture(1)

detector=cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})
trainImg=cv2.imread("TrainImg.jpg")
trainImgGray = cv2.cvtColor(trainImg,cv2.COLOR_BGR2GRAY)
trainKP,trainDesc=detector.detectAndCompute(trainImgGray,None)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h,w = gray.shape
    if(h > 720):
        gray = cv2.resize(gray, (0,0), fx=720/h, fy=720/h)
        h,w = gray.shape
    l = int((w-h)/2)
    r = h+l
    #print(w, h, l, r)
    gray = gray[0:h, l:r]
    contour = cv2.Canny(gray, 80, 200, apertureSize = 3)
    combined = cv2.addWeighted(contour, 0.8, gray, 0.2, 0)
    combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)

    # queryKP,queryDesc=detector.detectAndCompute(gray,None)
    # bf = cv2.BFMatcher()
    # matches=bf.knnMatch(queryDesc,trainDesc, k=2)
    # goodMatch=[]
    # for m,n in matches:
    #     if(m.distance<0.75*n.distance):
    #         goodMatch.append(m)
    # MIN_MATCH_COUNT=30
    # if(len(goodMatch)>=MIN_MATCH_COUNT):
    #     tp=[]
    #     qp=[]
    #     for m in goodMatch:
    #         tp.append(trainKP[m.trainIdx].pt)
    #         qp.append(queryKP[m.queryIdx].pt)
    #     tp,qp=np.float32((tp,qp))
    #
    #     H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
    #     h,w,c=trainImg.shape
    #     trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
    #     queryBorder=cv2.perspectiveTransform(trainBorder,H)
    #     cv2.polylines(combined,[np.int32(queryBorder)],True,(0,255,0),5)


    stream = np.concatenate((combined, combined), axis=1)
    cv2.imshow('frame',stream)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
