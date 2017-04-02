import numpy as np
from matplotlib import pyplot as plt
import cv2

capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(1)

while(True):
    ret, frameL = capL.read()
    ret, frameR = capR.read()
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    #print(gray.shape)
    grayL = grayL[0:720, 280:1000]
    grayR = grayR[0:720, 280:1000]
    contourL = cv2.Canny(grayL, 80, 200)
    contourR = cv2.Canny(grayR, 80, 200)
    
    
    combinedL = cv2.addWeighted(contourL, 0.8, grayL, 0.2, 0)
    combinedR = cv2.addWeighted(contourR, 0.8, grayR, 0.2, 0)
    
    stream = np.concatenate((combinedL, combinedR), axis=1)
    cv2.imshow('frame',stream)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

