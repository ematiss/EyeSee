import numpy as np
from matplotlib import pyplot as plt
import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print(gray.shape)
    gray = gray[0:720, 280:1000]
    contour = cv2.Canny(gray, 80, 200)


    combined = cv2.addWeighted(contour, 0.8, gray, 0.2, 0)

    stream = np.concatenate((combined, combined), axis=1)
    cv2.imshow('frame',stream)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
