import numpy as np
from matplotlib import pyplot as plt
import cv2

cap = cv2.VideoCapture(1)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contour = cv2.Canny(gray, 100, 200)
    cv2.imshow('frame',contour)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
