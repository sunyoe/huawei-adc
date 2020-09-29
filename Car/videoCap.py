#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
import numpy as np

cap = cv2.VideoCapture('/dev/video10')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame, 2)
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(2) == ord('q'):
            break
        else:
            break
cap.release()
out.release()
cv2.destroyAllWindows()