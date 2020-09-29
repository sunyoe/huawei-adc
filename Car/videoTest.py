#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
import numpy as np

class camera:
    def __init__(self):
        cv2.destroyAllWindows()
        self.cap = cv2.VideoCapture('Video_2.avi')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # src_points = np.array([[3,570], [387,460], [906,452], [1041,485]], dtype="float32")
        # dst_points = np.array([[266., 686.], [266., 19.], [931., 20.], [931., 701.]], dtype="float32")
        src_points = np.array([[3,570], [387,460], [1106,472], [1241,520]], dtype="float32")
        dst_points = np.array([[266., 686.], [266., 19.], [931., 20.], [931., 701.]], dtype="float32")
        self.M = cv2.getPerspectiveTransform(src_points, dst_points)

    def spin(self):
        ret, img = self.cap.read()

        line_color = (0, 255, 0)
        cv2.line(img, (3, 570), (387, 460), line_color)
        cv2.line(img, (3, 570), (1241, 520), line_color)
        cv2.line(img, (387, 460), (1106, 472), line_color)
        cv2.line(img, (1106, 472), (1241, 520), line_color)
        cv2.imshow('img_cap',img)
        cv2.waitKey(2)

        if ret == True:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kernel = np.ones((3,3), np.uint8)
            gray_img = cv2.erode(gray_img, kernel, iterations=1)
            cv2.imshow('gray_img', gray_img)

            origin_thr = np.zeros_like(gray_img)
            origin_thr[(gray_img >= 125)] = 255
            cv2.imshow('origin_thr', origin_thr)

            binary_warped = cv2.warpPerspective(origin_thr, self.M, (1280, 720), cv2.INTER_LINEAR)
            line_color = (255, 255, 255)
            cv2.line(binary_warped, (266, 686), (266, 19), line_color)
            cv2.line(binary_warped, (266, 686), (931, 701), line_color)
            cv2.line(binary_warped, (266, 19), (931, 20), line_color)
            cv2.line(binary_warped, (931, 20), (931, 701), line_color)
            cv2.imshow('binary_warped', binary_warped)
            histogram_x = np.sum(binary_warped[int(binary_warped.shape[0] /2):, :], axis=0)
            lane_base = np.argmax(histogram_x)
            print(lane_base)
            midpoint_x = int(histogram_x.shape[0]/2)
            print(midpoint_x)

            histogram_y = np.sum(binary_warped[0:binary_warped.shape[0], :], axis=1)
            midpoint_y = 320 #int(histogram.shape[0]/2)
            upper_half_histSum = np.sum(histogram_y[0:midpoint_y])
            lower_half_histSum = np.sum(histogram_y[midpoint_y: ])
            try:
                hist_sum_y_ratio = (upper_half_histSum)/(lower_half_histSum)
            except:
                hist_sum_y_ratio = 1
            print(hist_sum_y_ratio)
            nwindows = 10
            window_height = int(binary_warped.shape[0] / nwindows)
            print(window_height)
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            print(nonzeroy)
            print(nonzerox)

if __name__ == "__main__":
    cam = camera()
    while True:
        cam.spin()