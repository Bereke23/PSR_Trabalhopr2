#!/usr/bin/env python3
import cv2
import numpy as np
import json


def fall(x):
    #print(x)
    pass


def main():
    # initial setup
    team = {}
    alpha_slider_max = 255

    capture = cv2.VideoCapture(0)
    window_original = 'Janela de video real'
    window_segment = 'Janela de parametrizacao'
    cv2.namedWindow(window_original,cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(window_segment,cv2.WINDOW_AUTOSIZE)
    trachbaRmin = 'Rmin x %d' % alpha_slider_max
    trachbaRmax = 'Rmax x %d' % alpha_slider_max   
    trachbaGmin = 'Gmin x %d' % alpha_slider_max   
    trachbaGmax = 'Gmax x %d' % alpha_slider_max  
    trachbaBmin = 'Bmin x %d' % alpha_slider_max 
    trachbaBmax = 'Bmax x %d' % alpha_slider_max 

    cv2.createTrackbar(trachbaRmin, window_segment , 0, alpha_slider_max, fall)
    cv2.createTrackbar(trachbaRmax, window_segment , 255, alpha_slider_max, fall)
    cv2.createTrackbar(trachbaGmin, window_segment , 0, alpha_slider_max, fall)
    cv2.createTrackbar(trachbaGmax, window_segment , 255, alpha_slider_max, fall)
    cv2.createTrackbar(trachbaBmin, window_segment , 0, alpha_slider_max, fall)
    cv2.createTrackbar(trachbaBmax, window_segment , 255, alpha_slider_max, fall)
    while True:
        _, image_rgb = capture.read()  # get an image from the camera
 
         # add code to show acquired image
        Rmin = int(cv2.getTrackbarPos(trachbaRmin, window_segment))
        Rmax = int(cv2.getTrackbarPos(trachbaRmax, window_segment))
        Gmin = int(cv2.getTrackbarPos(trachbaGmin, window_segment))
        Gmax = int(cv2.getTrackbarPos(trachbaGmax, window_segment))
        Bmin = int(cv2.getTrackbarPos(trachbaBmin, window_segment))
        Bmax = int(cv2.getTrackbarPos(trachbaBmax, window_segment))


            #masking the image using inRange() function
        image_mask = cv2.inRange(image_rgb,(Rmin,Gmin,Bmin), (Rmax,Gmax,Bmax))
        
        cv2.imshow(window_original,image_rgb)
        cv2.imshow(window_segment,image_mask)
        k = cv2.waitKey(1)
        if k == ord('w'):
           dictionary = { "limits":{"B":{ "max": Bmax ,"min":  Bmin}, "G":{ "max":Gmax,"min": Gmin }, "R":{ "max":Rmax,"min": Rmin } }}
           with open("limits.json", "w") as outfile:
            json.dump(dictionary, outfile)
        if k == ord('q'):
            break

    
        # add code to wait for a key press
    capture.release()
    cv2.waitKey(0)

    
if __name__ == '__main__':
    main()


