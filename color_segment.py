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
    a = 10
    b = 50
    c = 50
    d = 255
    e = 25
    f = 255
    capture = cv2.VideoCapture(0)
    window_original = 'Janela de video real'
    window_segment = 'Janela de parametrização'
    cv2.namedWindow(window_original,cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(window_segment,cv2.WINDOW_AUTOSIZE)


    trachbaRmin = 'Rmin x %d' % alpha_slider_max
    trachbaRmax = 'Rmax x %d' % alpha_slider_max   
    trachbaGmin = 'Gmin x %d' % alpha_slider_max   
    trachbaGmax = 'Gmax x %d' % alpha_slider_max  
    trachbaBmin = 'Bmin x %d' % alpha_slider_max 
    trachbaBmax = 'Bmax x %d' % alpha_slider_max 
    cv2.createTrackbar(trachbaRmin, window_segment , 0, alpha_slider_max, fall)
    cv2.createTrackbar(trachbaRmax, window_segment , 0, alpha_slider_max, fall)
    cv2.createTrackbar(trachbaGmin, window_segment , 0, alpha_slider_max, fall)
    cv2.createTrackbar(trachbaGmax, window_segment , 0, alpha_slider_max, fall)
    cv2.createTrackbar(trachbaBmin, window_segment , 0, alpha_slider_max, fall)
    cv2.createTrackbar(trachbaBmax, window_segment , 0, alpha_slider_max, fall)
    while True:
        ret, image_rgb = capture.read()  # get an image from the camera
 
        if ret:
             # add code to show acquired image
            
            if int(cv2.getTrackbarPos(trachbaRmin, window_segment)) > 0:
                a = int(cv2.getTrackbarPos(trachbaRmin, window_segment))
                b = int(cv2.getTrackbarPos(trachbaRmax, window_segment))
                c = int(cv2.getTrackbarPos(trachbaGmin, window_segment))
                d = int(cv2.getTrackbarPos(trachbaGmax, window_segment))
                e = int(cv2.getTrackbarPos(trachbaBmin, window_segment))
                f = int(cv2.getTrackbarPos(trachbaBmax, window_segment))

            #masking the image using inRange() function
            image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2HSV)
            image_hsv_mask = cv2.inRange(image_hsv,(a, c, e), (b, d, f))
        
        cv2.imshow(window_original,image_rgb)
        cv2.imshow(window_segment,image_hsv_mask )
        k = cv2.waitKey(1)
        if k == ord('w'):
           dictionary = { "limits":{"B":{ "max":f ,"min": e }, "G":{ "max":d,"min": c }, "R":{ "max":b,"min": a } }}
           with open("limits.json", "w") as outfile:
            json.dump(dictionary, outfile)
        if k == ord('q'):
            break

    
        # add code to wait for a key press
    capture.release()
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    main()