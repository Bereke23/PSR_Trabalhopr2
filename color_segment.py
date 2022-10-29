#!/usr/bin/env python3
import cv2
import numpy as np
import json


def fall(x):
    #print(x)
    pass


def main():
    # initial setup
    alpha_slider_max = 255

    capture = cv2.VideoCapture(0)
    window_original_name = 'Janela de video real'
    window_segment_name = 'Janela de parametrizacao'
    cv2.namedWindow(window_original_name,cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(window_segment_name,cv2.WINDOW_AUTOSIZE)
    trachbaRmin = 'Rmin x %d' % alpha_slider_max
    trachbaRmax = 'Rmax x %d' % alpha_slider_max   
    trachbaGmin = 'Gmin x %d' % alpha_slider_max   
    trachbaGmax = 'Gmax x %d' % alpha_slider_max  
    trachbaBmin = 'Bmin x %d' % alpha_slider_max 
    trachbaBmax = 'Bmax x %d' % alpha_slider_max 

    cv2.createTrackbar(trachbaRmin, window_segment_name , 0, alpha_slider_max, fall)
    cv2.createTrackbar(trachbaRmax, window_segment_name , 255, alpha_slider_max, fall)
    cv2.createTrackbar(trachbaGmin, window_segment_name , 0, alpha_slider_max, fall)
    cv2.createTrackbar(trachbaGmax, window_segment_name , 255, alpha_slider_max, fall)
    cv2.createTrackbar(trachbaBmin, window_segment_name , 0, alpha_slider_max, fall)
    cv2.createTrackbar(trachbaBmax, window_segment_name , 255, alpha_slider_max, fall)
    while True:
        _, image = capture.read()  # get an image from the camera
 
         # add code to show acquired image
        Rmin = int(cv2.getTrackbarPos(trachbaRmin, window_segment_name))
        Rmax = int(cv2.getTrackbarPos(trachbaRmax, window_segment_name))
        Gmin = int(cv2.getTrackbarPos(trachbaGmin, window_segment_name))
        Gmax = int(cv2.getTrackbarPos(trachbaGmax, window_segment_name))
        Bmin = int(cv2.getTrackbarPos(trachbaBmin, window_segment_name))
        Bmax = int(cv2.getTrackbarPos(trachbaBmax, window_segment_name))


            #masking the image using inRange() function
        image_mask = cv2.inRange(image,(Rmin,Gmin,Bmin), (Rmax,Gmax,Bmax))
        
        cv2.imshow(window_original_name,image)
        cv2.imshow(window_segment_name,image_mask)
        k = cv2.waitKey(1)
        if k == ord('w'): # when w is pressed it creates a json file
           dictionary = { "limits":{"B":{ "max": str(Bmax) ,"min":  str(Bmin)}, "G":{ "max":str(Gmax),"min": str(Gmin) }, "R":{ "max":str(Rmax),"min": str(Rmin) } }}
           with open("limits.json", "w") as outfile:
            json.dump(dictionary, outfile)
        if k == ord('q'): # when q is pressed it breaks the loop
            break

    
    capture.release()
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    main()


