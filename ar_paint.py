#!/usr/bin/env python3
from copy import deepcopy
import json
from operator import le
from pickle import TRUE
import argparse
from re import T
import cv2
from cv2 import GC_BGD  
import numpy as np
from requests import patch
paintWindow = (0,0,0)
xs= []
ys= []
drawing = False
gui_image = None
cor =()
# Abre a imagem

def leitura(path):
    B ={}
    G ={}
    R ={}
    # Leitura da pasta JSON
    # Opening JSON file
    f = open(path)
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
    # Iterating through the json
    G ={'min': int(data['limits']['G']['min']), 'max': int(data['limits']['G']['max'])}
    B = {'min': int(data['limits']['B']['min']), 'max': int(data['limits']['B']['max'])}
    R = {'min': int(data['limits']['R']['min']), 'max': int(data['limits']['R']['max'])}
    # Closing file
    f.close()
    return R, G , B

def Inicializacao():
    # Definição dos argumentos de entrada:
    parser = argparse.ArgumentParser(description='Modo de funcionamento')
    parser.add_argument('-j_JSON','--json_JSON',type = str, required= True,
                    help='Full path to json file')
    args = vars(parser.parse_args())
    path = args['json_JSON'] # A localização do ficheiro json
    return path


def main():
    global gui_image
    height = 400 # Defenição do tamanho da window, ou seja a altura e o grossura
    width = 600
    path = Inicializacao()
    R,G,B = leitura(path)


    capture = cv2.VideoCapture(0)
    window_original = 'Janela de video real'
    window_paint = 'Paint'
    window_segment = 'Objeto parametrizado'
    cv2.namedWindow(window_original,cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow(window_paint,cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(window_original,height,width) # Mesma dimensão da janela
    window_paint = np.zeros((height,width,3)) + (255,255,255)
    gui_image = deepcopy(window_paint)
    #Inicialmente estamos a criar um array de apenas de zeros com tres canais (0,0,0,3) ( caso nós quissesemos uma janela preta bastava meter a primieira parte do comando sem somar nada)
    #Na segunda parte onde estamos a somar (255,255,0),nós estamos a alterar o array de zeros em um array de (255,255,0,3)
    #E segundo os padrões de RGB, o array (255,255,0) é cor azul clara e (255,255,255) é branco e (0,0,0) é preto
    
    #cv2.setMouseCallback('Paint',desenhar)

    while True:
        _, image = capture.read()  # get an image from the camera
        # add code to show acquired image
        image_mask = cv2.inRange(image,(R['min'],G['min'],B['min']), (R['max'],G['max'],B['max']))
        # Threshold it so it becomes binary
        _, thresh = cv2.threshold(image_mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # You need to choose 4 or 8 for connectivity type
        connectivity = 4  
        # Perform the operation
        output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
        # Get the results
        # The first cell is the number of labels
        num_labels = output[0]
        # The third cell is the stat matrix
        stats = output[2]
        # The fourth cell is the centroid matrix
        centroids = output[3]
        for k in range(1,num_labels):
            # size filtering
                image_copy = deepcopy(image) 
                area = stats[k, cv2.CC_STAT_AREA]
                if area < 150:continue
                x1 = stats[k, cv2.CC_STAT_LEFT]
                y1 = stats[k, cv2.CC_STAT_TOP]
                w = stats[k, cv2.CC_STAT_WIDTH]
                h = stats[k, cv2.CC_STAT_HEIGHT]
                pt1 = (x1, y1)
                pt2 = (x1+ w, y1+ h)
                (X, Y) = centroids[k]
                cv2.rectangle(image_copy,pt1,pt2,(0, 255, 0), 3)
                cv2.circle(image_copy, (int(X),int(Y)),4, (0, 0, 255), -1)
                cv2.imshow(window_original,image_copy)
        #cv2.imshow('Paint',gui_image)
        k= cv2.waitKey(1)
        if k == ord('q'):   # wait for esckey to exit
            break
    capture.release()      
    cv2.destroyAllWindows()
    


def desenhar(event,x,y,flags,userdata):
    global drawing, gui_image ,cor 
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing:
            drawing = False
        else:
            drawing = True
            del xs[:]
            del ys[:]
            c= cv2.waitKey(0) 
            if c == 98: # Red
                cor = (250,0,0)
            if c == 103: # Green
                cor = (0,250,0)
            if c == 114: # Blue
                cor = (0,0,250)

    if event == cv2.EVENT_MOUSEMOVE:
        if drawing and cor != (0,0,0):
            xs.append(x)
            ys.append(y)

            for n in range(0,len(xs)-1):
                x1 = xs[n]
                y1 = ys[n]
                x2 = xs[n+1]
                y2 = ys[n+1]
                cv2.line(gui_image,(x1,y1),(x2,y2),cor,2)
   

    



if __name__ == '__main__':
    main()       
