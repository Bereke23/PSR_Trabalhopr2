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
from color_segment import main

paintWindow = (0,0,0)
xs= []
ys= []
drawing = False
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
    parser.add_argument('-j','--json',type = str, required= True,
                        help='Full path to json file')
    args = vars(parser.parse_args())
    path = args['json'] # A localização do ficheiro json
    return path

# def desenhar(x,y):
#     global gui_image ,cor 
#     c= cv2.waitKey(0) 
#     if c == 98: # Red
#         cor = (250,0,0)
#     if c == 103: # Green
#         cor = (0,250,0)
#     if c == 114: # Blue
#         cor = (0,0,250)

#     if cor != (0,0,0):
#         xs.append(x)
#         ys.append(y)

#         for n in range(0,len(xs)-1):
#             x1 = xs[n]
#             y1 = ys[n]
#             x2 = xs[n+1]
#             y2 = ys[n+1]
#             cv2.line(gui_image,(x1,y1),(x2,y2),cor,2)
   
def main():
    
    capture = cv2.VideoCapture(0)
    #Inicialmente estamos a criar um array de apenas de zeros com tres canais (0,0,0,3) ( caso nós quissesemos uma janela preta bastava meter a primieira parte do comando sem somar nada)
    #Na segunda parte onde estamos a somar (255,255,0),nós estamos a alterar o array de zeros em um array de (255,255,0,3)
    #E segundo os padrões de RGB, o array (255,255,0) é cor azul clara e (255,255,255) é branco e (0,0,0) é preto
    
    while True:
        
        window_original_name = 'Janela de video 2'
        window_paint_name = 'Paint'
        window_segment_name = 'Objeto parametrizado'
# Definição do tamanho da window, ou seja a altura e o grossura
        cv2.namedWindow(window_original_name,cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(window_paint_name,cv2.WINDOW_AUTOSIZE)
        #cv2.namedWindow(window_segment_name,cv2.WINDOW_AUTOSIZE)
        
        
        _, image = capture.read()
        # get an image from the camera
        height, width, _ = image.shape
        print(image.shape)
        cv2.resizeWindow(window_original_name,height,width) # Mesma dimensão da janela
        cv2.imshow(window_original_name,image)

        #paint window
        window_paint = np.zeros((height,width,1)) 
        window_paint.fill(0)
        cv2.imshow(window_paint_name,window_paint)

        #mostra o que aparece no color segment
        # cv2.resizeWindow(window_original_name,height,width) # Mesma dimensão da janela
        # cv2.imshow(window_original_name,image)

        path = Inicializacao()
        R,G,B = leitura(path)

        # add code to show acquired image
        image_mask = cv2.inRange(image,(B['min'],G['min'],R['min']), (B['max'],G['max'],R['max']))
        print(image_mask.shape)
        # Threshold it so it becomes binary
        _, thresh = cv2.threshold(image_mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        print(thresh.shape)
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
                # cv2.rectangle(image_copy,pt1,pt2,(0, 255, 0), 3)
                # cv2.circle(image_copy, (int(X),int(Y)),4, (0, 0, 255), -1)
                cv2.imshow('centroid',image_copy)

        k = cv2.waitKey(0)
        if k == ord('q'):   # wait for esckey to exit
            break

        
        cv2.destroyAllWindows()


def desenhar(x,y):
    global gui_image ,cor 
    c= cv2.waitKey(0) 
    if c == 98: # Red
        cor = (250,0,0)
    if c == 103: # Green
        cor = (0,250,0)
    if c == 114: # Blue
        cor = (0,0,250)

    if cor != (0,0,0):
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
