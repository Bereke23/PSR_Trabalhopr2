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
    print(R)
    print(G)
    print(B)
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
    path = Inicializacao()
    R,G,B = leitura(path)
    

    global gui_image
    height = 400
    width = 600
    cv2.namedWindow('Paint')
    # Defenição do tamanho da window, ou seja a altura e o grossura

    
    paintWindow = np.zeros((height, width,3)) + (255,255,255)
    gui_image = deepcopy(paintWindow)
    #Inicialmente estamos a criar um array de apenas de zeros com tres canais (0,0,0,3) ( caso nós quissesemos uma janela preta bastava meter a primieira parte do comando sem somar nada)
    #Na segunda parte onde estamos a somar (255,255,0),nós estamos a alterar o array de zeros em um array de (255,255,0,3)
    #E segundo os padrões de RGB, o array (255,255,0) é cor azul clara e (255,255,255) é branco e (0,0,0) é preto

    
    cv2.setMouseCallback('Paint',desenhar)

    while True:
        cv2.imshow('Paint',gui_image)
        k= cv2.waitKey(20)
        if k == 27:   # wait for esckey to exit
            break

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
