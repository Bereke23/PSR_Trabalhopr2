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
import time



paintWindow = (0,0,0)
xs= []
ys= []
drawing = False
gui_image = None
cor = (0,0,0)
window_paint_name = 'Paint'
thickness = 5

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


def main():
    global gui_image
    capture = cv2.VideoCapture(0)
    path = Inicializacao() # Vai buscar o caminho do ficheiro JSON
    R,G,B = leitura(path) # Dicionario com os max e min de RGB cada um
    _, image = capture.read()  # get an image from the camera
    height,width, _ = np.shape(image)
    window_paint = np.zeros((height,width,3)) #+ (255,255,255) # Definição do pain (quadro branco)
    window_paint.fill(255)
    gui_image = deepcopy(window_paint)
    while True:

        _, image = capture.read()  # get an image from the camera
        height,width, _ = np.shape(image)
        image = cv2.resize(image,(width,height)) # Resize the image

        window_original = 'Janela de video real'

        cv2.namedWindow(window_original,cv2.WINDOW_AUTOSIZE)
        #cv2.namedWindow(window_paint,cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(window_original,height,width) # Mesma dimensão da janela
        cv2.imshow(window_original,image) # Show the image

        cv2.namedWindow(window_paint_name,cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(window_paint_name,height,width) # Mesma dimensão da janela
        cv2.imshow(window_paint_name,gui_image) # Show the image

        # Show the image
        # aplicamos a mask na imagem live stream
        image_mask = cv2.inRange(image,(R['min'],G['min'],B['min']), (R['max'],G['max'],B['max']))

        window_mask = 'Object detected with Mask'
        cv2.namedWindow(window_mask,cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(window_mask,height,width) # Mesma dimensão da janela
        image_wMask = cv2.bitwise_and(image, image, mask=image_mask)
        cv2.imshow(window_mask,image_wMask) # Show the image

        # A parte seguinte está relacionada com a identificação do centroide do objeto
        # Simplesmente segui as indicações do link que o prof mandou
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
            # Identifica as difrentes ares na imagem
            area = stats[k, cv2.CC_STAT_AREA]
            # Se a area for menor que 150 o programa começa a parar
            if area < 150:continue
            # Se a area for superior ele continua
            x1 = stats[k, cv2.CC_STAT_LEFT] # x do centroide
            y1 = stats[k, cv2.CC_STAT_TOP] # y do centroide
            w = stats[k, cv2.CC_STAT_WIDTH] # largura do objeto
            h = stats[k, cv2.CC_STAT_HEIGHT] # altura do objeto
            pt1 = (x1, y1)
            pt2 = (x1+ w, y1+ h)
            (X, Y) = centroids[k]
            cv2.rectangle(image_copy,pt1,pt2,(0, 255, 0), 3) # faz um retangulo a volta do objeto
            cv2.line(image_copy,(int(X)-5,int(Y)),(int(X)+5,int(Y)),(0, 0, 255),thickness=2)
            cv2.line(image_copy,(int(X),int(Y)-5),(int(X),int(Y)+5),(0, 0, 255),thickness=2)
            # cv2.circle(image_copy, (int(X),int(Y)),4, (0, 0, 255), -1) # faz um circulo no ponto do centroide
            cv2.imshow(window_original,image_copy) # mosta a imagem real com os contornos

            #cv2.setMouseCallback(X,Y) # Função que desenha na janela do paint

            desenhar(int(X),int(Y))

        k= cv2.waitKey(1)
        if k == ord('q'):   # wait for esckey to exit
            break
    capture.release()
    cv2.destroyAllWindows()


def desenhar(x,y):  # Função que desenha na janela do paint
    d = None
    #return time with _  as  a separator using time module
    tempo = time.ctime().replace(' ','_')
    file_name = 'drawing_' + str(tempo) + '.jpg'   
    global gui_image, cor, window_paint_name , thickness



    
    c= cv2.waitKey(1)
    if c == ord('b'): # Blue color
        cor = (255,0,0)
    elif c == ord('g'): # Green color
        cor = (0,255,0)
    elif c == ord('r'): # Red color
        cor = (0,0,255)
    elif c == ord('c'): # Clear paint window
        gui_image.fill(255)
    elif c == ord('+'):   # começa a desenhar com um pincel maior
        thickness=thickness + 1
    elif c == ord('-'):  # começa a desenhar com um pincel menor
        if thickness == 1: # não deixa mir abaixo de um
            thickness=thickness
        if thickness >1:
            thickness = thickness -1         
        
    elif c == ord('w'): # guarda a imagem ao clicar na tecla w
        cv2.imwrite(file_name,gui_image)
    if c==ord('q'):
        cv2.destroyAllWindows()
        exit(0)


    if cor != (0,0,0) and len(xs)>2:  # Se a cor for diferente de preto
        x1 = x
        y1 = y
        x2 = xs[len(xs)-1]
        y2 = ys[len(ys)-1]
        print(thickness)
        cv2.line(gui_image,(x1,y1),(x2,y2),cor,thickness)
        cv2.imshow(window_paint_name,gui_image)
    
    xs.append(x)
    ys.append(y)
  
    



if __name__ == '__main__':
    main()
