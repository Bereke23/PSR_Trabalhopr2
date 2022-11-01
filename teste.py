#!/usr/bin/env python3
from copy import deepcopy
import json
from operator import le
from pickle import TRUE
import argparse
from re import T, U
import cv2
from cv2 import GC_BGD
import numpy as np
from requests import patch
import time
import math


paintWindow = (0,0,0)
xs= []
ys= []
drawing = False
gui_image = None
cor = np.array([[0,0,0]])
window_paint_name = 'Paint'
video_window = 'Copy Janela de video (To draw)'
thickness_desenho = 5
usm = False

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
    parser.add_argument('-j','--json',type = str, required= True, help='Full path to json file')
    parser.add_argument('-usm','--use_shake_mode', action='store_true', help='Use shake prevention mode')
    args = vars(parser.parse_args())
    path = args['json'] # A localização do ficheiro json
    usm = args['use_shake_mode'] # Ativacao do use shake mode
    return path , usm

# def desenharPinguim(): # Funcionalidade avançada 4 - Pintura numerada



def main():
    global gui_image , usm
    capture = cv2.VideoCapture(0)
    path , usm = Inicializacao() # Vai buscar o caminho do ficheiro JSON
    R,G,B = leitura(path) # Dicionario com os max e min de RGB cada um
    _, image = capture.read()  # get an image from the camera
    height,width, _ = np.shape(image)
    window_paint = np.zeros((height,width,3)) #+ (255,255,255) # Definição do paint (quadro branco)
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


        video_copy = deepcopy(image)
        cv2.namedWindow(video_window,cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(video_window,height,width) # Mesma dimensão da janela
        cv2.imshow(video_window,video_copy)
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
            # if usm:
            #     listarea = [0,0,0,0,0,0]
            #     listarea.append(area)
            #     area1 = listarea[len(listarea)-1]
            #     area2 = listarea[len(listarea)-2]
            #     area3 = listarea[len(listarea)-3]
            #     area4 = listarea[len(listarea)-4]
            #     area5 = listarea[len(listarea)-5]
            #     media =(area1 + area2 + area3 + area4 + area5)/5
            #     if abs(media - area) > 494:
            #         pass
            #     if abs(media - area) < 494:
            #         if area < 150:continue
            #         x1 = stats[k, cv2.CC_STAT_LEFT] # x do centroide
            #         y1 = stats[k, cv2.CC_STAT_TOP] # y do centroide
            #         w = stats[k, cv2.CC_STAT_WIDTH] # largura do objeto
            #         h = stats[k, cv2.CC_STAT_HEIGHT] # altura do objeto
            #         pt1 = (x1, y1)
            #         pt2 = (x1+ w, y1+ h)
            #         (X, Y) = centroids[k]
            #         cv2.rectangle(image_copy,pt1,pt2,(0, 255, 0), 3) # faz um retangulo a volta do objeto
            #         cv2.line(image_copy,(int(X)-5,int(Y)),(int(X)+5,int(Y)),(0, 0, 255),thickness=2)
            #         cv2.line(image_copy,(int(X),int(Y)-5),(int(X),int(Y)+5),(0, 0, 255),thickness=2)
            #         cv2.imshow(window_original,image_copy) # mosta a imagem real com os contornos
            #         desenhar(int(X),int(Y),usm,video_copy)


            
            # Se a area for menor que 150 o programa começa a parar
            if area < 150:continue
            # Se a area for inferior ele continua
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
            cv2.imshow(window_original,image_copy) # mosta a imagem real com os contornos

            desenhar(int(X),int(Y),usm,video_copy)


        k= cv2.waitKey(1)
        if k == ord('q'):   # wait for esckey to exit
            break
    capture.release()
    cv2.destroyAllWindows()


def desenhar(x,y,usm,video_name):  # Função que desenha na janela do paint
    d = None
    #return time with _  as  a separator using time module
    tempo = time.ctime().replace(' ','_')
    file_name = 'drawing_' + str(tempo) + '.jpg'
    global gui_image, cor, window_paint_name , thickness_desenho

    c= cv2.waitKey(1)
    if c == ord('b'): # Blue color
        cor = np.append(cor, [[255,0,0]], axis=0)
    elif c == ord('g'): # Green color
        cor = np.append(cor, [[0,255,0]], axis=0)
    elif c == ord('r'): # Red color
        cor = np.append(cor, [[0,0,255]], axis=0)
    elif c == ord('c'): # Clear paint window
        gui_image.fill(255)
    elif c == ord('+'):   # começa a desenhar com um pincel maior
        thickness_desenho=thickness_desenho + 1
    elif c == ord('-'):  # começa a desenhar com um pincel menor
        if thickness_desenho == 1: # não deixa ir abaixo de um
            thickness_desenho=thickness_desenho
        if thickness_desenho >1: # caso for superior a 1 deixa diminuir a grossura
            thickness_desenho = thickness_desenho -1
    elif c == ord('w'): # guarda a imagem ao clicar na tecla w
        cv2.imwrite(file_name,gui_image)
    if c==ord('q'):
        cv2.destroyAllWindows()
        exit(0)

    if not np.array_equal(cor[cor.shape[0]-1], [0,0,0]):  # Se a cor for diferente de preto
        xs.append(x)
        ys.append(y)
        # cor = np.append(cor, cor, axis=0)
        if len(xs)>1:
            x = xs[len(xs)-2]
            y = ys[len(ys)-2]
            x2 = xs[len(xs)-1]
            y2 = ys[len(ys)-1]
            if usm:
                if math.dist((x,y),(x2,y2)) < 5:
                    cv2.line(gui_image,(x,y),(x2,y2),(int(cor[cor.shape[0]-1][0]),int(cor[cor.shape[0]-1][1]),int(cor[cor.shape[0]-1][2])),thickness_desenho)
                    cv2.imshow(window_paint_name,gui_image)
                else:
                    cv2.imshow(window_paint_name,gui_image)
            
            if not usm:
                cv2.line(gui_image,(x,y),(x2,y2),(int(cor[cor.shape[0]-1][0]),int(cor[cor.shape[0]-1][1]),int(cor[cor.shape[0]-1][2])),thickness_desenho)
                cv2.imshow(window_paint_name,gui_image)

            #cv2.line(gui_image,(x,y),(x2,y2),(int(cor[cor.shape[0]-1][0]),int(cor[cor.shape[0]-1][1]),int(cor[cor.shape[0]-1][2])),thickness_desenho)
            #cv2.imshow(window_paint_name,gui_image)
            # for i in range(len(xs)-1):
            #     x1 = xs[i]
            #     y1 = ys[i]
            #     x2 = xs[i-1]
            #     y2 = ys[i-1]
            #     if x1 != x2 or y1 != y2:
                    # cv2.line(video_name,(x1,y1),(x2,y2),(int(cor[i+1][0]),int(cor[i+1][1]),int(cor[i+1][2])),thickness_desenho)
                    # cv2.imshow(video_window,video_name)

        # PARTE USE SHAKE DETECTION
        # if usm == True and abs(x2 - x1) > 3 and abs(y2-y1) > 3:
        #    cv2.imshow(window_paint_name,gui_image)
        #    pass
        # if usm == False:
        #     cv2.line(gui_image,(x1,y1),(x2,y2),cor,thickness_desenho)
        #     cv2.imshow(window_paint_name,gui_image)
        # if usm and abs(x2 - x1) == 0:
        #     cv2.line(gui_image,(x1,y1),(x2,y2),cor,thickness_desenho)
        #     cv2.imshow(window_paint_name,gui_image)






if __name__ == '__main__':
    main()