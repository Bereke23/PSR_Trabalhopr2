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
import math


paintWindow = (0,0,0)
xs= []
ys= []
drawing = False
gui_image = None
cor = np.array([[0,0,0,0]])
window_paint_name = 'Paint'
video_window = 'Copy Janela de video (To draw)'
thickness_desenho = 5
usm = False
test_mode = False
cor_rato=(0,0,0)

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
    parser.add_argument('-t','--test_mode', action='store_true', help='test the program')
    args = vars(parser.parse_args())
    path = args['json'] # A localização do ficheiro json
    usm = args['use_shake_mode'] # Ativacao do use shake mode
    test_mode =args['test_mode']
    return path , usm , test_mode

# def desenharPinguim(): # Funcionalidade avançada 4 - Pintura numerada



def main():
    global gui_image , usm
    capture = cv2.VideoCapture(0)
    path , usm, test_mode = Inicializacao() # Vai buscar o caminho do ficheiro JSON
    R,G,B = leitura(path) # Dicionario com os max e min de RGB cada um
    _, image = capture.read()  # get an image from the camera
    height,width, _ = np.shape(image)
    window_paint = np.zeros((height,width,4)) #+ (255,255,255) # Definição do paint (quadro branco)
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
        # cv2.imshow(video_window,video_copy)
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
        if usm and test_mode:
            cv2.setMouseCallback(window_paint_name,desenharato)
        if usm and not test_mode:
            for k in range(1,num_labels):
                # size filtering
                image_copy = deepcopy(image)
                # Identifica as difrentes ares na imagem
                area = stats[k, cv2.CC_STAT_AREA]
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
        if not usm:
            for k in range(1,num_labels):
                    # size filtering
                    image_copy = deepcopy(image)
                    # Identifica as difrentes ares na imagem
                    
                    area = stats[k, cv2.CC_STAT_AREA]
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


def desenhar(x,y,usm,video_frame):  # Função que desenha na janela do paint
    d = None
    #return time with _  as  a separator using time module
    tempo = time.ctime().replace(' ','_')
    file_name = 'drawing_' + str(tempo) + '.jpg'
    global gui_image, cor, window_paint_name , thickness_desenho

    c= cv2.waitKey(1)
    if c == ord('b'):               # Blue color
        cor = np.append(cor, [[255,0,0,1]], axis=0)
    elif c == ord('g'):             # Green color
        cor = np.append(cor, [[0,255,0,1]], axis=0)
    elif c == ord('r'):             # Red color
        cor = np.append(cor, [[0,0,255,1]], axis=0)
    elif c == ord('c'):             # Clear paint window
        gui_image.fill(255)
    elif c == ord('+'):             # começa a desenhar com um pincel maior
        thickness_desenho=thickness_desenho + 1
    elif c == ord('-'):             # começa a desenhar com um pincel menor
        if thickness_desenho == 1:  # não deixa ir abaixo de um
            thickness_desenho=thickness_desenho
        if thickness_desenho >1:    # caso for superior a 1 deixa diminuir a grossura
            thickness_desenho = thickness_desenho -1
    elif c == ord('w'):             # guarda a imagem ao clicar na tecla w
        cv2.imwrite(file_name,gui_image)
    if c==ord('q'):
        cv2.destroyAllWindows()
        exit(0)
    if not np.array_equal(cor[cor.shape[0]-1], [0,0,0,0]):  # Se a cor for diferente de preto
        xs.append(x)
        ys.append(y)
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
            video_frame = cv2.cvtColor(video_frame,cv2.COLOR_BGR2BGRA)
            gui_image_h,gui_image_w,gui_image_c = gui_image.shape
            for i in range(0,gui_image_h):
                for j in range(0,gui_image_w):
                    if gui_image[i,j][3] != 255:
                        video_frame[i,j] = gui_image[i,j]
            cv2.imshow(video_window,video_frame)


def desenharato(event,x,y,flags,userdata):

    global drawing, gui_image ,cor_rato
     
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing:
            drawing = False
        else:
            drawing = True
            c= cv2.waitKey(1) 
            if c == 98: # Red
                cor_rato = (250,0,0)
            if c == 103: # Green
                cor_rato = (0,250,0)
            if c == 114: # Blue
                cor_rato = (0,0,250)    
            del xs[:]
            del ys[:]


    if event == cv2.EVENT_MOUSEMOVE:
        if drawing and cor_rato != (0,0,0):
            xs.append(x)
            ys.append(y)
            for n in range(0,len(xs)-1):
                x1 = xs[n]
                y1 = ys[n]
                x2 = xs[n+1]
                y2 = ys[n+1]
                cv2.line(gui_image,(x1,y1),(x2,y2),cor_rato,2)


if __name__ == '__main__':
    main()
