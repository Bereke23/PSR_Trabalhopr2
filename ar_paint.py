from copy import deepcopy
from pickle import TRUE
from re import T
import cv2  
import numpy as np
paintWindow = (0,0,0)
xs= []
ys= []
drawing = False
gui_image = None
cor =()
# Abre a imagem
def main():
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
