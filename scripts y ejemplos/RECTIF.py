#!/usr/bin/env python

import numpy as np
import cv2   as cv
import sys
import math


# Variable para guardar las coordenadas de las esquinas del carnet
esquinasRect = []

# Control del bucle principal
puntosMarcados = False

# Callback que va guardándose las coordenadas de los clicks mientras muestra los puntos

def getCoordsRect(event, x, y, flags, param):

    # Si se hace click izquierdo
    if event == cv.EVENT_LBUTTONDOWN:

        # Se añade la posición del click a la lista de esquinas
        esquinasRect.append([x,y])
        
        # Mientras no se tengan 4 puntos se pueden ir agregando y uniéndose mediante una línea
        nEsquinas = len(esquinasRect)

        if nEsquinas < 4:
            # Se dibuja el círculo donde se hace click
            cv.circle(imgToRectif, (x, y), 4, (255,0,0), -1)

            cv.line(imgToRectif, (esquinasRect[nEsquinas-2][0], esquinasRect[nEsquinas-2][1]), 
                                 (esquinasRect[nEsquinas-1][0], esquinasRect[nEsquinas-1][1]), 
                                 (255,0,0), 2)
        
        # Si se tienen 4 ya hay que cerrar el rectángulo
        elif nEsquinas == 4: 

            # Se dibuja el círculo donde se hace click
            cv.circle(imgToRectif, (x, y), 4, (255,0,0), -1)

            cv.line(imgToRectif, (esquinasRect[3][0], esquinasRect[3][1]), 
                                 (esquinasRect[0][0], esquinasRect[0][1]), 
                                 (255,0,0), 2)
            cv.line(imgToRectif, (esquinasRect[2][0], esquinasRect[2][1]), 
                                 (esquinasRect[3][0], esquinasRect[3][1]), 
                                 (255,0,0), 2)
            global puntosMarcados
            puntosMarcados = True

        # Cuando se hace click más de 4 veces se resetea la lista de esquinas
        else:
            esquinasRect.clear() 

# Variable para guardar las coordenadas de los puntos entre los que se quiere medir una distancia
puntos = []


# Callback que se utiliza para medir y mostrar la distancia entre dos puntos de la imagen rectificada

def getDistanciaPuntos(event, x, y, flags, param):

    # Si se hace click izquierdo
    if event == cv.EVENT_LBUTTONDOWN:
        
        puntos.append([x, y])

        # Si no hay dos puntos ya, se dibuja en ella un círculo
        if len(puntos) < 2:
            cv.circle(imgRectif, (x, y), 5, (255,0,0), -1)
            
        # Si ya se tienen dos puntos, se dibuja el segundo y se unen con una línea
        elif len(puntos) == 2:
            cv.circle(imgRectif, (x, y), 5, (255,0,0), -1)
            cv.line(imgRectif, (puntos[0][0], puntos[0][1]), (puntos[1][0], puntos[1][1]), (255,0,0), 2)
            
            # Cálculo de la distancia euclídea
            dx2 = (puntos[0][0]-puntos[1][0])**2          
            dy2 = (puntos[0][1]-puntos[0][1])**2
            distancia = math.sqrt(dx2 + dy2)
            
            # Se realiza una conversión a centímetros
            distanciaReal = round(distancia * h / longitud, 2)
            cv.putText(imgRectif, repr(distanciaReal) + " cm", org=(25, 60), fontFace=0, fontScale=1, color=(255,255,255), thickness=2)


# Medidas del carnet de conducir en cm
#w, h = 8.6, 5.4

# Medidas del paralelogramo que se va a medir en el área del campo
#w, h = 1650, 1100

# Medidas escorpión resina
w, h = 7.2, 4.2
aspectRatio = w/h

# Longitud del cuadro rectificado
longitud = 100

# Posiciones en la imagen del cuadro una vez rectificado
x,y = 200,500
real = np.array([[x,y],
                 [x+longitud*aspectRatio, y],
                 [x+longitud*aspectRatio, (y+longitud)],
                 [x, y+longitud]])

# Se lee la imagen de entrada a rectificar
imgToRectif = cv.imread(sys.argv[1])

# Control del bucle
isRectificada = False

while True:

    # Presionando 'Esc' nos salimos
    key = cv.waitKey(1) & 0xFF
    if key == 27: break
    
    # Se muestra la imagen para marcar los puntos del objeto con medidas conocidas
    cv.setMouseCallback("ImgToRectif", getCoordsRect)
    cv.imshow("ImgToRectif", imgToRectif)
    
    # Si se han marcado 
    if not isRectificada and puntosMarcados:
        view = np.array([
            [esquinasRect[0][0], esquinasRect[0][1]],
            [esquinasRect[1][0], esquinasRect[1][1]],
            [esquinasRect[2][0], esquinasRect[2][1]],
            [esquinasRect[3][0], esquinasRect[3][1]]])
    
        H, _ = cv.findHomography(view, real)
        # Se copia la imagen sin dibujos primero por si se quiere volver a medir
        # Ref: https://stackoverflow.com/questions/47005416/opencv-clear-screen
        cleanImgRectif = cv.warpPerspective(imgToRectif, H, (800, 800))
        imgRectif = cleanImgRectif.copy()
        isRectificada = True
        
    # Si ya se tienen los puntos sobre el objeto conocido se dibuja la imagen con rectificación de plano
    elif isRectificada:
        cv.setMouseCallback("RECTIF", getDistanciaPuntos)
        cv.imshow("RECTIF", imgRectif)

        # Si se quiere volver a medir, pulsar 'r' 
        if key == ord('r'):
            imgRectif = cleanImgRectif.copy()
            puntos.clear()
      
cv.destroyAllWindows()
