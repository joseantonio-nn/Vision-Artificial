#!/usr/bin/env python

import numpy as np
import cv2 as cv

from umucv.util import ROI, putText, Help
from umucv.stream import autoStream


help = Help(
"""
HELP WINDOW

g: gaussian blur filter
b: box filter
m: median filter
SPC: pausa

h: show/hide help
""")

def nada(v):
    pass

# Se crea la ventana que mostrará los frames capturados
cv.namedWindow("input")
cv.createTrackbar("Valor", "input", 5, 30, nada)
cv.moveWindow('input', 0, 0)

region = ROI("input")
gblur = False
box = False
median = False

# Se itera sobre cada frame capturado
for key, frame in autoStream():

    help.show_if(key, ord('h'))

    if key == ord('g'):
        gblur = True
        box = False
        median = False
    elif key == ord('b'):
        box = True
        gblur = False
        median = False
    elif key == ord('m'):
        median = True
        box = False
        gblur = False

    # Si se ha seleccionado una región se guardan sus coordenadas
    if region.roi:
        [x1,y1,x2,y2] = region.roi
        filterValue = cv.getTrackbarPos('Valor','input')
        
        reg = frame[y1:y2+1, x1:x2+1]

        # Se aplica el filtro que esté seleccionado
        filtro = reg
        if gblur:
            filtro = cv.GaussianBlur(reg, (0,0), filterValue) if filterValue > 0 else reg
            box = False
            median = False
        elif box:
            filtro = cv.boxFilter(reg, -1, (filterValue, filterValue)) if filterValue > 0 else reg
            gblur = False
            median = False
        elif median:
            if filterValue > 0 and filterValue % 2 == 1:
                filtro = cv.medianBlur(reg, filterValue)  
            elif filterValue > 0 and filterValue % 2 == 0:
                filtro = cv.medianBlur(reg, filterValue-1)
            else:
                filtro = reg  
            gblur = False
            box = False


        # Se aplica el filtro sobre la ROI
        frame[y1:y2+1, x1:x2+1] = filtro

        # Si se pulsa 'c' se copia la ROI a otra ventana
        if key == ord('c'):
            trozo = frame[y1:y2+1, x1:x2+1]
            cv.imshow("trozo", trozo)

        # Si se pulsa 'x' se deselecciona la ROI
        if key == ord('x'):
            region.roi = []

        # Se dibuja un rectángulo en las coordenadas de la ROI mostrando su tamaño
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))

    # Se dibuja cada frame captado por la webcam
    h,w,_ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow('input',frame)