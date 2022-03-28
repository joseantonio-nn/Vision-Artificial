#!/usr/bin/env python

import numpy as np
import cv2 as cv
from datetime import datetime

from umucv.util import ROI, putText
from umucv.util import Video
from umucv.stream import autoStream

# Se crea la ventana que mostrará los frames capturados
cv.namedWindow("input")
cv.moveWindow('input', 0, 0)

region = ROI("input")
video = Video(fps=15)

frame_anterior = None
frames_counter = 0

# Se itera sobre cada frame capturado
for key, frame in autoStream():
    
    # Se convierte la imagen a escala de grises
    frame_actual = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Se suaviza la imagen
    frame_actual = cv.GaussianBlur(frame_actual, (21, 21), 0)

    # Se obtiene el primer frame y se salta al siguiente
    if frame_anterior is None:
        frame_anterior = frame_actual
        continue

    video.ON = False
    if region.roi:
        # Se obtienen las coordenadas de la región
        [x1,y1,x2,y2] = region.roi

        # Se realiza la diferencia absoluta
        resta = cv.absdiff(frame_anterior[y1:y2+1, x1:x2+1], frame_actual[y1:y2+1, x1:x2+1])

        # Se guarda la diferencia a partir de un valor
        umbral = cv.threshold(resta, 25, 255, cv.THRESH_BINARY)[1]

        # Se dilata el umbral para tapar agujeros
        umbral = cv.dilate(umbral, None, iterations=2)

        # Buscamos contorno en la imagen
        contornos, hierarchy = cv.findContours(umbral.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
        # Recorremos todos los contornos encontrados
        for c in contornos:

            # Eliminamos los contornos más pequeños
            if cv.contourArea(c) <= 1500:
                cv.circle(frame, (15,15), 10, (0, 0, 255), -1)
                # Se inicia la grabación de la ROI si se detecta movimiento
                video.ON = True
                video.write(frame)
                break

        # Se dibuja un rectángulo en las coordenadas de la ROI mostrando su tamaño
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))

    frame_anterior = frame_actual

    # Se dibuja cada frame captado por la webcam
    h,w,_ = frame.shape
    cv.imshow('input',frame)