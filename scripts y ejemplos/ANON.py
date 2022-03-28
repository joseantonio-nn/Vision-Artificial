#!/usr/bin/env python

# Se parte del script sugerido en el enunciado

import face_recognition
import cv2 as cv
import numpy as np
import time
from umucv.util import putText
from umucv.stream import autoStream
import glob


def readrgb(filename):
    return cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB) 


def readModels(path):
    fmods = sorted([name for name in glob.glob(path+'/*.*') if name[-3:] != 'txt'])
    models = [ readrgb(f) for f in fmods ]
    return fmods, models


# Adaptado de: https://www.pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/
def pixelate_face(face_frame, blocks=3):
	
    # Se divide el frame en los bloques que se quieran
	(h, w) = face_frame.shape[:2]
	xCoords = np.linspace(0, w, blocks + 1, dtype="int")
	yCoords = np.linspace(0, h, blocks + 1, dtype="int")
	
    # Se recorren todos los bloques en los que ha dividido el frame 
	for i in range(1, len(yCoords)):
		for j in range(1, len(xCoords)):
			
            # Se toman las coordenadas del bloque actual
			x1 = xCoords[j - 1]
			y1 = yCoords[i - 1]
			x2 = xCoords[j]
			y2 = yCoords[i]

            # Se extrae la región de interés asociada al bloque actual
			roi = face_frame[y1:y2, x1:x2]

            # Se calcula el valor medio de esa región
			(B, G, R) = [int(x) for x in cv.mean(roi)[:3]]

            # Se establece ese valor medio en el bloque actual
			cv.rectangle(face_frame, (x1, y1), (x2, y2), (B, G, R), -1)

	return face_frame

filenames, models = readModels('anon')
names = [ x.split('/')[-1].split('.')[0].split('-')[0] for x in filenames ]
encodings = [ face_recognition.face_encodings(x)[0] for x in models ]

print(encodings[0].shape)

for key, frame in autoStream():

    t0 = time.time()

    face_locations = face_recognition.face_locations(frame)
    t1 = time.time()

    face_encodings = face_recognition.face_encodings(frame, face_locations)
    t2 = time.time()

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        match = face_recognition.compare_faces( encodings, face_encoding)

        name = "Unknown"
        for n, m in zip(names, match):
            if m:
                name = n

        # El código pixela la cara si se encuentra en una imagen del directorio /gente
        if name != "Unknown":
            frame[top:bottom,left:right] = pixelate_face(frame[top:bottom,left:right], 10)

        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
        putText(frame, name, orig=(left+3,bottom+16))

    putText(frame, f'{(t1-t0)*1000:.0f} ms {(t2-t1)*1000:.0f} ms')

    cv.imshow('Video', frame)