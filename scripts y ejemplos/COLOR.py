#!/usr/bin/env python

import cv2 as cv
import numpy as np

from umucv.util import ROI, putText, Help
from umucv.stream import autoStream
from matplotlib import pyplot as plt


help = Help(
"""
HELP WINDOW

c: add ROI to the list of models
x: clear list of models 
SPC: pause

h: show/hide help
""")

# Función que compara los histogramas del frame actual y el del ROI y devuelve su comparación utilizando Chi-cuadrado
def comparaHistogramas(roiActual, roi):
    
    # Se calculan los histogramas para ambas ROIs teniendo en cuenta los 3 canales
	histogramaRoiActual = cv.calcHist([roiActual], [0, 1, 2], None, [8, 8, 8], [0,256] + [0,256] + [0,256])
	histogramaRoi = cv.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0,256] + [0,256] + [0,256])
    
    # Se normaliza el histograma
	histogramaRoiActual = histogramaRoiActual / np.sum(histogramaRoiActual)  
	histogramaRoi = histogramaRoi / np.sum(histogramaRoi)

	return cv.compareHist(histogramaRoi, histogramaRoiActual, cv.HISTCMP_CHISQR)


# Se crea la ventana que mostrará los frames capturados
cv.namedWindow("Cam")
cv.moveWindow('Cam', 0, 0)
region = ROI("Cam")

limite = 5
modelos = []

# Se itera sobre cada frame capturado
for key, frame in autoStream():
		
	help.show_if(key, ord('h'))

	if region.roi:

        # Si hay un ROI se obtiene la imagen
		[x1,y1,x2,y2] = region.roi
		roiActual = frame[y1:y2+1, x1:x2+1]

		# Si se pulsa 'c' se añade la ROI a la lista de modelos
		if key == ord('c'):
			modelos.append(cv.resize(roiActual, (200,200)))
			modelosApilados = np.hstack(modelos)
			cv.imshow("Models", modelosApilados)
			
		# Si se pulsa 'x' se borran los modelos almacenados
		elif key == ord('x'):
			modelos.clear()
			cv.destroyWindow("Models")
			cv.destroyWindow("Detected")

		diferencias = []
		diferenciaMinima = (None, float('inf')) # la diferencia mínima inicial es infinito

		# Se recorren los modelos para guardar el que más se parezca la ROI actual y mostrarlo
		for m in modelos:
			diferencia = comparaHistogramas(roiActual, m)
			diferencias.append(round(diferencia,2))
			if diferenciaMinima[1] > diferencia:
				diferenciaMinima = (m, diferencia)

		putText(frame, str(diferencias))

		if diferenciaMinima[1] < limite:
			cv.imshow("Detected", diferenciaMinima[0])
		else:
			cv.destroyWindow("Detected")

		# Se dibuja el rectángulo de la ROI
		cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
		putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))

	# Se dibuja cada frame captado por la webcam
	cv.imshow('Cam',frame)

cv.destroyAllWindows()