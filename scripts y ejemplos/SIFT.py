#!/usr/bin/env python

import time
import cv2 as cv
import numpy as np

from umucv.util import putText,Help
from umucv.stream import autoStream


help = Help(
"""
HELP WINDOW

c: add model to compare
h: show/hide help
""")

sift = cv.AKAZE_create()
matcher = cv.BFMatcher()
modelos = []

for key, frame in autoStream():

	help.show_if(key, ord('h'))

	t0 = time.time()
	keypoints, descriptors = sift.detectAndCompute(frame, mask=None)
	if len(keypoints) == 0:
		descriptors = None
	
	t1 = time.time()
	putText(frame, f'{len(keypoints)} pts  {1000*(t1-t0):.0f} ms')	
	
	if key == ord('c'):
		modelos.append((keypoints, descriptors, frame))	

	if len(modelos) == 0:
		flag = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
		cv.drawKeypoints(frame, keypoints, frame, color=(100,150,255), flags=flag)
		cv.imshow('SIFT', frame)

	else:
		mejorPorcentaje = 0
		segundoMejorPorcentaje = 0
		mejorFrame = frame
		t2, t3 = 0, 0
		for modelo in modelos:

			t2 = time.time()
			# Se guardan las dos mejores coincidencias de cada punto
			matches = matcher.knnMatch(descriptors, modelo[1], k=2)
			t3 = time.time()

			# Se guardan las coincidencias que son mucho mejores que
			# que la segunda mejor. Si un punto se parece a dos puntos diferentes del modelo se elimina
			good = []
			for m in matches:
				if len(m) >= 2:
					best,second = m
					if best.distance < 0.75*second.distance:
						good.append(best)
			
			porcentaje = len(good) * 100 / len(keypoints)
			if mejorPorcentaje < porcentaje:
				segundoMejorPorcentaje = mejorPorcentaje
				mejorPorcentaje = porcentaje
				mejorFrame = modelo[2]
			elif porcentaje > segundoMejorPorcentaje:
				segundoMejorPorcentaje = porcentaje


		putText(frame, f'{round(mejorPorcentaje,2)}% {round(segundoMejorPorcentaje,2)}%',orig=(0,50))
		cv.imshow("SIFT",np.hstack([frame,mejorFrame]))

