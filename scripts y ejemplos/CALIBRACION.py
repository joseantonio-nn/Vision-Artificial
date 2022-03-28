#!/usr/bin/env python

import math
import cv2 as cv
from umucv.stream import autoStream
from umucv.util import putText

def fun(event, x, y, flags, param):
	global punto1, punto2
	if event == cv.EVENT_LBUTTONDOWN:
		if not punto1:
			punto1 = (x,y)
		elif not punto2:
			punto2 = (x,y)
		elif punto1 and punto2:
			punto2 = ()
			punto1 = (x,y)

cv.namedWindow("webcam")
cv.setMouseCallback("webcam", fun)

punto1 = ()
punto2 = ()

f = 656.95
w = 640
h = 480


def anguloEntrePuntos():
	u = (punto1[0] - w/2, punto1[1] - h/2, f)         
	v = (punto2[0] - w/2, punto2[1] - h/2, f)

	prod_escalar = sum([i*j for i,j in zip(u,v)])
	modulos = (sum([i ** 2 for i in u]) ** 0.5) * (sum([i ** 2 for i in v]) ** 0.5)

	return math.degrees(math.acos(prod_escalar / modulos))



for key, frame in autoStream():

	if punto1:
		cv.circle(frame, (punto1[0], punto1[1]), 5, (0,0,255), 2)
	if punto2:
		cv.circle(frame, (punto2[0], punto2[1]), 5, (0,0,255), 2)

	if punto1 and punto2:
		
		putText(frame, str(round(anguloEntrePuntos(), 2)) + " grados", orig=(5, 20), color=(200,255,200))
		cv.line(frame, punto1, punto2, (0,0,255), 2) 

	cv.imshow('webcam',frame)

cv.destroyAllWindows()