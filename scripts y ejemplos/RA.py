#!/usr/bin/env python

import cv2          as cv
import numpy        as np

from umucv.stream   import autoStream
from umucv.htrans   import htrans, Pose
from umucv.util     import lineType, cube, Help
from umucv.contours import extractContours, redu


help = Help(
"""
HELP WINDOW

c: increase the size of the shape
d: decrease the size of the shape
m: alter the shape
left click: change the shape color

h: show/hide help
""")

# matriz de calibración sencilla dada la
# resolución de la imagen y el fov horizontal en grados
def Kfov(sz,hfovd):
    hfov = np.radians(hfovd)
    f = 1/np.tan(hfov/2)
    
    w,h = sz
    w2 = w / 2
    h2 = h / 2
    return np.array([[f*w2, 0,    w2],
                     [0,    f*w2, h2],
                     [0,    0,    1 ]])


def polygons(cs,n,prec=2):
    rs = [ redu(c,prec) for c in cs ]
    return [ r for r in rs if len(r) == n ]


def rots(c):
    return [np.roll(c,k,0) for k in range(len(c))]


def bestPose(K,view,model):
    poses = [ Pose(K, v.astype(float), model) for v in rots(view) ]
    return sorted(poses,key=lambda p: p.rms)[0]  


# Callback para detectar clicks
def fun(event, x, y, flags, param):
    global puntoClick
    if event == cv.EVENT_LBUTTONDOWN:
        puntoClick = (y,x)

# Ref: https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
def dentro_contornos(contornos, punto):
    x = punto[0]
    y = punto[1]
    contornos = contornos[0]
    n = len(contornos)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = contornos[0]
    for i in range(n+1):
        p2x,p2y = contornos[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside


# Marcador del script base
marker = np.array(
       [[0,   0,   0],
        [0,   1,   0],
        [0.5, 1,   0],
        [0.5, 0.5, 0],
        [1,   0.5, 0],
        [1,   0,   0]])

# Figura pirámide (propia)
pyramid = np.array([[0, 0, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                    [0, 1, 1],
                    [0, 0, 1],
                    [0.5, 0.5, 1.5],
                    [1, 0, 1],
                    [0.5, 0.5, 1.5],
                    [1, 1, 1],
                    [0.5, 0.5, 1.5],
                    [0, 1, 1],
                    [0.5, 0.5, 1.5]])

# Figura pirámide invertida (propia)
pyramidInv = np.array([[0, 0, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                    [0, 1, 1],
                    [0, 0, 1],
                    [0.5, 0.5, 0.5],
                    [1, 0, 1],
                    [0.5, 0.5, 0.5],
                    [1, 1, 1],
                    [0.5, 0.5, 0.5],
                    [0, 1, 1],
                    [0.5, 0.5, 0.5]])


stream = autoStream()
HEIGHT, WIDTH = next(stream)[1].shape[:2]
size = WIDTH,HEIGHT

K = Kfov(size, 60)

# Direccion en la que se mueve el muñeco
reverse = False

cv.namedWindow('Realidad Aumentada')
cv.setMouseCallback('Realidad Aumentada', fun)  

factorTam = 1
mov = 0
puntoClick = None
formaActual = cube
cubo = True
nframes = 0
indiceMarker = 0
pyramidOrg = pyramid.copy()
clickFigura = False

for key,frame in stream:
    key = cv.waitKey(1) & 0xFF
    if key == 27: break

    # Si se detecta click se activa o desactiva la animación
    if puntoClick != None:
        ant = clickFigura
        clickFigura = dentro_contornos([htrans(M,formaActual/2 * factorTam + [0,mov,0]).astype(int) for M in poses], (puntoClick[1],puntoClick[0]))
        if ant == True: clickFigura = not clickFigura
        puntoClick = None

    # La pirámide realiza un desplazamiento en función de las aristas del marcador si se ha hecho click encima
    if clickFigura:
        if nframes % 10 == 0: 
            pyramid += marker[indiceMarker]
            indiceMarker += 1
            
        if indiceMarker == len(marker):
            pyramid = pyramidOrg.copy()
            indiceMarker = 0
        nframes += 1
    

    # Control de la figura mediante teclado
    if key == ord('c') and factorTam < 4.5: # Aumentar tam
        factorTam *= 1.1
    elif key == ord('d') and factorTam > 0.5: # Disminuir tam
        factorTam /= 1.1
    elif key == ord('m'): # Cambiar la forma de la figura
        if not cubo:
            formaActual = cube
            cubo = True
        else:
            formaActual = pyramidInv
            cubo = False

    g = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cs = extractContours(g, minarea=5, reduprec=2)

    good = polygons(cs,6,3)

    poses = []
    for g in good:
        pose = bestPose(K,g,marker)
        if pose.rms < 2:
            poses += [pose.M]

    # Se establece el color del click si se efectúa
    if puntoClick != None:
        colR = frame[puntoClick[0]][puntoClick[1]][0]
        colG = frame[puntoClick[0]][puntoClick[1]][1]
        colB = frame[puntoClick[0]][puntoClick[1]][2]
        colRGB = (int(colR), int(colG), int(colB))
    # Rojo por defecto
    else:
        colRGB = (0,0,255)


    # Se dibujan las figuras en función del control
    cv.drawContours(frame,[htrans(M,marker).astype(int) for M in poses], -1, (0,255,255), 1, lineType)
    cv.drawContours(frame,[htrans(M,pyramid/2 * factorTam + [0,mov,0]).astype(int) for M in poses], -1, colRGB, 3, lineType)
    cv.drawContours(frame, [htrans(M,formaActual/2 * factorTam + [0,mov,0]).astype(int) for M in poses], -1, colRGB, 3, lineType)
    
    cv.imshow('Realidad Aumentada',frame)