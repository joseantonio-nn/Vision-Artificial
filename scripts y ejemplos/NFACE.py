import sys
import dlib
import imutils
import cv2          as cv
import numpy        as np

from imutils        import face_utils
from umucv.stream   import autoStream


# Función que devuelve una cara normalizada según la posición exterior de los ojos
# y la parte superior de la boca

def normalizarCara(cara, landmarks):
    
    # Las referencias son los extremos de los ojos y la parte superior de la boca
    referencias = np.float32(landmarks[np.array([36, 45, 62])])
    # Se establecen los ojos a la misma altura y la boca entre ambos 
    nuevaPosicion = np.float32([[80,100],[160,100],[120,150]])

    # Se obtiene la matriz de transformación
    matriz = cv.getAffineTransform(referencias,nuevaPosicion)

    # Se normaliza la cara con la matriz de transformación
    caraNormalizada = cv.warpAffine(cara, matriz, (250, 250))
    return caraNormalizada


# Se carga el predictor
pathPredictor = "./dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(pathPredictor)

# Se lee la imagen
imagen = cv.imread(sys.argv[1]) # "./nface/monty-python1.jpg"
gray = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)

# Se detectan las caras de la imagen
dets = detector(gray, 1)
caras = []

# Se itera sobre las detecciones de la cara
for det in dets:
    
    # Se toman los 'landmarks' de la cara actual
    landmarks = predictor(gray, det)
    landmarks = face_utils.shape_to_np(landmarks)

    # Se añade la normalización de esa cara usando los 
    # 'landmarks' a la lista de caras
    caras.append(normalizarCara(imagen, landmarks))

# Se combinan las caras en fila en una sola imagen
carasApiladas = np.hstack(caras)

# Se puede salir del programa pulsando la tecla 'Esc'
while(True):
    key = cv.waitKey(1) & 0xFF
    if key == 27: break
    cv.imshow("NFACE", carasApiladas)