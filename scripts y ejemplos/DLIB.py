import dlib
import cv2 as cv
import numpy as np
from math import hypot
from umucv.stream import autoStream

# Código inspirado en: https://pysource.com/2019/03/25/pigs-nose-instagram-face-filter-opencv-with-python/

# Función que devuelve las posiciones relevantes del ojo izq dados unos landmarks

def getPosOjoIzq(landmarks):
    eyeCenter = (landmarks.part(36).x + 10, landmarks.part(40).y - 10)
    eyeLeft = (landmarks.part(36).x, landmarks.part(36).y)
    eyeRight = (landmarks.part(39).x, landmarks.part(39).y)

    return (eyeCenter, eyeLeft, eyeRight)


# Función que devuelve las posiciones relevantes del ojo der dados unos landmarks

def getPosOjoDer(landmarks):
    eyeCenter = (landmarks.part(42).x + 15, landmarks.part(45).y - 5)
    eyeLeft = (landmarks.part(42).x, landmarks.part(42).y)
    eyeRight = (landmarks.part(45).x, landmarks.part(45).y)

    return (eyeCenter, eyeLeft, eyeRight)


# Función que devuelve el área ocupada por un ojo

def getEyeArea(top_left, eyeHeight, eyeWidth, eyeMask):
    eyeArea = frame[top_left[1]: top_left[1] + eyeHeight,
                    top_left[0]: top_left[0] + eyeWidth]
    
    return cv.bitwise_and(eyeArea, eyeArea, mask=eyeMask)


# Se prepara la máscara con el número de filas y columnas del frame
stream = autoStream()
_, frame = next(stream)
rows, cols, _ = frame.shape
eyeMask = np.zeros((rows, cols), np.uint8)

# Se carga el detector de caras
# El predictor se puede descargar en 
# https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlib\\shape_predictor_68_face_landmarks.dat")

eye_image = cv.imread("dlib\\dollar.png")

for key, frame in stream:

    eyeMask.fill(0)
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # El detector obtiene las caras en la cámara y se itera sobre ellas
    faces = detector(frame)
    for face in faces:

        # Se obtienen los 68 landmarks de la cara
        landmarks = predictor(gray_frame, face)

        # Se obtienen las posiciones relevantes del ojo en función de los landmarks
        (centerEye1, leftEye1, rightEye1) = getPosOjoIzq(landmarks)
        (centerEye2, leftEye2, rightEye2) = getPosOjoDer(landmarks)

        eyeWidth = int(hypot(leftEye1[0] - rightEye1[0], leftEye1[1] - rightEye1[1]) * 2)
        eyeHeight = int(eyeWidth * 1)

        # Se toma la esquina superior izq para el cálculo del área
        top_left1 = (int(centerEye1[0] - eyeWidth / 2),
                     int(centerEye1[1] - eyeHeight / 2))

        top_left2 = (int(centerEye2[0] - eyeWidth / 2),
                     int(centerEye2[1] - eyeHeight / 2))
        
        # Se cambia el tamaño de la imagen para que se adapte
        dollarEye = cv.resize(eye_image, (eyeWidth, eyeHeight))
        dollarEyeGray = cv.cvtColor(dollarEye, cv.COLOR_BGR2GRAY)
        _, eyeMask = cv.threshold(dollarEyeGray, 25, 255, cv.THRESH_BINARY_INV)
        
        # Se obtiene el área que va a ocupar la imagen en los ojos
        eyeArea1 = getEyeArea(top_left1, eyeHeight, eyeWidth, eyeMask)
        eyeArea2 = getEyeArea(top_left2, eyeHeight, eyeWidth, eyeMask)

        # Se coloca la imagen en ambos ojos
        frame[top_left1[1]: top_left1[1] + eyeHeight, 
              top_left1[0]: top_left1[0] + eyeWidth] = cv.add(eyeArea1, dollarEye)

        frame[top_left2[1]: top_left2[1] + eyeHeight, 
              top_left2[0]: top_left2[0] + eyeWidth] = cv.add(eyeArea2, dollarEye)

    cv.imshow("DLIB", frame)

    # Pulsar 'Esc' para salir del bucle
    key = cv.waitKey(1)
    if key == 27: break