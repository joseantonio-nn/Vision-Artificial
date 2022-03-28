import cv2 as cv
import numpy as np 
import sys
import matplotlib.pyplot as plt
from glob import glob


def getKeypointsAndDescriptors(readImgs):
    kpsAndDes = []
    for img in readImgs:
        kps, des = sift.detectAndCompute(img, None)
        kpsAndDes.append((img, kps, des))
    return kpsAndDes


def findCandidate(kpsAndDes, kad):
    
    bestRatio, i = 0, 0
    bestGood, bestImg, bestKad = None, None, None
    
    for kadi in kpsAndDes:    
        ratio,good = bestMatch(kad[2], kadi[2], kad[1])
        
        if ratio > bestRatio:
            bestRatio = ratio
            bestGood = good
            bestKad = kadi
            i += 1
    
    return (bestKad, bestGood, i)


def bestMatch(des1, des2, kps1):
    good = []
    matches = matcher.knnMatch(des2, des1, k=2)

    for m in matches:
        if len(m) == 2:
            best,second = m
            if best.distance < 0.75 * second.distance:
                good.append(best)
    #if len(good) < 4: return 4, None
    ratio = len(good) / len(kps1)
    return (ratio,good)


def desp(desp):
	dx,dy = desp
	return np.array([[1,0,dx],
			         [0,1,dy],
			         [0,0,1]])


def combineImages(img1, img2, H):
	return np.maximum(cv.warpPerspective(img2, desp((50,150)) @ np.eye(3),(1800,600)),
                      cv.warpPerspective(img1, desp((50,150)) @ H,(1800,600)))
        

def readImages(inputImgs):
    readImgs = []
    for i in inputImgs:
        img = cv.imread(i)
        readImgs.append(img)

    return readImgs


sift = cv.AKAZE_create()
matcher = cv.BFMatcher()	

def main():

    inputImgs = glob(sys.argv[1])
    
    readImgs = readImages(inputImgs)
    newImg = None 

    kpsAndDes = getKeypointsAndDescriptors(readImgs)

    while len(kpsAndDes) != 1:

        kad = kpsAndDes[0]
        (bestKad, good, imgIndex) = findCandidate(kpsAndDes[1:], kad)

        src_pts = np.array([kad[1][m.trainIdx].pt for m in good]).astype(np.float32).reshape(-1,2)
        dst_pts = np.array([bestKad[1][m.queryIdx].pt for m in good]).astype(np.float32).reshape(-1,2)
        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 3)

        newImg = combineImages(kad[0], bestKad[0], H)

        kpsAndDes.pop(imgIndex)
        kpsAndDes = kpsAndDes[1:]
        k,d = sift.detectAndCompute(newImg, None)
        kpsAndDes.insert(0,(newImg, k, d))
        

    while(True):
        key = cv.waitKey(1) & 0xFF
        if key == 27: break
        cv.imshow('PANO', kpsAndDes[0][0])

main()


