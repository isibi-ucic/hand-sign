import cv2

from cvzone.HandTrackingModule import HandDetector

import numpy as np
import math
import time


cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)


offset = 20 #ruang untuk kosong

imgSize = 300 #memberikan size untuk menyamakan ukuran untuk img crop


folder = "Data/C"
counter = 0

while True:

    success, img = cap.read()

    hands, img = detector.findHands(img)
    

    if hands: 

        hand = hands[0] #jumlah tangan 1

        x,y,w,h = hand['bbox'] #box pembatas


        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255 #biar semua crop output sama keluaran ukurannya


        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset] #matriks, y tambah tinggi maka lebar awal akan menjadi x dan lebar akhir menjadi x tambah w


        imgCropShape = imgCrop.shape

     

        aspecRatio = h/w


        if aspecRatio >1: #menyamakan agar vidio  crop dapat memilki ukuran yang sama dan selalu menyesuaikan dengan ukuran kotak putih

            k = imgSize/h

            wCal = math.ceil(k*w)

            imgResize = cv2.resize(imgCrop, (wCal, imgSize))


            imgResizeShape = imgResize.shape 


            wGap = math.ceil((300-wCal)/2)


            imgWhite[:, wGap:wCal+wGap] = imgResize #masukkan image crop pada img white (persegi)


        else: #agar tidak ada kedipan

            k = imgSize/w

            hCal = math.ceil(k*h)

            imgResize = cv2.resize(imgCrop, (imgSize, hCal))


            imgResizeShape = imgResize.shape 


            hGap = math.ceil((300-hCal)/2)


            imgWhite[hGap:hCal+hGap, :] = imgResize


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)


    cv2.imshow("Image", img)

    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite) 
        print(counter)

