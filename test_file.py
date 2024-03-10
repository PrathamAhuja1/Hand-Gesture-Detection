import cv2
import time
import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
count = 0
index_pre = 0

model = tf.keras.models.load_model(r"C:\Users\Computer_PA24\Downloads\hand-gesture-recognition-code\Model\hand_gesture_model_weights.h5")
offset = 20
imgSize = 300
labels = ["Backward", "Flip", "Forward", "Land", "Left", "Right", "Up"]

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        try:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            if h > w:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                WidthGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, WidthGap:wCal + WidthGap] = imgResize
                resize = tf.image.resize(imgWhite, (64, 64))
                plt.imshow(resize.numpy().astype(int))

            if w > h:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                heightGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[heightGap:hCal + heightGap, :] = imgResize
                resize = tf.image.resize(imgWhite, (64, 64))
                plt.imshow(resize.numpy().astype(int))

        except Exception as e:
            continue
        prediction = model.predict(np.expand_dims(resize, 0))
        index = np.argmax(prediction)
        text_size = cv2.getTextSize(labels[index], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.rectangle(img, (x - 30, y - 35 - text_size[1]), (x + w // 2 + text_size[0] - 15, y - 25), (252, 3, 198), -1)
        print(prediction, index)
        # if(index == index_pre or count>5):
        #     count=0
        #     index_pre=index
        cv2.putText(img, labels[index], (x - 20, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)
        # else:
        #     count+=1
        #     time.sleep(0.05)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImgWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break