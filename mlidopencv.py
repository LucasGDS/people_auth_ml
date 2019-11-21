import os
import numpy as np
from subprocess import Popen

from PIL import Image
from PIL import ImageTk
# import Tkinter as tki #python2
# import tkinter as tki
# import threading
import datetime
# import imutils
import cv2
import argparse

from easygui import *

recording = None

def identify():
    print("bla")
    
def record(user):
    global recording
    comando = ["arecord", "--duration", "4", "--format", "cd", str(user)+".wav"]
    gravacao=Popen(comando)
    
def register():
    cap = cv2.VideoCapture(0)

    img_counter = 0
    font = cv2.LINE_AA

    while cap.isOpened():
    #     now = time.time()
    #     Capture frame-by-frame
        ret, frame = cap.read()
        shown_frame = frame.copy()
        if frame.shape[0] == 0:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    #     if thresh:
    #         bboxes = face_detector.predict(rgb_frame, thresh)
    #     else:
    #         bboxes = face_detector.predict(rgb_frame)

    #     if len(bboxes) != 0:
    #         results = []
    #  
    #     cv2.imshow('window', rgb_frame)       for x,y,w,h,p in bboxes :
    #             ####
    #             x0=int(x-w/2)
    #             y0=int(y-h/2)
    #             x0plusw = int(x+w/2)
    #             y0plush = int(y+h/2)
    #             ####
    #             identity, image = who_is_it_crop(rgb_frame[y0: y0plush,x0 : x0plusw], database)
    #             results.append((x,y,w,h,p,identity))
    #         ann_frame = boxing(frame,results)
    #     else:
    #         ann_frame = frame

        #ann_frame = annotate_image(frame, bboxes)
#         cv2.putText(shown_frame, " press spacebar to register ",(255,255),font, 0.6, (255, 255, 255), 1)
#         cv2.putText(shown_frame, " press C for identifier",(255,280),font, 0.6, (255, 255, 255), 1)
        cv2.putText(shown_frame, " press spacebar to register ",(200,400),font, 0.6, (0, 255, 255), 1)
        cv2.imshow('window',shown_frame)
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            print("q hit, closing...")
            break
        elif k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k & 0xFF == ord('c'):
            identify()
        elif k%256 == 32:
            # SPACE pressed
            print("save?")
            user = enterbox("Enter name")
            if user == None:
                continue 
            elif user == "":
                user = "default"
            shown_frame = frame.copy()
            save_msg = "Register user "+user+"?"
            cv2.putText(shown_frame, save_msg,(200,400),font, 0.6, (0, 255, 255), 1)
            cv2.putText(shown_frame, "Press spacebar to start recording voice and register",(100,425),font, 0.6, (0, 255, 255), 1)
    #         command = ["zenity", "--title", "Gimme some text!", "--entry","--text","Enter your text here"]
    #         user = Popen(command)

#             msgbox("Press OK to see your picture. Press spacebar to save it and start recording voice")    
            while True:
                cv2.imshow('window',shown_frame)
#                 cv2.putText(shown_frame, " press spacebar to register ",(200,400),font, 0.6, (0, 255, 255), 1)
                k = cv2.waitKey(1)
                if k%256 == 32:
                    record(user)
                    path = "./dataset"
    #                 img_name = "opencv_frame_{}.png".format(img_counter)
                    img_name = user+"_0.png"
                    cv2.imwrite(os.path.join(path,img_name), frame)
                    print("{} written!".format(img_name))
    #                 img_counter += 1
                    break
                elif k%256 == 27:
                    break


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def mainmenu():
    font = cv2.LINE_AA
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    
#     img_1 = np.zeros([512,512,1],dtype=np.uint8)
#     img_1.fill(255)
#     # or img[:] = 255
#     cv2.imshow('Single Channel Window', img_1)
#     print("image shape: ", img_1.shape)

#     img_3 = np.zeros([512,512,3],dtype=np.uint8)
    img_3 = np.zeros([720,1280,3],dtype=np.uint8)
    img_3.fill(255)
    # or img[:] = 255
    cv2.putText(img_3, " Welcome to MLID ",(0,30),font, 0.6, (255, 0, 0), 1)
    cv2.putText(img_3, " Press spacebar to register ",(0,255),font, 0.6, (255, 0, 0), 1)
    cv2.putText(img_3, " Press C for identifier",(0,280),font, 0.6, (255, 0, 0), 1)
    cv2.imshow('window', img_3)
    print("image shape: ", img_3.shape)
    
    while True:
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
                print("q hit, closing...")
                break
        elif k & 0xFF == ord('c'):
            identify()
            break
        elif k%256 == 32:
            register()
            break
    
#     cv2.waitKey(0)
    cv2.destroyAllWindows()

mainmenu()