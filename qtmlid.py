from PyQt5.QtWidgets import QApplication, QGraphicsScene #, QLabel
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import qtmliddes #pyuic5 qtmlid.ui -o qtmliddes.py
import cv2
import numpy as np
from subprocess import Popen
import os

import facenet
from draw_boxes import *
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
from faced import FaceDetector #for better cpu support
from faced.utils import annotate_image

from keras.models import Model, Sequential, model_from_json
from keras import optimizers

# from PyQt5.QtCore import Qt, QTimer

# app = QApplication([])
# label = QLabel('hello world')
# label.resize(800, 600)
# label.setAlignment(Qt.AlignCenter)
# label.show()
# app.exec()

fps = 30
cap = cv2.VideoCapture(0)
ds_path = "./dataset"
thresh = 0.5

sess = tf.Session()
facenet.load_model("./20170512-110547/20170512-110547.pb")
image_placeholder = tf.get_default_graph().get_tensor_by_name("input:0") 
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0") 
train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

face_detector = FaceDetector()

# load json and create model
json_file = open('modelfunctional.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("modelregister-2020-01-08-19:08.h5")
print("Loaded model from disk")

networkoptimizer = optimizers.SGD(lr=0.01)
loaded_model.compile(loss='binary_crossentropy',
              optimizer=networkoptimizer,
              metrics=['accuracy'])

#Funcao que calcula a distancia euclidiana entre dois vetores
def distance(vector1, vector2):
    return np.sqrt(np.sum((vector1 - vector2)**2))

def get_embedding(img_path): 
    img_size = 160
    img = cv2.imread(img_path)
    #o opencv abre a imagem em BGR, necessario converter para RGB
    if img is None:
        print("Imagem não pode ser aberta.")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #Preparando a imagem de entrada
    resized = cv2.resize(img, (img_size,img_size),interpolation=cv2.INTER_CUBIC)
    reshaped = resized.reshape(-1,img_size, img_size,3)
    #Configurando entrada e execucao do FaceNet
    feed_dict = {image_placeholder: reshaped, train_placeholder: False}
    embedding = sess.run(embeddings , feed_dict=feed_dict) 
    return embedding[0], img

def get_embedding_img(img): 
    img_size = 160
    #Preparando a imagem de entrada
    resized = cv2.resize(img, (img_size,img_size),interpolation=cv2.INTER_CUBIC)
    reshaped = resized.reshape(-1,img_size, img_size,3)
    #Configurando entrada e execucao do FaceNet
    feed_dict = {image_placeholder: reshaped, train_placeholder: False}
    embedding = sess.run(embeddings , feed_dict=feed_dict) 
    return embedding[0]

def who_is_it_crop(img, database):
    min_dist = 1000 
    identity = -1
    #Calculando o embedding do visitante
    visitor = get_embedding_img(img)
    #Calculando a distacia do visitante com os demais funcionarios
    
    for name, employee in database.items():
        dist = distance(visitor, employee)
        if dist < min_dist:
            min_dist = dist 
            identity = name
    #verificando a identidade
    if min_dist > 0.5:
        print("Essa pessoa nao esta cadastrada!")
        return None, img
    else:
        return identity, img

def whose_voice(visitor, database):
    max_sim = 0 
    identity = -1

    xp1 = np.array( [visitor,] )

    X_test_l = xp1.reshape(*xp1.shape, 1)

    #Calculando o embedding do visitante
    #Calculando a distacia do visitante com os demais funcionarios
    for name, employee in database.items():
        xp2 = np.array( [employee,] )
        X_test_r = xp2.reshape(*xp2.shape, 1)

        y_pred = loaded_model.predict([X_test_l, X_test_r])

        print ("y_pred:",y_pred)
        similarity = y_pred[0][1]
        if similarity > max_sim:
            max_sim = similarity 
            identity = name
    #verificando a identidade
    if max_sim < 0.99:
        print("Essa pessoa nao esta cadastrada!")
        return None
    else:
        return identity

class Mlid(QtWidgets.QMainWindow, qtmliddes.Ui_MainWindow):
    

    def __init__(self, parent=None):
        super(Mlid, self).__init__(parent)
        self.trocas = 0
        # self.cadastrarview = self.graphicsView_2
        self.setupUi(self)
        self.set_trocas(0)
        self.pushButton_cad.clicked.connect(self.buttoncad_press)
        self.pushButton_rec.clicked.connect(self.buttonrec_press)
        self.pushButton_id.clicked.connect(self.buttonid_press)
        self.tabWidget.currentChanged.connect(self.onUpdate)
        self.listWidget.currentTextChanged.connect(self.listChange)
        # cadastrarview.
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.onUpdate)  
        self.timer.start(0)#miliseconds

        self.progressbartimer = QtCore.QTimer(self)
        self.progressbartimer.timeout.connect(self.recording)

        self.timer_id = QtCore.QTimer(self)
        self.timer_id.timeout.connect(self.id_routine)

        self.timer_id_voicerecord = QtCore.QTimer(self)
        self.timer_id_voicerecord.timeout.connect(self.id_voice)

        self.database_img = {}
        self.database_snd = {}  
        self.id_state = 0
        self.soundprocessingflag = False
        # self.id_rgb_frame = None
            
    def set_trocas(self,valor):
        self.trocas = valor
        scenecadastrar = QGraphicsScene()
        scenecadastrar.addSimpleText(str(valor))
        # cadastrarview = self.graphicsView_2
        self.graphicsView_2.setScene(scenecadastrar)
    
    def populate_list(self):
        self.listWidget.clear()
        
        paths = os.listdir('dataset')
        for path in paths:
            self.listWidget.addItem(path[:-4])
        pass
    
    def listChange(self,current_text):
        img = cv2.imread("./dataset/%s.png"%current_text)
        if img is None:
            print("Imagem não pode ser aberta.")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320,240),interpolation=cv2.INTER_CUBIC)

        image = QtGui.QImage(img, img.shape[1],\
                            img.shape[0], img.shape[1] * 3,QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap(image)
        scenecadastrar = QGraphicsScene()
        scenecadastrar.addPixmap(pix)
        self.graphicsView.setScene(scenecadastrar)

    def draw_reg(self):
        ret, frame = cap.read()
        if frame.shape[0] == 0:
            pass

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        image = QtGui.QImage(rgb_frame, rgb_frame.shape[1],\
                            rgb_frame.shape[0], rgb_frame.shape[1] * 3,QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap(image)
        scenecadastrar = QGraphicsScene()
        scenecadastrar.addPixmap(pix)

        tab_ind = self.tabWidget.currentIndex()
        if  tab_ind == 1:
            self.graphicsView_2.setScene(scenecadastrar)
        elif tab_ind == 2:
            self.graphicsView_3.setScene(scenecadastrar)


        # # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        # photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(rgb_frame))
        # # Add a PhotoImage to the Canvas
        # canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        pass

    def buttoncad_press(self):
        ret, frame = cap.read()
        self.set_trocas(self.trocas+1)
        user = self.lineEdit.text()
        if user == "":
            user = "default"
        img_name = user+".png"
        cv2.imwrite(os.path.join(ds_path,img_name), frame)
        self.label_reg_status.setText("%s.png salvo!"%user)

    def buttonrec_press(self):
        user = self.lineEdit.text()
        if user == "":
            user = "default"
        # elif user == None:
        #     continue 
        print(user)
        comando = ["arecord", "--duration", "4", "--format", "cd", "./voice/"+user+".wav"]
        gravacao=Popen(comando)
        self.progressbartimer.start(1000)        

    def buttonid_press(self):
        comando = ["arecord", "--duration", "4", "--format", "cd", "./id/idrecord.wav"] #TODO:runtime filename creation 
        gravacao=Popen(comando)
        self.id_state = 1
        self.timer_id_voicerecord.start(5000)
        
        ret, frame = cap.read()
        if frame.shape[0] == 0:
            pass
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        bboxes = face_detector.predict(rgb_frame, thresh)
        if len(bboxes) != 0:
            for x,y,w,h,p in bboxes :
                ####
                x0=int(x-w/2)
                y0=int(y-h/2)
                x0plusw = int(x+w/2)
                y0plush = int(y+h/2)
                ####
                cv2.imshow("window", frame[y0: y0plush,x0 : x0plusw])
                identity, image = who_is_it_crop(rgb_frame[y0: y0plush,x0 : x0plusw], self.database_img)
                print ("Facenet: ",identity)
        pass
    
    def onUpdate(self):
        tabWidget = self.tabWidget
        tab_ind = tabWidget.currentIndex()
        if tab_ind == 0:
            # print("populatelist")
            self.label_reg_status.setText("-")
            self.populate_list()
            self.timer.stop()
            self.timer_id.stop()
            pass
        elif tab_ind == 1:
            # print("register")
            self.timer_id.stop()
            self.timer.start(1000/fps)#miliseconds
            self.draw_reg()
        elif tab_ind == 2:
            self.prepare_id()
            self.id_state = 0
            self.timer.stop()
            self.timer_id.start(1000/fps)
            # print("identificate")
            pass
    
    def recording(self):
        value = self.progressBar.value()
        user = self.lineEdit.text()
        if user == "":
            user = "default"
        if value == 0:
            self.label_reg_status.setText("%s.wav salvo!"%user)
            self.progressBar.setValue(4)
            self.progressBar.setFormat("%vs")
            self.progressbartimer.stop()
        else:
            self.progressBar.setValue(value-1)
            self.progressBar.setFormat("REC")
    
    def prepare_id(self):
        self.database_img = {}
        self.database_snd = {}

        paths = os.listdir(ds_path)
        for path in paths:
            if os.path.exists('./voice/'+path[:-4]+".wav"):
                # self.database_img[path[:-4]], img = get_embedding(os.path.join(ds_path,path))#.addItem(path[:-4])
                img = cv2.imread("./dataset/%s"%path)
                if img is None:
                    print("Imagem ",path, " não pode ser aberta.")
                    return None
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                bboxes = face_detector.predict(rgb_img, thresh)
                if len(bboxes) != 0:
                    for x,y,w,h,p in bboxes :
                        ####
                        x0=int(x-w/2)
                        y0=int(y-h/2)
                        x0plusw = int(x+w/2)
                        y0plush = int(y+h/2)
                        ####
                        # cv2.imshow(path[:-4], img[y0: y0plush,x0 : x0plusw])
                print('loaded '+path)
                self.database_img[path[:-4]] = get_embedding_img(rgb_img[y0: y0plush,x0 : x0plusw])
                audio, sr = librosa.load('./voice/'+path[:-4]+".wav", sr=22000, duration=2, mono=True,offset=1)
                processed = librosa.feature.melspectrogram(y=audio, sr=sr)
                if (processed.shape == (128, 86)):
                    self.database_snd[path[:-4]] = processed
        print (self.database_snd)
        

    def id_routine(self):
        if self.id_state == 0:
            self.draw_reg()
            pass
        elif self.id_state == 1: # recording voice, changed in buttonid_press():
            # self.timer_id.stop()
            self.soundprocessingflag == True
            while self.soundprocessingflag == True:    
                if self.id_state == 2: # processing

                    pass
                elif self.id_state == 3: # results

                    pass
    
    def id_voice(self):
        self.timer_id_voicerecord.stop()
        audio, sr = librosa.load('./id/idrecord.wav', sr=22000, duration=2, mono=True)
        processed = librosa.feature.melspectrogram(y=audio, sr=sr)

        identity = whose_voice(processed, self.database_snd)
        print("Voice: ",identity)
        pass


def main():
    if not(os.path.exists('dataset')):
        print("first time run!creating pics folder")
        os.mkdir('dataset')
    if not(os.path.exists('voice')):    # True
        print("creating voices folder")
        os.mkdir('voice')
    if not(os.path.exists('id')):    # True
        print("creating id folder")
        os.mkdir('id')
    app = QApplication(sys.argv)
    form = Mlid()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()