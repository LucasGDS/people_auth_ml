from PyQt5.QtWidgets import QApplication, QGraphicsScene #, QLabel
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import qtmliddes #pyuic5 qtmlid.ui -o qtmliddes.py
import cv2
import numpy as np
from subprocess import Popen
import os
# from PyQt5.QtCore import Qt, QTimer

# app = QApplication([])
# label = QLabel('hello world')
# label.resize(800, 600)
# label.setAlignment(Qt.AlignCenter)
# label.show()
# app.exec()

fps = 30
cap = cv2.VideoCapture(0)
path = "./dataset"

class Mlid(QtWidgets.QMainWindow, qtmliddes.Ui_MainWindow):
    

    def __init__(self, parent=None):
        super(Mlid, self).__init__(parent)
        self.trocas = 0
        # self.cadastrarview = self.graphicsView_2
        self.setupUi(self)
        self.set_trocas(0)
        self.pushButton_cad.clicked.connect(self.buttoncad_press)
        self.pushButton_rec.clicked.connect(self.buttonrec_press)
        self.tabWidget.currentChanged.connect(self.onUpdate)
        self.listWidget.currentTextChanged.connect(self.listChange)
        # cadastrarview.
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.onUpdate)  
        self.timer.start(0)#miliseconds

        self.progressbartimer = QtCore.QTimer(self)
        self.progressbartimer.timeout.connect(self.recording)  
    
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
        self.graphicsView_2.setScene(scenecadastrar)


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
        cv2.imwrite(os.path.join(path,img_name), frame)
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
    
    def onUpdate(self):
        tabWidget = self.tabWidget
        tab_ind = tabWidget.currentIndex()
        if tab_ind == 0:
            # print("populatelist")
            self.label_reg_status.setText("")
            self.populate_list()
            self.timer.stop()
            pass
        elif tab_ind == 1:
            # print("register")
            self.timer.start(1000/fps)#miliseconds
            self.draw_reg()
        elif tab_ind == 2:
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


def main():
    if not(os.path.exists('dataset')):
        print("first time run!creating pics folder")
        os.mkdir('dataset')
    if not(os.path.exists('voice')):    # True
        print("creating voices folder")
        os.mkdir('voice')
    app = QApplication(sys.argv)
    form = Mlid()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()