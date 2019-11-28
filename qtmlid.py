from PyQt5.QtWidgets import QApplication, QGraphicsScene #, QLabel
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import qtmliddes
import cv2
import numpy as np
# from PyQt5.QtCore import Qt, QTimer

# app = QApplication([])
# label = QLabel('hello world')
# label.resize(800, 600)
# label.setAlignment(Qt.AlignCenter)
# label.show()
# app.exec()

cap = cv2.VideoCapture(0)

class Mlid(QtWidgets.QMainWindow, qtmliddes.Ui_MainWindow):
    

    def __init__(self, parent=None):
        super(Mlid, self).__init__(parent)
        self.trocas = 0
        # self.cadastrarview = self.graphicsView_2
        self.setupUi(self)
        self.set_trocas(0)
        self.pushButton_cad.clicked.connect(self.buttoncad_press)
        # cadastrarview.
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.onUpdate)
        self.timer.start(0)#miliseconds  
    
    def set_trocas(self,valor):
        self.trocas = valor
        scenecadastrar = QGraphicsScene()
        scenecadastrar.addSimpleText(str(valor))
        # cadastrarview = self.graphicsView_2
        self.graphicsView_2.setScene(scenecadastrar)

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
        self.set_trocas(self.trocas+1)
    
    def onUpdate(self):
        tabWidget = self.tabWidget
        tab_ind = tabWidget.currentIndex()
        if tab_ind == 0:
            print("populatelist")
            
        elif tab_ind == 1:
            print("register")
            self.draw_reg()
        elif tab_ind == 2:
            print("identificate")
        print(self.trocas)


def main():
    app = QApplication(sys.argv)
    form = Mlid()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()