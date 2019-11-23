# import Tkinter as tk #python2
import tkinter as tk #python3
# from PIL import Image, ImageTk
import PIL.Image, PIL.ImageTk
import mlidopencv
import tkiaux
import cv2

width = 1200
height = 600
capture = False
cap = cv2.VideoCapture(0)

def register_press():
    #mlidopencv.register()
    global capture,cap
    capture = True
    canvas.grid(row=0) # canvas.pack()

def list_press():
    canvas.grid_remove() # canvas.pack_forget()
    capture = False


master=tk.Tk() #where m is the name of the main window object
master.geometry(str(width)+"x"+str(height))
master.title('MLID')

frameL = tk.Frame(master)
frameL.grid(row=0,column=0, sticky="n")

frameM = tk.Frame(master)
frameM.grid(row=0,column=1, sticky="n")
##
canvas = tk.Canvas(frameM, width = width, height = height)
# register = tk.Button(master, text='Cadastrar', width=25, command=mlidopencv.register)
register = tk.Button(frameL, text='Cadastrar', width=25, command=register_press)
verify = tk.Button(frameL, text='Listar', width=25, command=list_press)
identify = tk.Button(frameL, text='Identificar', width=25, command=master.destroy)
close = tk.Button(frameL, text='Sair', width=25, command=master.destroy)



register.grid(row=0)
verify.grid(row=1)
identify.grid(row=2)
close.grid(row=3)
# register.pack()
# verify.pack()
# identify.pack()
# close.pack()

##
# master.mainloop()

# while True: # master.mainloop() equivalent
#     tk.update_idletasks()
    
#     tk.update()

while True: # master.mainloop() equivalent
    master.update_idletasks()
    if capture == True:
        ret, frame = cap.read()
        if frame.shape[0] == 0:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(rgb_frame))
        # Add a PhotoImage to the Canvas
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)


    master.update()