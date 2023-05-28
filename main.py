from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from tkinter import Label, Pack, BOTTOM, StringVar
import tkinter as tk
import cv2
import re
import numpy as np
on = True
model = YOLO('chili.pt')
gui = tk.Tk()
gui.geometry("900x900")
gui.title("Chili Maturity and Health Inspection System")
cam = cv2.VideoCapture(0)
maturity = StringVar()
health = StringVar()
remarks = StringVar()
tk.Label(gui, text="Chili Maturity and Health Inspection System", font="Verdana 25 bold").pack(fill=tk.X)
tk.Label(gui, text="Maturity:", font="Arial 20 bold", padx=10, pady=55 ).pack(fill=tk.X)
tk.Label(gui, text="", font="Arial 15", textvariable=maturity).pack(fill=tk.X)
tk.Label(gui, text="Health", font="Arial 20 bold", padx=10, pady=55).pack(fill=tk.X)
tk.Label(gui, text="", font="Arial 15", textvariable=health).pack(fill=tk.X)
tk.Label(gui, text="Remarks", font="Arial 20 bold", padx=10, pady=55).pack(fill=tk.X)
tk.Label(gui, text="", font="Arial 15", textvariable=remarks).pack(fill=tk.X)
tk.Button(gui, height=3, width=50, text="STOP SCANNING", command=lambda y=False: quit(y)).pack()

classes = {
    9: 'No object detected',
    0: 'Half-Ripe',
    1: 'Ripe',
    2: 'Unripe',
    3: ''

}

health_desc = {
    9: 'No description available',
    0: 'No presence of damage, decay and diseases',
    1: 'No presence of damage, decay and diseases',
    2: 'No presence of damage, decay and diseases',
    4: 'The chili has physical damage',
    5: 'The chili is affected with disease',
    6: 'The chili is decaying'
}
remarks = {

}
def quit(y):
    gui.destroy()
    global on
    on = y

while on:
    ret, frame = cam.read()
    results = model(frame, conf=0.3, verbose=False, max_det = 5)
    annotated_frame = results[0].plot()
    text = str(results[0].boxes.cls)
    num = int(re.search("\d+" , text+"9")[0])
    cv2.imshow("Camera", annotated_frame)
    maturity.set(classes[num])
    health.set(health_desc[num])
    gui.update()
    if (on == False):
        break


