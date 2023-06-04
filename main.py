from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from tkinter import Label, Pack, BOTTOM, StringVar
import tkinter as tk
import cv2
import re
on = True
model = YOLO('v2.pt')
gui = tk.Tk()
gui.geometry("900x900")
gui.title("SpiceSee")
cam = cv2.VideoCapture(1)
maturity = StringVar()
health = StringVar()
remarks = StringVar()
tk.Label(gui, text="SpiceSee", font="Verdana 25 bold").pack(fill=tk.X)
tk.Label(gui, text="Maturity:", font="Arial 20 bold", padx=10, pady=55 ).pack(fill=tk.X)
tk.Label(gui, text="", font="Arial 15", textvariable=maturity).pack(fill=tk.X)
tk.Label(gui, text="Health", font="Arial 20 bold", padx=10, pady=55).pack(fill=tk.X)
tk.Label(gui, text="", font="Arial 15", textvariable=health).pack(fill=tk.X)
tk.Label(gui, text="Remarks", font="Arial 20 bold", padx=10, pady=55).pack(fill=tk.X)
tk.Label(gui, text="", font="Arial 15", textvariable=remarks).pack(fill=tk.X)
tk.Button(gui, height=3, width=50, text="STOP SCANNING", command=lambda y=False: quit(y)).pack()

classes = {
    9: 'No object detected',
    0: 'Disease',
    1: 'Physically damaged',
    2: 'Ripe',
    3: 'Decaying',
    4: 'Unripe'
}

health_desc = {
    9: 'No description available',
    0: 'The chili is affected with disease',
    1: 'The chili has physical damage',
    2: 'No presence of damage, decay and diseases',
    3: 'The chili is decaying',
    4: 'No presence of damage, decay and diseases',
}

def quit(y):
    gui.destroy()
    global on
    on = y

RejectArr = [0,1,3]
PassedArr = [2,4]
while on:
    ret, frame = cam.read()
    results = model(frame, conf=0.25, verbose=False, max_det = 1)
    annotated_frame = results[0].plot()
    text = str(results[0].boxes.cls)
    num = int(re.search("\d+" , text+"9")[0])
    cv2.imshow("Camera", annotated_frame)
    maturity.set(classes[num])
    health.set(health_desc[num])
    if num in RejectArr:
        remarks.set('Reject')
    elif num in PassedArr:
        remarks.set('Passed')
    else:
        remarks.set('No remarks available')
    gui.update()
    if (on == False):
        break


