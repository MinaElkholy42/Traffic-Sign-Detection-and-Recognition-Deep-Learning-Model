import cv2
import numpy as np
import skimage as sk
import helper_function as hp
from skimage import morphology, filters
import tkinter as tk
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from tkinter import *
from gtts import gTTS
from pygame import mixer
import os
# from PIL import Image, ImageTk

IMG_SIZE = 32


def histogram_equalize(image):
    kernel = sk.morphology.disk(30)
    img_local = sk.filters.rank.equalize(image, selem=kernel)
    return img_local


def predict(test_img, filename, master):
    list1 = master.winfo_children()
    for i in range(1, len(list1)):
        list1[i].destroy()
    test_img_data = histogram_equalize(test_img)
    test_img_data = cv2.resize(test_img_data, (IMG_SIZE, IMG_SIZE)) / 255.0
    x_test_3d = np.expand_dims(test_img_data, axis=0)
    x_test_4d = np.expand_dims(x_test_3d, axis=3)
    # Model
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model("model.h5")
    print("Loaded model from disk")
    print('--------------------------------')

    prediction = model.predict(x_test_4d)
    calss_name = np.argmax(prediction)
    label = tk.Label(master)
    img = PhotoImage(file=filename)
    if img.width() < 200:
        label.img = img.zoom(4, 4)
    elif 200 <= img.width() <= 300:
        label.img = img.zoom(3, 3)
    else:
        label.img = img

    # get Class Name Of the Image
    label_name = hp.Get_sign_description(calss_name)
    # Audio Sound
    if not(os.path.exists("class_name_speech" + str(calss_name)+".mp3")):
        language = 'en'  # Name of Used Language
        myobj = gTTS(text=label_name, lang=language, slow=False)
        myobj.save("class_name_speech" + str(calss_name)+".mp3")
    # Run the Mp3 File
    mixer.init()
    mixer.music.load("class_name_speech" + str(calss_name)+".mp3")
    mixer.music.play()

    #######################################
    label.config(image=label.img)
    # label.grid(row=3, column=3)
    label.pack()
    layers_label = tk.Label(
        master,
        text=label_name,
        fg='black',
        bg="#A9A9A9",
        font='verdana 30 bold')
    layers_label.pack(side="bottom", ipady=50, fill="x")
    print("Image successfully done")
