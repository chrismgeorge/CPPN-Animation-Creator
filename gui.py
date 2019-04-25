from tkinter import *
import argparse
import os

import numpy as np
from model import CPPN

import pickle

from PIL import Image, ImageTk


def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("--ml", type=str, default=None) # model name
    parser.add_argument("--model_dir", type=str, default='many_models/models')
    parser.add_argument("--of", type=str, default=None) # outfile

    # Parse arguments
    args = parser.parse_args()

    return args


# Here, we are creating our class, Window, and inheriting from the Frame
# class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
class Window(Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, x_dim=1280, y_dim=720, z_dim=5, scale=16, neurons_per_layer=6, number_of_layers=8,
         color_channels=1, number_of_stills=5, interpolations_per_image=24, file_name='./p.png',
         model_name=None, outfile=None, model_dir=None, master=None):

        if (model_name == None or outfile == None or model_dir == None):
            print('Supply a model name and outfile!')
            exit(0)

        checkpoint_path = os.path.join(model_dir, model_name)
        outfile_name = checkpoint_path + 'meta_data'
        with open (outfile_name, 'rb') as fp: # 'outfile' can be renamed
            data = pickle.load(fp)

        x_dim = data[0]
        y_dim = data[1]
        z_dim = data[2]
        scale = data[3]
        neurons_per_layer = data[4]
        number_of_layers = data[5]
        color_channels = data[6]
        test = data[7]

        self.cppn = CPPN(x_dim, y_dim, z_dim, scale, neurons_per_layer, number_of_layers,
                color_channels, interpolations_per_image, test=test)
        self.cppn.neural_net(True)

        self.cppn.load_model(model_name=model_name, model_dir=model_dir)

        self.outfile = outfile

        self.z_vector = [0]*self.cppn.z_dim

        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master)

        #reference to the master widget, which is the tk window
        self.master = master

        self.varHolders = []
        self.saved_data = []
        self.scaler = DoubleVar()

        self.save_image_name = 'image_name.png'

        #with that, we want to then run init_window, which doesn't yet exist
        self.init_window()

    #Creation of init_window
    def init_window(self):

        # changing the title of our master widget
        self.master.title("GUI")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        for i in range(self.cppn.z_dim):
            m = DoubleVar()
            w = Scale(self.master, from_=-2, to=2, variable=m, resolution=.01,
                      command=self.updateValue, orient=HORIZONTAL)
            w.pack(side=LEFT)
            self.varHolders.append(m)

        w = Scale(self.master, from_=1, to=48, variable=self.scaler, resolution=.1,
                      command=self.updateValue, orient=HORIZONTAL)
        w.pack(side=LEFT)

        # creating a button instance
        saveDataButton = Button(self.master, text="save data",command=self.saveData)
        saveDataButton.pack()

        pickleButton = Button(self.master, text="pickle data",command=self.pickleData)
        pickleButton.pack()

    def pickleData(self):
        print('Data pickled!')
        with open(self.outfile, 'wb') as f: # can change 'outfile'
            pickle.dump(self.saved_data, f)

    def saveData(self):
        print('saved ', len(self.saved_data))
        zs = []
        for value in self.varHolders:
            zs.append(value.get())
        zs.append(self.scaler.get())
        self.saved_data.append(zs)

    def updateValue(self, event):
        for index, value in enumerate(self.varHolders):
            self.z_vector[index] = value.get()
        z = np.array(self.z_vector)
        self.cppn.scale = self.scaler.get()

        # if save set to false, just sets cppn.curImage to image
        self.cppn.save_png(z, self.save_image_name, save=False)
        self.showImg()

    def showImg(self):
        load = self.cppn.curImage
        resized = load.resize((800, 600),Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)

        # labels can be text or images
        img = Label(self, image=render)
        img.image = render
        img.place(x=0, y=0)


if __name__ == '__main__':
    # Parse the arguments
    args = parseArguments()

    root = Tk()

    root.geometry("2000x1500")

    #creation of an instance
    app = Window(model_name=args.ml, outfile=args.of,
                 model_dir=args.model_dir, master=root)

    #mainloop
    root.mainloop()
