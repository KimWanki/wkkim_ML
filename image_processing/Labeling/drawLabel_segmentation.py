#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import cv2
import os

## create main program window
CUR_DIR = 'C:/Users/USER/Downloads/Keras-Class-Activation-Map-master/image/total/Adeno'

def to_matrix(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]

class App:
    def __init__(self, master):
        self.xlist = []
        self.ylist = []
        self.master = master
        self.master.title('Python Labeling Program')
        self.master.minsize(width=640, height=480)
        self.leftview = Frame()
        self.rightview = Frame()
        self.drawcanvas = Canvas(self.leftview, bg="white", height=480, width=640)
        self.controlbox = Frame(self.leftview, bg="green",  width=200)
        self.listbox = Listbox(self.controlbox, height=20, width=60)

        for idx, line in enumerate(os.listdir(CUR_DIR)):
            if '.jpg' in line:
                self.listbox.insert(idx, str(line))
            elif '.bmp' in line:
                self.listbox.insert(idx, str(line))
            elif '.tif' in line:
                self.listbox.insert(idx, str(line))


        self.saveButton = Button(self.controlbox, height=20, width=10, text='Save')
        self.labelcanvas = Canvas(self.rightview, bd=1, bg="green", height=480, width=640)
        self.labelcanvas.old_coords = None
        self.datacanvas = Canvas(self.rightview, bd=1, bg="green", width=640)
        self.datacanvas.old_coords = None

        self.leftview.pack(side='left', anchor='nw', expand=False, fill='both')
        self.drawcanvas.pack(expand=False, fill='none', anchor='nw')
        self.listbox.pack(side='left')
        self.saveButton.pack(side='right')
        self.controlbox.pack(expand=False, fill='none')
        self.rightview.pack(side='right', anchor='nw', expand=True, fill='none')
        self.labelcanvas.pack(expand=True, anchor='nw', fill='none')
        self.datacanvas.pack(expand=True, fill='none')

        # Bind KeyPress KeyFunction
        self.drawcanvas.bind('<ButtonPress-1>', self.draw_line)
        self.drawcanvas.bind('<ButtonRelease-1>', self.draw_line)
        self.drawcanvas.bind("<B1-Motion>", self.draw_line)
        self.drawcanvas.bind("<Configure>", self.update_canvas)
        self.listbox.bind("<Double-1>", self.get_filename)
        self.saveButton.bind('<ButtonPress-1>', self.save_label)

        self.cur_filename = os.listdir(CUR_DIR)[0]
        self.mask = None
        self.load_image(path=None)
        self.drawed = False
        self.check_listbox()

    def check_listbox(self):
        idx = 0
        while True:
            fname = self.listbox.get(idx)
            if fname == '':
                break
            if os.path.exists(os.path.join(CUR_DIR, fname.replace('tif','bmp').replace('bmp','jpg').replace('jpg', 'npy'))):
                self.listbox.itemconfig(idx, {'bg': 'red'})
            idx += 1
        print(idx)
    def save_label(self, event):
        fname_label = self.cur_filename.replace('tif','bmp').replace('bmp','jpg').replace('jpg', 'npy')
        from skimage.transform import resize

        print("변환 전 (Shape)", self.mask.shape)
        print("변환 전 size   ", self.mask.size)

        save_mask = resize(self.mask, (self.temphei,self.tempwid), anti_aliasing=True)
        # print(save_mask)
        save_mask = save_mask % 2 == 1
        # print(save_mask)
        np.save(os.path.join(CUR_DIR, fname_label), save_mask)
        # self.mask = resize(save_mask, (480,640),anti_aliasing=True)

        print("변환 후 (Shape)",save_mask.shape)
        print("변환 후 size   ",save_mask.size)
        print("saved")
        self.check_listbox()

    def get_filename(self, event):
        self.cur_filename = self.listbox.selection_get()
        self.load_image(path=None)

        self.imwid = self.image.width()
        self.imhei = self.image.height()

        if self.cur_filename.replace('tif','bmp').replace('bmp','jpg').replace('jpg', 'npy') in os.listdir(CUR_DIR):
            self.mask = np.load(os.path.join(CUR_DIR, self.cur_filename.replace('tif','bmp').replace('bmp','jpg').replace('jpg', 'npy')), allow_pickle=True)
            # from matplotlib import pyplot as plt
            # plt.imshow(self.mask, cmap='gray')
            # plt.show()
        else:
            self.mask = np.zeros((self.imhei, self.imwid))

        self.mask_show = ImageTk.PhotoImage(image=Image.fromarray(self.mask))

        self.drawed = False
        self.update_canvas('asd')
        self.draw_line('ButtonRelease')

    def draw_line(self, event):
        self.xlist.append(event.x)
        self.xlist.append(event.y)
        self.drawcanvas.create_oval(event.x, event.y, event.x + 1, event.y + 1, fill="black", outline="black",
                                    tags='final_circle_objects')
        if event == 'ButtonRelease' or str(event.type) == 'ButtonRelease':
            self.drawcanvas.create_polygon(self.xlist)
            point = to_matrix(self.xlist, 2)

            if not self.drawed:
                self.mask = np.zeros((self.imhei, self.imwid))
                # self.mask = np.zeros((self.imhei, self.imwid), np.uint8)

            mask_temp = np.zeros((self.imhei, self.imwid))
            mask_temp = cv2.fillPoly(mask_temp, [np.array(point)], (255,255,255))
            self.drawed = True

            self.mask = np.logical_or(self.mask, mask_temp)
            self.mask_show = ImageTk.PhotoImage(image=Image.fromarray(self.mask))

            del self.xlist[:]
            del point[:]
            self.update_canvas('asd')

    def reset_coords(self, event):
        self.drawcanvas.old_coords = None

    def update_canvas(self, event):
        self.master.update()
        self.datacanvas.delete('all')
        if not self.drawed:
            self.drawcanvas.delete('all')

        self.imwid = self.image.width()
        self.imhei = self.image.height()

        self.canwid_data = self.datacanvas.winfo_width()
        self.canhei_data = self.datacanvas.winfo_height()

        self.canwid_draw = self.drawcanvas.winfo_width()
        self.canhei_draw = self.drawcanvas.winfo_height()

        self.canwid_label = self.labelcanvas.winfo_width()
        self.canhei_label = self.labelcanvas.winfo_height()

        self.locx_data = round((self.canwid_data - self.imwid) / 2.0)
        self.locy_data = round((self.canhei_data - self.imhei) / 2.0)

        self.locx_draw = round((self.canwid_draw - self.imwid) / 2.0)
        self.locy_draw = round((self.canhei_draw - self.imhei) / 2.0)

        self.locx_label = round((self.canwid_label - self.imwid) / 2.0)
        self.locy_label = round((self.canhei_label - self.imhei) / 2.0)

        self.datacanvas.create_image(self.locx_data, self.locy_data, image=self.image, anchor='nw')
        if not self.drawed:
            self.drawcanvas.create_image(self.locx_draw, self.locy_draw, image=self.image, anchor = 'nw' )
        self.labelcanvas.create_image(self.locx_label, self.locy_label, image=self.mask_show, anchor='nw')


        # self.datacanvas.create_image(self.locx_data, self.locy_data, image=self.image, anchor='nw')
        # if not self.drawed:
        #     self.drawcanvas.create_image(self.locx_draw, self.locy_draw, image=self.image, anchor='nw')
        # self.labelcanvas.create_image(self.locx_label, self.locy_label, image=self.mask_show, anchor='nw')

    def load_image(self, path):
        # self.imagesprite = Image.open(os.path.join(CUR_DIR, self.cur_filename)).convert('L')
        self.imagesprite = Image.open(os.path.join(CUR_DIR, self.cur_filename))
        self.tempwid,self.temphei = self.imagesprite.size
        print (self.tempwid,self.temphei)
        imagesprite = self.imagesprite.resize((640, 480), Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(imagesprite)

        # self.image = ImageTk.PhotoImage(Image.open(os.path.join(CUR_DIR, self.cur_filename)))


window = Tk()

## create window container
app = App(window)

window.mainloop()