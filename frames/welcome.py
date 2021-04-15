"""
@author   Maksim Penkin @MaksimPenkin
@author   Oleg Khokhlov @okhokhlov
"""

import tkinter as tk
from PIL import ImageTk, Image
from frames.base import BaseFrame


class WelcomeFrame(BaseFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.width, self.height = 196, 196
        self.image = Image.open('./imgs/welcome.png')
        self.createWidgets()

    def createWidgets(self):
        self.canvas = tk.Canvas(self, width=200, height=200)
        self.canvas.pack(fill='both', expand=True)

        self.photo = ImageTk.PhotoImage(self.image.resize((self.width, self.height), Image.ANTIALIAS))
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

        self.bind('<Configure>', self.resizer)

    def resizer(self, e):
        self.width, self.height = e.width, e.height
        self.photo = ImageTk.PhotoImage(self.image.resize((self.width, self.height), Image.ANTIALIAS))
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
