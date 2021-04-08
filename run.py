""" 
 @author   Maksim Penkin @MaksimPenkin
 @author   Oleg Khokhlov @okhokhlov
"""

import numpy as np
import tkinter as tk
from PIL import ImageTk, Image


class BaseFrame(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

    def createWidgets(self):
        pass


class BaseGridFrame(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)        
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        for i in range(1):
            self.rowconfigure(i, weight=1)
        for j in range(1):
            self.columnconfigure(j, weight=1)
        self.grid(sticky="NEWS")

    def createWidgets(self):
        pass


class WelcomeFrame(BaseFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.width, self.height = 196, 196
        self.image = Image.open("./imgs/welcome.png")
        self.createWidgets()
    
    def createWidgets(self):
        self.canvas = tk.Canvas(self, width=200, height=200)
        self.canvas.pack(fill="both", expand=True)
 
        self.photo = ImageTk.PhotoImage(self.image.resize((self.width, self.height), Image.ANTIALIAS))
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        self.bind("<Configure>", self.resizer)

    def resizer(self, e):
        self.width, self.height = e.width, e.height
        self.photo = ImageTk.PhotoImage(self.image.resize((self.width, self.height), Image.ANTIALIAS))
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")


class MainFrame(BaseGridFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        for i in range(3):
            self.rowconfigure(i, weight=1)
        for j in range(3):
            self.columnconfigure(j, weight=1)
        self.createWidgets()

    def createWidgets(self):
        self.settingButton = tk.Button(self, text='Settings', command=None)
        self.visualButton = tk.Button(self, text='Visualization', command=None)
        self.logsButton = tk.Button(self, text='Logs', command=None)
        self.quitButton = tk.Button(self, text='Quit', command=self.master.quit)

        self.workingFrame = WelcomeFrame(self)

        self.settingButton.grid(row=0, column=0, sticky="NEWS")
        self.visualButton.grid(row=0, column=1, sticky="NEWS")
        self.logsButton.grid(row=0, column=2, sticky="NEWS")
        self.workingFrame.grid(row=1, column=0, columnspan=3, sticky="NEWS")
        self.quitButton.grid(row=2, column=2, sticky="NEWS")


if __name__ == "__main__":
    app = MainFrame()
    app.master.title('Graphical CNN App')
    app.mainloop()


