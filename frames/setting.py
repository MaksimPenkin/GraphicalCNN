""" 
 @author   Maksim Penkin @MaksimPenkin
 @author   Oleg Khokhlov @okhokhlov
"""

import tkinter as tk
from frames.base import BaseGridFrame


class SettingFrame(BaseGridFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        for i in range(3):
            self.rowconfigure(i, weight=1)
        for j in range(3):
            self.columnconfigure(j, weight=1)
        self.createWidgets()

    def createWidgets(self):
        self.label = tk.Label(self, text='TODO Setting Frame')
        self.label.grid(row=0, column=0, sticky="NEWS")

        optionList = ('train', 'plane', 'boat')
        self.v = tk.StringVar()
        self.v.set(optionList[0])
        self.om = tk.OptionMenu(self, self.v, *optionList)

        self.om.grid(row=1,column=0)


