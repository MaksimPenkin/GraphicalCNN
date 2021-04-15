""" 
 @author   Maksim Penkin @MaksimPenkin
 @author   Oleg Khokhlov @okhokhlov
"""

import tkinter as tk
from frames.base import BaseGridFrame
from widgets.optionmenu import OptionMenuWidget
from widgets.entry import EntryWidget


class SettingFrame(BaseGridFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        for i in range(3):
            self.rowconfigure(i, weight=1)
        for j in range(3):
            self.columnconfigure(j, weight=1)
        self.createWidgets()

    def createWidgets(self):
        architectureList = ('ConvBlock', 'ResBlock')
        lossList = ('BCE', 'CrossEntropy')
        optimizeList = ('Adam', 'SGD')

        self.architectureOM = OptionMenuWidget(self, architectureList, 'Architecture')
        self.lossOM = OptionMenuWidget(self, lossList, 'Loss')
        self.optimizeOM = OptionMenuWidget(self, optimizeList, 'Optimizer')

        self.batch_size_box = EntryWidget(self, 'Batch size')
        self.epoch_number_box = EntryWidget(self, 'Epoch number')

        self.architectureOM.grid(row=0, column=0)
        self.lossOM.grid(row=1, column=0)
        self.optimizeOM.grid(row=2, column=0)

        self.batch_size_box.grid(row=0, column=2)
        self.epoch_number_box.grid(row=1, column=2)


