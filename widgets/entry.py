""" 
 @author   Maksim Penkin @MaksimPenkin
 @author   Oleg Khokhlov @okhokhlov
"""

import tkinter as tk
from frames.base import BaseFrame


class EntryWidget(BaseFrame):
    def __init__(self, master, label='EntryName', *args, **kwargs):
        super().__init__(master, **kwargs)        
        for i in range(1):
            self.rowconfigure(i, weight=1)
        for j in range(2):
            self.columnconfigure(j, weight=1)
        self.label_text = label
        self.createWidgets()
    
    def createWidgets(self):
        self.label = tk.Label(self, text=self.label_text)
        self.label.grid(row=0, column=0)

        self.text = tk.Entry(self)
        self.text.grid(row=0, column=1)

    def get(self):
        return self.text.get()


