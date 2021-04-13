""" 
 @author   Maksim Penkin @MaksimPenkin
 @author   Oleg Khokhlov @okhokhlov
"""

import tkinter as tk


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
        self.grid(sticky='NEWS')

    def createWidgets(self):
        pass


