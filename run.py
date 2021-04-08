""" 
 @author   Maksim Penkin @MaksimPenkin
 @author   Oleg Khokhlov @okhokhlov
"""

import tkinter as tk
from frames.base import BaseGridFrame
from frames.welcome import WelcomeFrame
from frames.setting import SettingFrame
from frames.visual import VisualFrame
from frames.logs import LogsFrame


class MainFrame(BaseGridFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        for i in range(3):
            self.rowconfigure(i, weight=1)
        for j in range(3):
            self.columnconfigure(j, weight=1)
        self.createWidgets()

    def createWidgets(self):
        self.settingButton = tk.Button(self, text='Settings', command=self.get_settingFrame)
        self.visualButton = tk.Button(self, text='Visualization', command=self.get_visualFrame)
        self.logsButton = tk.Button(self, text='Logs', command=self.get_logsFrame)
        self.quitButton = tk.Button(self, text='Quit', command=self.master.quit)

        self.workingFrame = WelcomeFrame(self)

        self.settingButton.grid(row=0, column=0, sticky="NEWS")
        self.visualButton.grid(row=0, column=1, sticky="NEWS")
        self.logsButton.grid(row=0, column=2, sticky="NEWS")
        self.workingFrame.grid(row=1, column=0, columnspan=3, sticky="NEWS")
        self.quitButton.grid(row=2, column=2, sticky="NEWS")

    def update_workingFrame(self):
        self.workingFrame.grid_forget()
        self.workingFrame.grid(row=1, column=0, columnspan=3, sticky="NEWS")
    
    def get_settingFrame(self):
        self.workingFrame = SettingFrame(self)
        self.update_workingFrame()
    
    def get_visualFrame(self):
        self.workingFrame = VisualFrame(self)
        self.update_workingFrame()

    def get_logsFrame(self):
        self.workingFrame = LogsFrame(self)
        self.update_workingFrame()


if __name__ == "__main__":
    app = MainFrame()
    app.master.title('Graphical CNN App')
    app.mainloop()


