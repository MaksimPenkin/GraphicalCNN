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
        self.settingButton = tk.Button(self, text='Settings', command=lambda : self.show_frame(SettingFrame))
        self.visualButton = tk.Button(self, text='Visualization', command=lambda : self.show_frame(VisualFrame))
        self.logsButton = tk.Button(self, text='Logs', command=lambda : self.show_frame(LogsFrame))
        self.quitButton = tk.Button(self, text='Quit', command=self.master.quit)
        self.homeButton = tk.Button(self, text='Home', command=lambda : self.show_frame(WelcomeFrame))

        containerFrame = tk.Frame(self)

        self.settingButton.grid(row=0, column=0, sticky='NEWS')
        self.visualButton.grid(row=0, column=1, sticky='NEWS')
        self.logsButton.grid(row=0, column=2, sticky='NEWS')
        containerFrame.grid(row=1, column=0, columnspan=3, sticky='NEWS')
        self.quitButton.grid(row=2, column=2, sticky='NEWS')
        self.homeButton.grid(row=2, column=0, sticky='NEWS')

        self.allFrames = {} 
        for F in (WelcomeFrame, SettingFrame, VisualFrame, LogsFrame):
            frame = F(containerFrame)
            self.allFrames[F] = frame
            frame.grid(row=0, column=0, sticky='NEWS')
        self.show_frame(WelcomeFrame)

    def show_frame(self, choice):
        self.allFrames[choice].tkraise()


if __name__ == '__main__':
    app = MainFrame()
    app.master.title('Graphical CNN App')
    app.mainloop()


