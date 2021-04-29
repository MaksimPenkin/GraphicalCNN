# """
# @author   Maksim Penkin @MaksimPenkin
# @author   Oleg Khokhlov @okhokhlov
# """

import tkinter as tk
from frames.base import BaseGridFrame
from frames.welcome import WelcomeFrame
from frames.setting import SettingFrame
from frames.visual import VisualFrame
from frames.logs import LogsFrame
import argparse
from localisation import localisation


class IORedirector(object):
    """A general class for redirecting I/O to this Text widget."""

    def __init__(self, text_area):
        """Constructor method."""
        self.text_area = text_area


class StdoutRedirector(IORedirector):
    """A class for redirecting stdout to this Text widget."""

    def write(self, str):
        """Method for writing string to text_area.

        :param str: string to write to the text_area
        """
        self.text_area.insert("end", str)
        self.text_area.see("end")

    def flush(self, *args, **kwargs):
        """Method for flushing text_area before exiting application.

        :param *args: arguments
        :param **kwargs: named-arguments
        """
        self.text_area.delete("1.0", tk.END)


class MainFrame(BaseGridFrame):
    """A class to represent a main window."""

    def __init__(self, *args, **kwargs):
        """Constructor method."""
        super().__init__(*args, **kwargs)
        for i in range(3):
            self.rowconfigure(i, weight=1)
        for j in range(3):
            self.columnconfigure(j, weight=1)
        self.createWidgets()

    def createWidgets(self):
        """Method for creating widgets."""
        self.settingButton = tk.Button(self,
                                       text=localisation['settings'][lang],
                                       command=lambda: self.show_frame(SettingFrame))
        self.visualButton = tk.Button(self,
                                      text=localisation['visualization'][lang],
                                      command=lambda: self.show_frame(VisualFrame))
        self.logsButton = tk.Button(self,
                                    text=localisation['logs'][lang],
                                    command=lambda: self.show_frame(LogsFrame))
        self.quitButton = tk.Button(self,
                                    text=localisation['quit'][lang],
                                    command=self.master.quit)
        self.homeButton = tk.Button(self,
                                    text=localisation['home'][lang],
                                    command=lambda: self.show_frame(WelcomeFrame))

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
        """Method for switching between frames.

        :param choice: chosen frame
        """
        self.allFrames[choice].tkraise()

    def redirector(self, inputStr=""):
        """Method for switching stdout to tkinter text box.

        :param inputStr: initial string for tkinter text box
        """
        import sys
        T = self.allFrames[LogsFrame].text_out
        sys.stdout = StdoutRedirector(T)
        T.insert(tk.END, inputStr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-language", type=str, help="set app language", default="eng")
    args = parser.parse_args()

    global lang
    if args.language in ["rus", "eng"]:
        lang = args.language
    else:
        lang = 'eng'

    app = MainFrame()
    app.redirector()
    app.master.title(localisation['main title'][lang])
    app.mainloop()
