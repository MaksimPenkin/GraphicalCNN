# """
# @author   Maksim Penkin @MaksimPenkin
# @author   Oleg Khokhlov @okhokhlov
# """

import tkinter as tk


class BaseFrame(tk.Frame):
    """A class to represent a base frame."""

    def __init__(self, master=None, **kwargs):
        """Constructor method."""
        super().__init__(master, **kwargs)

    def createWidgets(self):
        """Method for creating widgets."""
        pass


class BaseGridFrame(tk.Frame):
    """A class to represent a base frame with grid."""

    def __init__(self, master=None, **kwargs):
        """Constructor method."""
        super().__init__(master, **kwargs)
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        for i in range(1):
            self.rowconfigure(i, weight=1)
        for j in range(1):
            self.columnconfigure(j, weight=1)
        self.grid(sticky='NEWS')

    def createWidgets(self):
        """Method for creating widgets."""
        pass
