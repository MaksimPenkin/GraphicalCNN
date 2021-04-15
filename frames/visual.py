# """
# @author   Maksim Penkin @MaksimPenkin
# @author   Oleg Khokhlov @okhokhlov
# """

import tkinter as tk
from frames.base import BaseGridFrame


class VisualFrame(BaseGridFrame):
    """A class to represent a visual window."""

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
        self.label = tk.Label(self, text='TODO Visual Frame')
        self.label.grid(row=0, column=0, sticky='NEWS')
