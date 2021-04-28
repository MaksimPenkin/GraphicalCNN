# """
# @author   Maksim Penkin @MaksimPenkin
# @author   Oleg Khokhlov @okhokhlov
# """

import tkinter as tk
from frames.base import BaseGridFrame


class LogsFrame(BaseGridFrame):
    """A class to represent a logs window."""

    def __init__(self, *args, **kwargs):
        """Constructor method."""
        super().__init__(*args, **kwargs)
        for i in range(1):
            self.rowconfigure(i, weight=1)
        for j in range(1):
            self.columnconfigure(j, weight=1)
        self.createWidgets()

    def createWidgets(self):
        """Method for creating widgets."""
        self.text_out = tk.Text(self)
        self.text_out.grid(row=0, column=0, sticky='NEWS')
