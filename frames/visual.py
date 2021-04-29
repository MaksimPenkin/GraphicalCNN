# """
# @author   Maksim Penkin @MaksimPenkin
# @author   Oleg Khokhlov @okhokhlov
# """

import os
import tkinter as tk
from frames.base import BaseGridFrame
from localisation import localisation


class VisualFrame(BaseGridFrame):
    """A class to represent a visual window."""

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

        lang = os.environ.get('graph_cnn_app_lang')
        self.label = tk.Label(self, text=localisation['сoming soon'][lang])
        self.label.grid(row=0, column=0, sticky='NEWS')
