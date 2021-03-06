# """
# @author   Maksim Penkin @MaksimPenkin
# @author   Oleg Khokhlov @okhokhlov
# """

import tkinter as tk
from frames.base import BaseFrame


class OptionMenuWidget(BaseFrame):
    """A class to represent an option menu widget."""

    def __init__(self, master, options, label='OptionMenuName', *args, **kwargs):
        """Constructor method."""
        super().__init__(master, **kwargs)
        for i in range(1):
            self.rowconfigure(i, weight=1)
        for j in range(2):
            self.columnconfigure(j, weight=1)
        self.option_list = options
        self.label_text = label
        self.var = tk.StringVar()
        self.createWidgets()

    def createWidgets(self):
        """Method for creating widgets."""
        self.label = tk.Label(self, text=self.label_text)
        self.label.grid(row=0, column=0)

        self.var.set(self.option_list[0])
        self.om = tk.OptionMenu(self, self.var, *self.option_list)
        self.om.grid(row=0, column=1)

    def get(self):
        """Method for getting option-menu value."""
        return self.var.get()
