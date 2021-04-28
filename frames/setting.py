# """
# @author   Maksim Penkin @MaksimPenkin
# @author   Oleg Khokhlov @okhokhlov
# """

import tkinter as tk
from tkinter import messagebox
from frames.base import BaseGridFrame
from widgets.optionmenu import OptionMenuWidget
from widgets.entry import EntryWidget
from neural.runner import Runner
import threading


class SettingFrame(BaseGridFrame):
    """A class to represent a settings window."""

    def __init__(self, *args, **kwargs):
        """Constructor method."""
        super().__init__(*args, **kwargs)
        for i in range(4):
            self.rowconfigure(i, weight=1)
        for j in range(3):
            self.columnconfigure(j, weight=1)
        self.createWidgets()

    def createWidgets(self):
        """Method for creating widgets."""
        architectureList = ('ConvBlock', 'ResBlock')
        lossList = ('BCE', 'L2')
        optimizeList = ('Adam', 'SGD')

        self.architectureOM = OptionMenuWidget(self, architectureList, 'Architecture')
        self.lossOM = OptionMenuWidget(self, lossList, 'Loss')
        self.optimizeOM = OptionMenuWidget(self, optimizeList, 'Optimizer')

        self.batch_size_box = EntryWidget(self, 'Batch size')
        self.epoch_number_box = EntryWidget(self, 'Epoch number')
        self.experiment_name_box = EntryWidget(self, 'Experiment name')

        self.trainButton = tk.Button(self, text='Train!',
                                     command=lambda: threading.Thread(target=self.train).start(),
                                     bg='green3',
                                     activebackground='green4')
        self.stopButton = tk.Button(self, text='Stop!',
                                    command=self.stop,
                                    bg='red3',
                                    activebackground='red4')

        self.architectureOM.grid(row=0, column=0)
        self.lossOM.grid(row=1, column=0)
        self.optimizeOM.grid(row=2, column=0)

        self.batch_size_box.grid(row=0, column=2)
        self.epoch_number_box.grid(row=1, column=2)
        self.experiment_name_box.grid(row=2, column=2)

        self.trainButton.grid(row=3, column=0)
        self.stopButton.grid(row=3, column=2)

    def train(self):
        """Method for initiate training loop."""
        self.stop_event = threading.Event()
        try:
            batch_size_curr = int(self.batch_size_box.get()) if self.batch_size_box.get() != '' else 8
            num_epoch_curr = int(self.epoch_number_box.get()) if self.epoch_number_box.get() != '' else 100
            arch_curr = str(self.architectureOM.get())
            loss_curr = str(self.lossOM.get())
            opt_curr = str(self.optimizeOM.get())
            if self.experiment_name_box.get() != '':
                experiment_name_curr = str(self.experiment_name_box.get())
            else:
                experiment_name_curr = 'test_1'

            r = Runner(batch_size=batch_size_curr,
                       num_epochs=num_epoch_curr,
                       arch=arch_curr,
                       loss=loss_curr,
                       opt=opt_curr,
                       experiment_name=experiment_name_curr)

            r.train(self.stop_event)
        except Exception as e:
            messagebox.showerror("E:", "Message: " + str(e))

    def stop(self):
        """Method for breaking training loop."""
        self.stop_event.set()
