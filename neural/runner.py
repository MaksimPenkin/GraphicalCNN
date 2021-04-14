""" 
 @author   Maksim Penkin @MaksimPenkin
 @author   Oleg Khokhlov @okhokhlov
"""

import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

from neural.models import Generator
from neural.dataset.paired_dataset import SegmDataset
from neural.dataset.transforms import HorizontalFlipRandom, VerticalFlipRandom, Rot90Random, ScaleRotate, ToTensor
import utility.utils as utils


class Runner:
    def __init__(self, batch_size=8, num_epochs=100, arch='ResBlock', num_filters=8, num_blocks=4, loss='L2', opt='Adam', experiment_name='test_1'):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.arch = arch 
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.loss = loss
        self.opt = opt 
        self.experiment_name = experiment_name

        assert self.batch_size > 0
        assert self.num_epochs > 0
        assert self.arch in ['ResBlock', 'ConvBlock']
        assert self.num_filters > 0
        assert self.num_blocks > 0
        assert self.loss in ['L2', 'CrossEntropy']
        assert self.opt in ['Adam', 'SGD']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def assert_train_pathes(self, force=False):
        self.path_data = './data'
        self.path_experiment = os.path.join('./checkpoints', self.experiment_name)
        utils.create_folder(self.path_experiment, force=True)
    
    def prepare_trainval_dataset(self):
        transform = transforms.Compose([
            HorizontalFlipRandom(),
            VerticalFlipRandom(),
            Rot90Random(),
            ScaleRotate(s_min=1., s_max=1., a_min=-5., a_max=5.),
            ToTensor()
        ])

        # Train dataset
        trainset = SegmDataset(os.path.join(self.path_data, 'train.csv'), 
                                root_dir=os.path.join(self.path_data, 'train'),
                                transform=transform)
        self.trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=0)

        # Validation dataset
        valset = SegmDataset(os.path.join(self.path_data, 'val.csv'), 
                                root_dir=os.path.join(self.path_data, 'val'),
                                transform=ToTensor())
        self.valloader = DataLoader(valset, batch_size=8, shuffle=False, num_workers=0)
    
    def get_loss(self):
        return torch.nn.BCEWithLogitsLoss()

    def get_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=0.001)
    
    def train(self):
        self.assert_train_pathes(force=False)
        self.prepare_trainval_dataset()

        model = Generator(self.num_filters, num_blocks=self.num_blocks, batch_norm=False, arch=self.arch).to(self.device)
        model.train()

        criterion = self.get_loss()
        optimizer = self.get_optimizer(model)

        print_freq = 10
        print('[*] Training is starting...')
        for epoch in range(self.num_epochs):
            start = time.time()
            running_loss = 0.0
            epoch_loss = 0.0
            for i, data in enumerate(self.trainloader):
                img, mask = data['image'].to(self.device), data['mask'].to(self.device)
                
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(img)
                loss = criterion(outputs, mask)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                epoch_loss += loss.item()

                # print statistics
                if (i % print_freq == 0) and (i != 0) and (i != len(self.trainloader) - 1):
                    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ': Epoch {0:3} [{1:4}/{2:4}] '
                        'loss: {3:.8f}'.format(epoch+1, i+1, len(self.trainloader), running_loss / print_freq)
                        )
                    running_loss = 0.0
                elif (i == len(self.trainloader) - 1):
                    pass
                    #writer.add_scalar("Loss/train", epoch_loss / len(trainloader), epoch)
            torch.save(model.state_dict(), os.path.join(self.path_experiment, "model_e{}.pth".format(epoch+1)))

            print('[*] Validating...')
            with torch.no_grad():
                val_epoch_loss = 0.0
                for i, data in enumerate(self.valloader):
                    img, mask = data['image'].to(self.device), data['mask'].to(self.device)
                    out_logit = model(img)
                    out_prob = torch.nn.functional.softmax(out_logit, dim=1)
                    loss = criterion(out_logit, mask)
                    print('[*] Validate Epoch {0:3} '
                        'loss: {1:.4f}'.format(epoch+1, loss.item())
                        )
                    val_epoch_loss += loss.item()
                #writer.add_scalar("Loss/val", val_epoch_loss / len(valloader), epoch)

            print('[*] ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ': Time for epoch {0} is {1:.2f}'.format(epoch+1, time.time()-start))
                    
        #writer.close()
        print('Finished Training')


