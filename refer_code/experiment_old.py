"""
File: experiment.py
Author: 李泽宇
Date: 
Description: Time series classification experiment
"""

import torch.distributed
from model.model_AC import TSProject
from model.loss import DetectionLoss
from utils.tools import Config, collate_fn
import torch
import torch.nn as nn
import data.DataProvider as DP
import numpy as np
import random
from torch.utils.data import random_split, DataLoader, RandomSampler, BatchSampler, SequentialSampler
from utils.log import Logger
import os
import time
from tqdm import tqdm
from typing import Iterable
import pandas as pd
from datetime import datetime
from torch.autograd.profiler import profile
from torch.autograd import Variable
from torch.utils import tensorboard






default_type = torch.float64
torch.set_default_dtype(default_type)



def printf(msg):
    tqdm.write('{} {}'.format(datetime.now(), str(msg)))


class Experiment():
    def __init__(self, args):
        
        os.system('clear')
        printf('this process(PID: {}) would use GPU {}'.format(os.getpid(), args.cuda))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
        self.Config = Config(args.config)
        self.ExpConfig = self.Config.config.pop('ExpSetting')
        self.logger = Logger(self.ExpConfig['log_store_path'])
        self.lossdf = pd.DataFrame(columns=['epoch', 'loss', 'type'])
        self.epoch = 0
        self.minloss = 99999

        
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
        self.Model = nn.parallel.DataParallel(TSProject(self.Config.config)).cuda()
        self.logger('Model has been created successfully')
        self.Dataset = DP.DatasetProvider.get_dataset(self.Config['DataProvider'])
        self.logger('Dataset has been created successfully')
        self.Loss = DetectionLoss(**self.Config['Loss'])
        self.Optimizer = torch.optim.AdamW(params=self.Model.parameters(), **self.Config['Optim'])

        if self.Config['Lr_config']:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.Optimizer, **self.Config['Lr_config'])

        if self.ExpConfig['RESUME']:
            path_checkpoint = input('Please input checkpoint file')
            checkpoint = torch.load(path_checkpoint)
            self.Model.load_state_dict(checkpoint['model'])
            self.Optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr'])
            self.epoch = checkpoint['epoch']

        self.writer = tensorboard.SummaryWriter(self.ExpConfig['log_store_path'])
        self.logger('Experiment is ready')

    def saveModel(self, filename:str):
        model = self.Model.state_dict()
        path = os.path.join(self.ExpConfig['exp_path'], filename)
        torch.save(model, path)
        self.logger('Model state dict has been saved in {}'.format(path))


    def checkpoint(self):
        return {
            'model': self.Model.state_dict(),
            'optimizer': self.Optimizer.state_dict(),
            'lr': self.lr_scheduler.state_dict(),
            'epoch': self.epoch
        }
    
    def save(self, filename:str):
        cp = self.checkpoint()
        path = os.path.join(self.ExpConfig['checkpoint_path'], filename)
        try:
            if not os.path.isdir(self.ExpConfig['checkpoint_path']):
                os.mkdir(self.ExpConfig['checkpoint_path'])
            torch.save(cp, path)
            self.logger('Checkpoint at epoch {}, weight file has been stored at file {}, current min loss is {}'.format(self.epoch, path, self.minloss))
            return True
        except Exception as e:
            self.logger(e, 4)
            return False
    
    @torch.no_grad()
    def evaluate(self, data_loader, epoch):
        self.Model.eval()
        self.Loss.eval()

        weight_dict = self.Loss.weight_dict
        loss = torch.tensor(0, dtype=torch.float64).cuda()
        loss_dict = {k:torch.tensor(0, requires_grad=False, dtype=torch.float64).cuda() for k,v in weight_dict.items()}

        for samples, targets in tqdm(data_loader, desc='Evaluating epoch {} is processing'.format(epoch), leave=False):

            samples = samples.cuda()
            targets = [{k: v.cuda() for k,v in t.items()} for t in targets]

            outputs = self.Model(samples)


            losses_dict = self.Loss(outputs, targets)
            
            for k in losses_dict.keys():
                if k in weight_dict:
                    loss_dict[k] = loss_dict[k] + losses_dict[k] * weight_dict[k]
                else:
                    pass

        for k in loss_dict.keys():
            loss_dict[k] = loss_dict[k] / len(data_loader)
            loss = loss + loss_dict[k]
        
        if loss < self.minloss:
            self.minloss = loss
            self.saveModel('best.pth')


        printf('----------------------------- current avg evaluating loss is {:.4f}, and its detail is {}'.format(loss, loss_dict))
        self.logger('Evaluation Loss of epoch {} is {:.4f}, and its detail is {}'.format(epoch, loss, loss_dict))
        self.lossdf.loc[len(self.lossdf)] = [epoch, loss.cpu(), 'val']
    
    def train_one_epoch(self, data_loader: Iterable, epoch: int, max_norm:float=0):
        self.Model.train()
        self.Loss.train()
        self.logger('Training epoch {} start'.format(epoch))

        weight_dict = self.Loss.weight_dict

        loss = torch.tensor(0, requires_grad=False, dtype=torch.float64).cuda()
        loss_dict = {k: torch.tensor(0, requires_grad=False, dtype=torch.float64).cuda() for k,v in weight_dict.items()}
        

        for samples, targets in tqdm(data_loader, desc='Training epoch {} is processing'.format(epoch), colour='#33FF66', leave=False):
            samples = samples.cuda()
            targets = [{k: v.cuda() for k,v in t.items()} for t in targets]
            # printf(samples)
            # printf('-------------------------------------------')
            # printf(targets)
            # printf('data transfer cost memory: {}'.format(torch.cuda.memory_allocated() / 1024 / 1024 / 1024))
            outputs = self.Model(samples)
            # printf('Inference cost memory: {}'.format(torch.cuda.memory_allocated() / 1024 / 1024 / 1024))

            losses_dict = self.Loss(outputs, targets)
            # printf('Loss cost memory: {}'.format(torch.cuda.memory_allocated() / 1024 / 1024 / 1024))

            losses = torch.zeros_like(losses_dict['loss_ce'])
            for k in losses_dict.keys():
                if k in weight_dict:
                    losses = losses + losses_dict[k] * weight_dict[k]
                    loss_dict[k] = loss_dict[k] + (losses_dict[k] * weight_dict[k]).detach()
                else:
                    pass

            with torch.autograd.set_detect_anomaly(False):
                self.Optimizer.zero_grad()
                printf(losses)
                losses.backward()
                # if max_norm > 0:
                #     torch.nn.utils.clip_grad_norm_(self.Model.parameters(), max_norm)
                self.Optimizer.step()

            # for name, parms in self.Model.named_parameters():
            #     printf('---->name: {} -->grad_requires: {} --> grad_values: {}'.format(name, parms.requires_grad, parms))

        for k in loss_dict.keys():
            loss_dict[k] = loss_dict[k] / len(data_loader)
            loss = loss + loss_dict[k]

        printf('----------------------------- Current epoch avg training loss is {:.4f}, and its detail is {}'.format(loss, loss_dict))
        self.logger('Training loss of epoch {} is {:.4f}, and its detail is {}'.format(epoch, loss, loss_dict))
        self.lossdf.loc[len(self.lossdf)] = [epoch, loss.cpu(), 'tra']
            
        
    def expReady(self):
        lossdf_path = os.path.join(self.ExpConfig['exp_path'], 'loss.csv')
        if self.ExpConfig['seed']:
            torch.manual_seed(self.ExpConfig['seed'])
            np.random.seed(self.ExpConfig['seed'])
            random.seed(self.ExpConfig['seed'])

        self.logger('Spliting the dataset')
        train_size = int(self.ExpConfig['train_ratio'] * len(self.Dataset))
        val_size = len(self.Dataset) - train_size

        train_dataset, val_dataset = random_split(self.Dataset, [train_size, val_size])
        
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

        train_batchSampler = BatchSampler(train_sampler, self.Config['DataLoader']['batch_size'], True)

        train_DataLoader = DataLoader(train_dataset, batch_sampler=train_batchSampler, collate_fn=collate_fn)
        val_DataLoader = DataLoader(val_dataset, sampler=val_sampler, drop_last=False, **self.Config['DataLoader'], collate_fn=collate_fn)
        return train_DataLoader, val_DataLoader

    def start(self):
        self.logger('---------------- Experiment start ----------------')
        train_DataLoader, val_DataLoader = self.expReady()
        start_time = time.time()
        printf('------------------------------ Start Training ------------------------------')
        for epoch in tqdm(range(self.epoch+1, self.ExpConfig['max_num_epoch']+1), desc='Training is Processing', colour='#FFFF00'):

            self.epoch = epoch

            self.train_one_epoch(train_DataLoader, epoch, max_norm=1)

            self.evaluate(val_DataLoader, epoch)

            if self.lr_scheduler:
                self.lr_scheduler.step()

            if ((epoch) % self.ExpConfig['check_point']) == 0 or (epoch) % self.Config['Lr_config']['step_size'] == 0:
                self.save('Checkpoint{}.pth'.format(epoch))

            self.lossdf.to_csv(lossdf_path)

        end_time = time.time()
        cost_time = end_time - start_time
        printf('-------------- Training End -- time cost: {}s --------------'.format(cost_time))
        self.saveModel('final.pth')
        self.save('final_state_dict.pth')
        self.lossdf.to_csv(lossdf_path)
        printf('------------------------- model has been saved ----------------------------')

