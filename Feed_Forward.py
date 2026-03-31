# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:25:36 2023

@author: rahul
"""

"""Module for FeedForward model"""
import torch
import torch.nn as nn


class FeedForward(torch.nn.Module):
    def __init__(self,input_dim,output_dim):
        super(FeedForward, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = nn.Linear(self.input_dim, 20)  # 2 input nodes, 4 hidden nodes
        # self.layer2 = nn.Linear(20,10)  # 4 hidden nodes, 4 hidden nodes
        # self.layer3 = nn.Linear(256,128)  # 4 hidden nodes, 4 hidden nodes
        self.layer4 = nn.Linear(20,self.output_dim)  # 4 hidden nodes, 1 output node
        self.activation1 = nn.ReLU()  # Activation function
        self.activation2 = nn.ReLU()
        self.activation3 = nn.Tanh()
        self.activation4 = nn.Tanh()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.activation1(out)
        # out = self.layer2(out)
        # out = self.activation2(out)
        # out = self.layer3(out)
        # out = self.activation3(out)
        out = self.layer4(out)
        out = self.activation4(out)
        return out
