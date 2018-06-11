import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import math

class ReNet(nn.Module):
    def __init__(self, num_input, num_units, patch_size=(1,1)):
        super(ReNet, self).__init__()
        self.patch_size_height = int(patch_size[0])
        self.patch_size_width = int(patch_size[1])

        self.rnn_horz = nn.GRU(input_size=num_input*self.patch_size_height*self.patch_size_width , hidden_size=num_units, num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=True)
        self.rnn_vert = nn.GRU(input_size=2*num_units, hidden_size=num_units, num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=True)


    def rnn_forward(self, x, h_or_v):
    	assert h_or_v in ['horz', 'vert']

    def forward(self,x):
    

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()

		self.vgg = models.vgg16(pretrained=True)	
        
	def forward(self,x):
		x = self.vgg(x)

		return x

class ReSeg(nn.Module):
    def __init__(self, num_classes):
        super(ReSeg, self).__init__()
        self.num_classes = num_classes		
        self.cnn = CNN()
        self.renet = ReNet()

    def forward(self,x):
        input = x
        x = self.cnn(x)
        x = self.renet

        return out
