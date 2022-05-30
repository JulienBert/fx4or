#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 10:51:01 2021

@author: mvilla
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class downsampling_block(nn.Module):
   
    def __init__(self,conv_layers=2,input_channels=3,init_filters=4,level=0,current_conv=0):
        super().__init__()
        self.convolutions = nn.ModuleList()
        for i in range(conv_layers):
            
            if level == 0 and i ==0 :
                input_channels=input_channels
            else:
                input_channels = init_filters*(2**(level))
            output = init_filters*(2**(level))*(i+1)
                                            
    
            self.convolutions.append(nn.Conv3d(input_channels,output,3,padding=1))
               
    def forward(self,x):
        for conv_layer in self.convolutions :
            x = conv_layer(x)
            x = F.relu(x)
            # x.size()
        return x
    
    
class upsampling_layer(nn.Module):
   
    def __init__(self,level=3,init_filters=4):
        super().__init__()
        dilation = 1 # common value
        
        if level==3:
            dilation = 2
        factor =int(init_filters)
        self.transpose = nn.ConvTranspose3d(in_channels=factor*2**(level+1), out_channels = factor * 2**(level+1), 
                                                    kernel_size=2,stride =2,dilation=dilation)
        
        self.conv1 = nn.Conv3d(in_channels = factor * (2**(level+1)),
                                                   out_channels = factor * 2**(level), kernel_size=3,padding=1)
        
        self.conv2 = nn.Conv3d(in_channels= factor * 2**(level), 
                               out_channels = factor * 2**(level), kernel_size=3,padding=1)
        
    def forward(self,x1):
        x = self.transpose(x1)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.relu(x)
        x = F.relu(x)
        return x
    
class intermediate_layers(nn.Module):
    def __init__(self,aux):
        super().__init__()   
           
        self.conv1 = nn.Conv3d(aux+1,aux,3,padding=1)
        self.conv2 = nn.Conv3d(aux,aux*2,3,padding=1)

    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x


class fluence_net_v2(nn.Module):
    
    def __init__(self,layers=4,
                  init_filters=4,
                  image_shape = [100,100,100],
                  n_params = 2,
                  ct_shape = (100,100,100),inference_mode = False
                  ) :

        super().__init__()
        self.layers = layers
        self.init_filters = init_filters
        self.image_shape = np.array( image_shape,dtype = int)

        self.n_params = n_params
        self.ct_shape = np.array(ct_shape)
        self.create_model()
        self.inference_mode = inference_mode

        
    def create_model(self):
        self.down_layers= nn.ModuleList([downsampling_block(input_channels=1,init_filters = self.init_filters,level=i) for i in range(self.layers-1)])
        
        self.linear_1 =  nn.Linear(in_features = self.n_params ,
                                      out_features=256)
        
        self.linear_1.bias.data.fill_(0.1)
        self.linear_2 =  nn.Linear(in_features = 256,
                                   out_features=(self.ct_shape/8).astype(int).prod())
        
    
        aux = self.init_filters*2**(self.layers-1)

        self.intermediate_layers =intermediate_layers(aux)
        self.up_layers= nn.ModuleList([upsampling_layer(level=i,init_filters=self.init_filters) for i in range(self.layers-1,0,-1)])
        self.output_layer = nn.Conv3d(self.init_filters*2, 1, 3,padding=1)
    def forward(self,x,param):
        input_=x
        for layer in self.down_layers:
            x = layer(x)
            x = F.max_pool3d(x,2,2) 
        
        a = self.linear_1(param)
   
        a = self.linear_2(a)
        shape = [a.size()[0],1,x.shape[2],x.shape[3],x.shape[4]]
        a = torch.reshape(a,shape)
        x = torch.cat((x,a),1)
        x = self.intermediate_layers(x)
        
        for layer in self.up_layers:
            x = layer(x)
        x = self.output_layer(x)
        
        if self.inference_mode == True:
             x =  torch.pow(10,x/100) ### This is included only in inference !!!!.
        x[...,42:59,40:61,38:63]=0
        
     
        return x
    
##############################################################################################    


    
#######################################################################################################    
    
if __name__=="__main__":
    import matplotlib.pyplot as plt
    from PIL import Image
    x = torch.randn(2,1,100,100,100)
    
    param = torch.randn(2,2)
    model = fluence_net_v2(init_filters=4) 
    output = model(x,param).detach().numpy()
    
    aMIP = np.amax(output, axis=0)
    aMIP /= aMIP.max()
    aMIP *= 255
    
    aMIP = aMIP.astype("uint8")
    b = aMIP[0].tolist()
    b = np.array(b)
    
    doseSlice = b[20, :, :]
    
    plt.imshow(doseSlice)
    plt.show()
