import torch

import SimpleITK as sitk # pip install SimpleITK

import numpy as np
import time

## Padd the array of the patient image to fit the fixed input 
#  size of 100 x 100 x 100

def pad_3D_array(arr,input_dimensions,value):
        
    offset =np.subtract(input_dimensions, arr.shape)
    even = offset%2
    offset = (offset/2).astype(int)
    pad=[(offset[0],offset[0]+even[0]) ,(offset[1],offset[1]+even[1]),\
         (offset[2],offset[2]+even[2])]
    new_arr = np.pad(arr,pad,'constant',constant_values=value)

    return new_arr

## Load the patient image that should be in format .mhd, with a voxel size
## of  5 x 5 x 5 mm

def load_patient_mhd (filename,input_dimensions):
    

    img_patient = sitk.ReadImage(filename)
    arr_patient = sitk.GetArrayFromImage(img_patient)
    arr_patient = pad_3D_array(arr_patient,input_dimensions,-1050)
    
    arr_patient = np.reshape (arr_patient,(1,1,input_dimensions[0],
                              input_dimensions[1],
                              input_dimensions[2])).astype("float32")
    return arr_patient

## Convert the output to a ITK image

def output_to_itk(output_scattermap):
    scatter_image = sitk.GetImageFromArray(output_scattermap)
    scatter_image.SetSpacing((20,20,20))
    return scatter_image



        
        
        
        
