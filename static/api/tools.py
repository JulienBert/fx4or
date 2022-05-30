 
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import torch


##########################################################################################################################"

def getFakeMIPScatteringMap():
    #Get image
    vol  = sitk.ReadImage('scatter_map_patient_5_orbital_0_angular_0_kvp_100_x_0_y_0_z_0.mhd', imageIO='MetaImageIO')
    aVol = sitk.GetArrayFromImage(vol)
    aVol /= aVol.max()

    aMIP = np.amax(aVol, axis=1)
    aMIP /= aMIP.max()
    aMIP *= 255

    aMIP = aMIP.astype("uint8")
    sizey, sizex = aMIP.shape
    aMIP = aMIP.flatten()

     #3 layers (rgb) - format uint8
    rgbImg = []
    rgbImg.append( aMIP.tolist() )
    rgbImg.append( aMIP.tolist() )
    rgbImg.append( aMIP.tolist() )

    #return rgbImg, sizex, sizey, aMIP.dtype.str

    # # plot isodose
    # levelsCS1 = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 1e-2, 1]
    # levelsCS2 = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 1e-2, 1]

    # colors = [(0.6,1.0,0.0, 0.0), (0.8,1.0,0.0, 0.0), (1.0,1.0,0.0), (1.0,0.8,0.0), (1.0,0.6,0.0), (1.0,0.4,0.0), (1.0,0.2,0.0), (1.0,0.0,1.0), (1.0,0.0,1.0)]

    # ny, nx = aMIP.shape
    # fig, axs = plt.subplots(figsize=(6,6))
    # CS = plt.contourf(range(nx), range(ny), aMIP, levelsCS1, colors=colors)
    # CS2 = plt.contour(range(nx), range(ny), aMIP, levelsCS2, colors='k')
    # plt.axis('off')
    # plt.savefig('static/img/fakeMIPScattering.png', bbox_inches='tight')
    # plt.show()
##########################################################################################################################"

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

##########################################################################################################################"
