
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

def getFakeMIPScatteringMap():
    # Get image
    vol  = sitk.ReadImage('test/scatter_map_patient_72_orbital_0_angular_0_kvp_100_x_0_y_0_z_0.mhd', imageIO='MetaImageIO')
    aVol = sitk.GetArrayFromImage(vol)
    # aVol /= aVol.max()

    aMIP = np.amax(aVol, axis=1)
    aMIP /= aMIP.max()
    aMIP *= 255

    aMIP = aMIP.astype("uint8")
    sizey, sizex = aMIP.shape
    aMIP = aMIP.flatten()

    # 3 layers (rgb) - format uint8
    rgbImg = []
    rgbImg.append( aMIP.tolist() )
    rgbImg.append( aMIP.tolist() )
    rgbImg.append( aMIP.tolist() )

    return rgbImg, sizex, sizey, aMIP.dtype.str

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