from flask import Flask, render_template, request, make_response, jsonify

import torch
import SimpleITK as sitk 
from static.api.demo_inference_single import pad_3D_array, load_patient_mhd , output_to_itk
from static.api.fluence_net import fluence_net_v2
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/get-fake-mip-scattering', methods=['POST'])

def getFakeMip():
    request_data = request.get_json()
   
    o = request_data['LAORAO']
    a = request_data['CAUCRA']
    x = request_data['TRANSLATION X']
    y = request_data['TRANSLATION Y']
    z = request_data['TRANSLATION Z']
    kvp = request_data['TENSION DU TUBE']
    print(kvp)
    print("Get POST: request fake MIP for laorao ", o, " and caucra ", a , "x", x, "y",y, "z", z, "Tension du tube", kvp)
    
    print(o)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    patient = 5

    input_dimensions = (100,100,100)
    
    ## Model loading
    model = fluence_net_v2(image_shape = input_dimensions,
                                       init_filters=4, n_params=6,
                                       inference_mode=True)  
    model_wts = "model.pt"

    checkpoint = torch.load(model_wts)
    model.load_state_dict(checkpoint)
    model.to(device)        

   ## Read the patient input as numpy array and then convert to Tensor
    ct_scan = torch.from_numpy(load_patient_mhd("patient_images/image_{}_resampled.mhd".format(patient),
                                          input_dimensions = input_dimensions))

	 ## Create a Tensor with the parameters

    # device ="CPU"
    count = 0;t=0
    
    ct_scan = ct_scan.to(device);#print(x.shape)
    N_REPETITIONS = 5000
    print("start")
    
    save_file_root= "output/scatter_map_patient_" +str(patient) + \
    "_orbital_" + str(o) + "_angular_" + str(a) + "_kvp_" + \
    str(kvp) + "_x_" + str(x) +"_y_" + str(y)+"_z_"+str(z)
    
   
    with torch.no_grad():
        
        param = torch.FloatTensor([o,a,x,y,z,kvp],)
        param = param[None,:]        
            
        param = param.to(device)
        y_pred = model(ct_scan,param).cpu().numpy()[0][0];
                
            
        sitk.WriteImage(output_to_itk(y_pred),save_file_root+".mhd") ## Save as mhd
        np.save(save_file_root+".npy",y_pred ) # save as numpy
        
   ## doseSlice = y_pred[10, :, :] ## 
  
    doseSlice = np.amax( y_pred, axis = 1)
    
    aMIP = doseSlice.astype("uint8")

    from matplotlib.colors import LinearSegmentedColormap
    cmm=LinearSegmentedColormap.from_list('my_colormap', ['#be0aff','#023e8a','#386641','#007f5f','#0aff99','#a1ff0a','#deff0a','#deff0a','#ffea00','#ffd670','#ffd670','#ff9500','#ff9000','#f26419','#bc3908','#bc3908','#bc3908','#dc2f02','#dc2f02','#dc2f02','#dc2f02','#ef2b2b','#ef2b2b','#ef2b2b','#de1616','#de1616','#de1616','#cd0000','#cd0000','#cd0000','#bb0000','#bb0000','#bb0000','#aa0000','#aa0000','#aa0000','#6a040f','#6a040f','#6a040f','#6a040f','#6a040f','#6a040f','#6a040f','#6a040f','#6a040f','#6a040f','#6a040f','#6a040f' ] , N=200) 



    origin = 'lower'
    
    x, y = np.meshgrid(np.arange(100), np.arange(100))
    z = aMIP
    
    fig1, ax2 = plt.subplots(constrained_layout=True)
    
    CS = ax2.contourf(x, y, z, 80,cmap=cmm, origin=origin)
    
    
    CS2 = ax2.contour(CS, levels=CS.levels[::1],linewidths = 0.3, colors='k', origin=origin)
    ax2.set_axis_off()
    fig1.savefig('static/img/test1.png')
    #plt.imsave('static/img/test1.png', doseSlice, vmin=1, vmax=255, cmap=cmm)

    plt.imshow(doseSlice, vmin=1, vmax=200,cmap=cmm)
    plt.colorbar(label="Color Ratio")

    plt.show()
    
        
    return ("fine")



# Example
@app.route('/json-example', methods=['POST'])
def json_example():

    request_data = request.get_json()
   
    laorao = request_data['LAORAO']
    caucra = request_data['CAUCRA']
    print("Get POST", laorao, caucra)

    msg = {
        "msg": "Working"
    }

    response = make_response(jsonify(msg))

    return response

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)

