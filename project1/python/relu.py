import numpy as np

def relu_forward(input_data):
    output = {
        'height': input_data['height'],
        'width': input_data['width'],
        'channel': input_data['channel'],
        'batch_size': input_data['batch_size'],
    }

    ###### Fill in the code here ######
    # Replace the following line with your implementation.
    output['data'] = np.zeros_like(input_data['data'])
    
    for b in range(input_data["batch_size"]):
        for i in range(input_data["data"].shape[0]):
            if input_data["data"][i,b] < 0:
                output["data"][i,b] = 0
            else:
                output["data"][i,b] = input_data["data"][i,b] 

    #print(output["data"].shape)

    return output

def relu_backward(output, input_data, layer):
    ###### Fill in the code here ######
    # Replace the following line with your implementation.
    input_od = np.zeros_like(input_data['data'])
    inp = input_data['data'].copy()
    inp[inp<=0] = 0
    inp[inp>0] = 1
    input_od = output['diff'] * inp
    return input_od
