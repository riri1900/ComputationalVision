import numpy as np


def inner_product_forward(input, layer, param):
    """
    Forward pass of inner product layer.

    Parameters:
    - input (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """

    d, k = input["data"].shape
    n = param["w"].shape[1]


    ###### Fill in the code here ######

    #print(input["data"].T.shape)
    #print(param["b"].shape)
    output_data = np.dot(param["w"].T, input["data"]) + param["b"].T
    #print(output_data.shape)
    # Initialize output data structure
    output = {
        "height": n,
        "width": 1,
        "channel": 1,
        "batch_size": k,
        "data": output_data # replace 'data' value with your implementation
    }
    #print(output["data"].shape)


    return output


def inner_product_backward(output, input_data, layer, param):
    """
    Backward pass of inner product layer.

    Parameters:
    - output (dict): Contains the output data.
    - input_data (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """
    param_grad = {}
    ###### Fill in the code here ######
    # Replace the following lines with your implementation.
    param_grad['b'] = np.zeros_like(param['b'])
    param_grad['w'] = np.zeros_like(param['w'])
    input_od = None

 
    param_grad["b"] = np.sum(output["diff"], axis = 1).T
    param_grad["w"] = np.dot(input_data['data'], output["diff"].T) 

    input_od = np.dot( param["w"], output["diff"])

    return param_grad, input_od