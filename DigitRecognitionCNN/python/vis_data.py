import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt

# Load the model architecture
layers = get_lenet()
params = init_convnet(layers)

# Load the network
data = loadmat('../results/lenet.mat')
params_raw = data['params']

for params_idx in range(len(params)):
    raw_w = params_raw[0,params_idx][0,0][0]
    raw_b = params_raw[0,params_idx][0,0][1]
    assert params[params_idx]['w'].shape == raw_w.shape, 'weights do not have the same shape'
    assert params[params_idx]['b'].shape == raw_b.shape, 'biases do not have the same shape'
    params[params_idx]['w'] = raw_w
    params[params_idx]['b'] = raw_b

# Load data
fullset = False
xtrain, ytrain, xvalidate, yvalidate, xtest, ytest = load_mnist(fullset)
m_train = xtrain.shape[1]

batch_size = 1
layers[0]['batch_size'] = batch_size

img = xtest[:,0]
img = np.reshape(img, (28, 28), order='F')
plt.imshow(img.T, cmap='gray')
plt.show()

output = convnet_forward(params, layers, img)
output_1 = np.reshape(output[0]['data'], (28,28), order='F')

##### Fill in your code here to plot the features ######


conv_layer_data = np.reshape(output[1]["data"],(output[1]["height"], output[1]["width"], output[1]["channel"]), order ="F")


#conv layer
plt.figure()
plt.title("Conv layer")

for i in range(20):
    plt.subplot(4,5,i+1)
    plt.imshow(conv_layer_data[:,:,i],cmap = "gray")
    plt.axis('off')

plt.show()

#relu layer
relu_layer_data = np.reshape(output[2]["data"],(output[2]["height"], output[2]["width"], output[2]["channel"]) , order='F')
plt.figure()
plt.title("ReLu Layer")

for i in range(20):
    plt.subplot(4,5,i+1)
    plt.imshow(relu_layer_data[:,:, i], cmap = "gray")
    plt.axis('off')

plt.show()