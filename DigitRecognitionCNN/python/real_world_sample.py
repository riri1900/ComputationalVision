import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt
from PIL import Image

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

images = np.zeros((784,9))
for i in range(1,10):
	img_path = f'../images/samples/{i}.jpg'
	#print(img_path)
	img = Image.open(img_path).convert('L')
	img_ar = np.array(img)
	#print(img_ar.shape)
	sample = img_ar.reshape(-1,1)
	#print(img)
	images[:,i - 1] = sample[:,0]


layers[0]['batch_size'] = images.shape[1]
#print(images)
output,P = convnet_forward(params,layers,images,test=True)
Pred = np.argmax(P, axis = 0)

count = np.sum(Pred == np.arange(1,10))
print('Number of correct predictions:', count)
print(np.arange(1,10))
print(Pred)





