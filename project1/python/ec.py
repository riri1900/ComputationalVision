import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from cv2 import imshow

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


image_path =  f'../images/image1.jpg'
image1 = cv2.imread(image_path)


gray_img1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)


thresh, image1_bin =cv2.threshold(gray_img1,0,255,cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)
cv2.floodFill(image1_bin, None, (0, 0), 0)
contours, hierarchy = cv2.findContours(image1_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image1_copy = image1.copy()

for i, con in enumerate(contours):
    main_data = np.zeros((784,1))
    bounding_box =cv2.boundingRect(con)

    x = bounding_box[0]
    y = bounding_box[1]
    h = bounding_box[2]
    w = bounding_box[3]

    ar = w * h
    min_ar =10

    if ar > min_ar:
        color = (255, 0, 0)
        cv2.rectangle(image1_copy, (int(x-30), int(y-30)),(int(x + w+20), int(y + h+70)), color, 1)
        #pil_img1_box = Image.fromarray(image1_copy)
        #pil_img1_box.show()
        current_box = image1_bin[y-30:y + h+70, x-30:x + w+ 20]
        padded_box = cv2.copyMakeBorder(current_box, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
        # pil_cropped = Image.fromarray(padded_box)
        # pil_cropped.show()
        #print(current_box.shape)
        resizedbox = cv2.resize(padded_box, (28, 28))
        data = np.reshape(resizedbox,(-1,1))
        #pil_cropped = Image.fromarray(resizedbox)
        #pil_cropped.show()

        main_data[:,0] = data[:,0] 
        layers[0]['batch_size'] = main_data.shape[1] 
        output,P = convnet_forward(params,layers,main_data,test=True)
        Pred = np.argmax(P, axis = 0)
        pred_text_size, _ = cv2.getTextSize(str(Pred),cv2.FONT_HERSHEY_SIMPLEX ,0.5 , 1)
        text_x = int(x + w / 2 - pred_text_size[0] / 2)
        text_y = int(28)
        cv2.putText(image1_copy, str(Pred), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
pil_cropped = Image.fromarray(image1_copy)
pil_cropped.show()


        


################################### image 2######################################

image_path =  f'../images/image2.jpg'
image1 = cv2.imread(image_path)


gray_img1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)


thresh, image1_bin =cv2.threshold(gray_img1,0,255,cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)
cv2.floodFill(image1_bin, None, (0, 0), 0)

contours, hierarchy = cv2.findContours(image1_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image1_copy = image1.copy()

for i, con in enumerate(contours):
    main_data = np.zeros((784,1))
    bounding_box =cv2.boundingRect(con)

    x = bounding_box[0]
    y = bounding_box[1]
    h = bounding_box[2]
    w = bounding_box[3]

    ar = w * h
    min_ar =10

    if ar > min_ar:
        color = (255, 0, 0)
        cv2.rectangle(image1_copy, (int(x-20), int(y-20)),(int(x + w+10), int(y + h+70)), color, 1)
        #pil_img1_box = Image.fromarray(image1_copy)
        #pil_img1_box.show()
        current_box = image1_bin[y-20:y + h+70, x-20:x + w+10]
        padded_box = cv2.copyMakeBorder(current_box, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
        #pil_cropped = Image.fromarray(padded_box)
        #pil_cropped.show()
        #print(current_box.shape)
        resizedbox = cv2.resize(padded_box, (28, 28))
        data = np.reshape(resizedbox,(-1,1))
        #pil_cropped = Image.fromarray(resizedbox)
        #pil_cropped.show()

        main_data[:,0] = data[:,0] 
        layers[0]['batch_size'] = main_data.shape[1] 
        output,P = convnet_forward(params,layers,main_data,test=True)
        Pred = np.argmax(P, axis = 0)
        pred_text_size, _ = cv2.getTextSize(str(Pred),cv2.FONT_HERSHEY_SIMPLEX ,0.5 , 1)
        text_x = int(x + w / 2 - pred_text_size[0] / 2)
        text_y = int(28)
        cv2.putText(image1_copy, str(Pred), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
pil_cropped = Image.fromarray(image1_copy)
pil_cropped.show()


####################################image3################################################# 



image_path =  f'../images/image3.png'
image1 = cv2.imread(image_path)


gray_img1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)


thresh, image1_bin =cv2.threshold(gray_img1,0,255,cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)
cv2.floodFill(image1_bin, None, (0, 0), 0)
contours, hierarchy = cv2.findContours(image1_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image1_copy = image1.copy()

for i, con in enumerate(contours):
    main_data = np.zeros((784,1))
    bounding_box =cv2.boundingRect(con)

    x = bounding_box[0]
    y = bounding_box[1]
    h = bounding_box[2]
    w = bounding_box[3]

    ar = w * h
    min_ar =10

    if ar > min_ar:
        color = (255, 0, 0)
        cv2.rectangle(image1_copy, (int(x-5), int(y-10)),(int(x + w/2)-10, int(y + h+100)), color, 1)
        #pil_img1_box = Image.fromarray(image1_copy)
        #pil_img1_box.show()
        current_box = image1_bin[y-10:y + h+100, x-5:int(x+w/2)-10]
        padded_box = cv2.copyMakeBorder(current_box, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
        # pil_cropped = Image.fromarray(padded_box)
        # pil_cropped.show()
        #print(current_box.shape)
        resizedbox = cv2.resize(padded_box, (28, 28))
        data = np.reshape(resizedbox,(-1,1))
        #pil_cropped = Image.fromarray(resizedbox)
        #pil_cropped.show()

        main_data[:,0] = data[:,0] 
        layers[0]['batch_size'] = main_data.shape[1] 
        output,P = convnet_forward(params,layers,main_data,test=True)
        Pred = np.argmax(P, axis = 0)
        pred_text_size, _ = cv2.getTextSize(str(Pred),cv2.FONT_HERSHEY_SIMPLEX ,0.5 , 1)
        text_x = int( x+2 )
        text_y = int(10)
        cv2.putText(image1_copy, str(Pred), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
pil_cropped = Image.fromarray(image1_copy)
pil_cropped.show()




####################################image4############################################



image_path =  f'../images/image4.jpg'
image1 = cv2.imread(image_path)


gray_img1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)


thresh, image1_bin =cv2.threshold(gray_img1,0,255,cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)
cv2.floodFill(image1_bin, None, (0, 0), 0)
contours, hierarchy = cv2.findContours(image1_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image1_copy = image1.copy()

for i, con in enumerate(contours):
    main_data = np.zeros((784,1))
    bounding_box =cv2.boundingRect(con)

    x = bounding_box[0]
    y = bounding_box[1]
    h = bounding_box[2]
    w = bounding_box[3]

    ar = w * h
    min_ar =10

    if ar > min_ar:
        color = (255, 0, 0)
        cv2.rectangle(image1_copy, (int(x-5), int(y-5)),(int(x + w-10), int(y + h+20)), color, 1)
        #pil_img1_box = Image.fromarray(image1_copy)
        #pil_img1_box.show()
        current_box = image1_bin[y-5:y + h+20, x-5:int(x+w-10)]
        padded_box = cv2.copyMakeBorder(current_box, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
        # pil_cropped = Image.fromarray(padded_box)
        # pil_cropped.show()
        #print(current_box.shape)
        resizedbox = cv2.resize(padded_box, (28, 28))
        data = np.reshape(resizedbox,(-1,1))
        #pil_cropped = Image.fromarray(resizedbox)
        #pil_cropped.show()

        main_data[:,0] = data[:,0] 
        layers[0]['batch_size'] = main_data.shape[1] 
        output,P = convnet_forward(params,layers,main_data,test=True)
        Pred = np.argmax(P, axis = 0)
        pred_text_size, _ = cv2.getTextSize(str(Pred),cv2.FONT_HERSHEY_SIMPLEX ,0.5 , 1)
        text_x = int( x-2 )
        text_y = int(y+30)
        cv2.putText(image1_copy, str(Pred), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
pil_cropped = Image.fromarray(image1_copy)
pil_cropped.show()

