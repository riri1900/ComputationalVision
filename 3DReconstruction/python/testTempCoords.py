import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
from PIL import Image
from eightpoint import eightpoint
from epipolarCorrespondence import epipolarCorrespondence
from essentialMatrix import essentialMatrix
from camera2 import camera2
from triangulate import triangulate
from displayEpipolarF import displayEpipolarF
from epipolarMatchGUI import epipolarMatchGUI

# Load images and points
img1 = cv2.imread('../data/im1.png')
img2 = cv2.imread('../data/im2.png')
pts = np.load('../data/someCorresp.npy', allow_pickle=True).tolist()
pts1 = pts['pts1']
pts2 = pts['pts2']
M = pts['M']

#Part 3.1.1 eightpoint to find F
F = eightpoint(pts1,pts2,M)
print("Estimated F(3.1.1): ")
print(F)
#displayEpipolarF(img1,img2,F)
#part 3.1.2
#epipolarMatchGUI(img1,img2,F)

#Load the points in image 1 within templecoords and running epipolarCorrespondence
temple_Coords = np.load('../data/templeCoords.npy', allow_pickle=True).tolist()
img_pts1 = temple_Coords['pts1']

img_pts2 = epipolarCorrespondence(img1,img2, F, img_pts1)

# write your code here
#part 3.1.3
K = np.load('../data/intrinsics.npy', allow_pickle=True).tolist()
K1, K2 = K['K1'], K['K2']
print("Estimated E(3.1.3): ")
E = essentialMatrix(F, K1 , K2)
print(E)

#First camera matrix P1
P1 = K1 @ np.hstack([np.eye(3), np.zeros((3,1))])
#using camera2.py to get P2
P2_cam = camera2(E)
#print(P2_cam.shape)

#Run triangulate function using the 4 sets of camera matrix
max_d = -np.inf

for i in range(4):
	P2 = K2 @ P2_cam[:,:,i]
	pt3d = triangulate(P1,img_pts1,P2,img_pts2)
	#print(pt3d[0])
	#re-projection 
	x_pt1 = P1 @ np.hstack((pt3d, np.ones((pt3d.shape[0], 1)))).T
	x_pt2 = P2 @ np.hstack((pt3d, np.ones((pt3d.shape[0], 1)))).T 
	x_pt1 = x_pt1/x_pt1[2,:]
	x_pt2 = x_pt2/x_pt2[2,:] 
	re_proj1 = np.linalg.norm(img_pts1 - x_pt1[:2,:].T) / pt3d.shape[0]
	re_proj2 = np.linalg.norm(img_pts2 - x_pt2[:2,:].T) /pt3d.shape[0]
	count = np.sum(pt3d[:2]>0)
	if max_d <= count:
		max_d = count
		plt_y = pt3d
		ind_num = i
		pts1_er = re_proj1
		pts2_er = re_proj2

#re-projection errors
print("Re-projection errors:")
print(f"pts1 error: {pts1_er}")
print(f"pts2 error: {pts2_er}")

#plot point correspondence
fig = plt.figure()
plot = fig.add_subplot(111, projection='3d')
plot.scatter(plt_y[:, 0], plt_y[:, 1], -plt_y[:, 2], marker='.')
plt.axis('equal')
plt.show()

R1, t1 = np.eye(3), np.zeros((3, 1))
R2, t2 = np.eye(3), np.zeros((3, 1))
R1 = np.linalg.inv(K1) @ P1[:, :3]
t1 = np.linalg.inv(K1) @ P1[:, 3]
R2 = np.linalg.inv(K2) @ P2[:, :3]
t2 = np.linalg.inv(K2) @ P2[:, 3]

# save extrinsic parameters for dense reconstruction
np.save('../results/extrinsics', {'R1': R1, 't1': t1, 'R2': R2, 't2': t2})
