import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os

#read the text file
def read_templetxt(file):
    img_params = {}
    with open(file, 'r') as f:
        lines = f.readlines()


    for l in lines:
        img = l.split()[0]
        parts = l.strip().split()
        val =[]
        for i in parts[1:]:
            val.append(float(i))
        K = np.array(val[:9]).reshape(3,3)
        R = np.array(val[9:18]).reshape(3,3)
        t = np.array(val[18:])

        img_params[img] = {'K':K, 'R':R, 't':t}
    return img_params


#print(img_params.items())

# calculates the projection matrix
def proj_mat(img_params):
    projection__matrices ={}
    for img_name, params in img_params.items():
        projmats =[]
        #print(params)
        
        K = params['K']
        R = params['R']
        t = params['t']

        Ext_mat = np.hstack((R, t.reshape(-1,1)))
        projmat = np.dot(K,Ext_mat)
    

        projection__matrices[img_name] = projmat
    return projection__matrices

# calculates the corners for images based on projection matrix and the camera corner values
def project_corners(corners, Proj_mat):

    corners_3d = np.hstack((corners, np.ones((corners.shape[0], 1))))
    projected_corners = (np.dot(Proj_mat, corners_3d.T).T)
    projected_corners = projected_corners[:, :2] / projected_corners[:, 2, np.newaxis]
    #print(projected_corners)
    return projected_corners


def Get3dCoord(q,I0,d):
    x,y = q
    P = I0['projection__matrix']
    q_coord = np.array([x,y,1])
    dx = np.dot(d,q_coord)
    d_p = dx - P[:,3]
    _3d_coord = np.dot(np.linalg.inv(P[:,:3]), d_p )
    #print(_3d_coord)

    X = _3d_coord[0]/_3d_coord[2]
    Y = _3d_coord[1]/_3d_coord[2]
    Z = _3d_coord[2]/_3d_coord[2]
    return X, Y, Z

def ComputeConsistency(I_0, I_1, X):
    X = np.array(X)
    #print(X)
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    projected_coords_I0 = (np.dot(I_0['projection__matrix'] , X.T).T)
    projected_coords_I0 = projected_coords_I0[:,:2]/ projected_coords_I0[:,2,np.newaxis]
    projected_coords_I1 = (np.dot(I_1['projection__matrix'] , X.T).T)
    projected_coords_I1 = projected_coords_I1[:,:2]/ projected_coords_I1[:,2,np.newaxis]
    #ensuring values are not beyond the values possible
    projected_coords_I1[:, 0] = np.clip(projected_coords_I1[:, 0], 0, I1['img'].shape[0] - 1)
    projected_coords_I1[:, 1] = np.clip(projected_coords_I1[:, 1], 0, I1['img'].shape[1] - 1)
    projected_coords_I0[:, 0] = np.clip(projected_coords_I0[:, 0], 0, I0['img'].shape[0] - 1)
    projected_coords_I0[:, 1] = np.clip(projected_coords_I0[:, 1], 0, I0['img'].shape[1] - 1)
    #print(I_1['projection__matrix'])
    C1 = I1['img'][projected_coords_I1[:,0].astype(int),projected_coords_I1[:,1].astype(int)]
    C0 = I0['img'][projected_coords_I0[:,0].astype(int),projected_coords_I0[:,1].astype(int)]
    #print(C0)
    return normalized_cross_corr(C0,C1)

def normalized_cross_corr(C0,C1):
    #print(C1)
    C0_m = np.mean(C0)
    C1_m = np.mean(C1)
    #print(C1_m)
    C0_sub = C0 - C0_m
    C1_sub = C1 - C1_m
    #print(C0_sub)
    C0_norm = np.linalg.norm(C0_sub)
    C1_norm = np.linalg.norm(C1_sub)
    #print(C0_norm)
    C0 = C0 /C0_norm
    C1 = C1 / C1_norm
    #print(C1)
    result = np.dot(C0.flatten(),C1.flatten())
    
    return result

def depthmap_algorithm(I0, I1, I2, I3,I4, min_depth, max_depth, depth_step, w_S, threshold):
    height, width, _ = I0['img'].shape
    count = 0
    depthmap = np.zeros((height, width), dtype=float)
    
    for y in range(height):
        for x in range(width):
            p = I0['img'][y, x]

            # Skip pixels that are 0
            if np.all(p == 0):  
                continue

            reg = I0['img'][max(0, y - w_S // 2):max(0, y - w_S // 2)+S,max(0, x - w_S // 2):max(0, x - w_S // 2)+w_S]
            #print(reg)
            #print(max(0, y - S // 2))
            #print(min(height, (y + S // 2) + 1))

            best_depth = 0
            best_consistency = -1

            for d in np.arange(min_depth, max_depth, depth_step):
                best_depth = d
                X = []
                #print(X)
                for q_y in range(max(0, y - w_S // 2),max(0, y - w_S // 2)+S):
                    for q_x in range(max(0, x - w_S // 2),max(0, x - w_S // 2)+S):
                        coord = Get3dCoord((q_x, q_y), I0, d)
                        X.append(coord)
             
                
                score01 = ComputeConsistency(I0, I1, X)
                score02 = ComputeConsistency(I0, I2, X)
                score03 = ComputeConsistency(I0, I3, X)
                score04 = ComputeConsistency(I0, I4, X)
                scores = np.array([score01,score02,score03,score04])
                #taking care of nan values
                scores_nan = ~np.isnan(scores)
                scores = scores[scores_nan]
                #print(scores)
                average_consistency = np.mean(scores)
                #print(average_consistency)
                if average_consistency>=threshold:
                    if average_consistency>=best_consistency:
                        best_consistency = average_consistency
                        best_depth = best_depth 
                    else:
                        best_depth = best_depth - 1

            #print(best_depth)
            depthmap[y, x] = best_depth
        count += 1 
        #print(count)

    return depthmap

#creatng a point cloud.
def create_point_cloud(depth_map, Proj_mat):
    height = depth_map.shape[0]
    width = depth_map.shape[1]
    y , x = np.nonzero(depth_map)
    points=[]
    for i in range(len(y)):
        X,Y,Z = Get3dCoord((x[i], y[i]), I0, depth_map[y[i], x[i]])
        points.append([X,Y,Z])
    return np.array(points)

#initializing the images 
I0={}
I1={}
I2={}
I3={}
I4={}

I0['img'] = cv2.imread('../data/templeR0013.png')
I1['img'] = cv2.imread('../data/templeR0014.png')
I2['img'] = cv2.imread('../data/templeR0016.png')
I3['img'] = cv2.imread('../data/templeR0043.png')
I4['img'] = cv2.imread('../data/templeR0045.png')
#reading the txt file and obtaining the projection matrices based on the value
img_params = read_templetxt('../data/templeR_par.txt')
projection__matrices = proj_mat(img_params)

#projection matrices
I0['projection__matrix'] = projection__matrices['templeR0013.png']
I1['projection__matrix'] = projection__matrices['templeR0014.png']
I2['projection__matrix'] = projection__matrices['templeR0016.png']
I3['projection__matrix'] = projection__matrices['templeR0043.png']
I4['projection__matrix'] = projection__matrices['templeR0045.png']


#grayscale images
img1 = cv2.imread('../data/templeR0013.png',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('../data/templeR0014.png',cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('../data/templeR0016.png',cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread('../data/templeR0043.png',cv2.IMREAD_GRAYSCALE)
img5 = cv2.imread('../data/templeR0045.png',cv2.IMREAD_GRAYSCALE)

#camera values
minxyz = [-0.023121, -0.038009, -0.091940]
maxxyz = [0.078626, 0.121636, -0.017395]

minx, miny, minz = minxyz
maxx, maxy, maxz = maxxyz

corners = np.array([[minx, miny, minz],
    [minx, maxy, minz],
    [maxx, maxy, minz],
    [maxx, miny, minz],
    [minx, miny, maxz],
    [minx, maxy, maxz],
    [maxx, maxy, maxz],
    [maxx, miny, maxz]])
#calculating th eprojected corners based on the camera values
projected_corners={}
for idx, img in enumerate([I0,I1,I2,I3,I4]):
    projected_corners[idx]=project_corners(corners, np.array(img['projection__matrix']))


# ploting the corners on the images
for idx, img in enumerate([I0,I1,I2,I3,I4]):
    plt.figure()
    plt.imshow(cv2.cvtColor(img['img'],cv2.COLOR_BGR2RGB))
    plt.scatter(projected_corners[idx][:, 0], projected_corners[idx][:, 1], c='red', marker='x')
    plt.show()



min_depth = int(np.min(projected_corners[0]))
max_depth = int(np.max(projected_corners[0]))
depth_step = 200
S = 5
threshold = 0.5
img_d1 = cv2.imread('../data/templeR0013.png',cv2.IMREAD_GRAYSCALE)
img_d3 = cv2.imread('../data/templeR0016.png',cv2.IMREAD_GRAYSCALE)
depth_map = depthmap_algorithm(I0, I1, I2, I3, I4, min_depth, max_depth, depth_step, S, threshold)



img1 = cv2.imread('../data/templeR0013.png')
img2 = cv2.imread('../data/templeR0014.png')
img3 = cv2.imread('../data/templeR0016.png')
img4 = cv2.imread('../data/templeR0043.png')
img5 = cv2.imread('../data/templeR0045.png')

depth_map_3d = np.stack([depth_map] * 3, axis = -1)
for i, img in enumerate([img1,img2,img3,img4,img5], start = 1):
    
    img_filtered = cv2.GaussianBlur(img, (5, 5), 0)
    plt.figure()
    plt.imshow(depth_map_3d * (img_filtered > 50), cmap='gray')
    plt.axis('image')
    plt.title('Depth Map')
    plt.savefig(f'../results/I{i}.png', dpi=600)


PointCloud1 = create_point_cloud(depth_map, I0['projection__matrix'])
os.makedirs(os.path.dirname('../results/point_cloud.obj'), exist_ok=True)
with open('../results/point_cloud.obj','w') as f:
    for p in PointCloud1:
        f.write(f'v {p[0]} {p[1]} {p[2]}\n')
