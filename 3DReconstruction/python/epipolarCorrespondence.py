import numpy as np
import cv2

def epipolarCorrespondence(im1, im2, F, pts1):
    """
    Args:
        im1:    Image 1
        im2:    Image 2
        F:      Fundamental Matrix from im1 to im2
        pts1:   coordinates of points in image 1
    Returns:
        pts2:   coordinates of points in image 2
    """

    pts2 = np.zeros_like(pts1)

    w_size = 5

    for i in range(pts1.shape[0]):
        x = int(pts1[i,0])
        y = int(pts1[i,1])
        v1 = np.array([x,y,1])

        ep_l = np.dot(F, v1)
        ep_l = ep_l / -ep_l[1]


        #creating a window around the point
        w_1 = im1[y - w_size:y + w_size + 1, x - w_size:x + w_size + 1, :].astype(float)

        #distance
        min_dist = np.inf

        for j in range(x-10,x+11):

            v2 = np.array([j , ep_l[0]*j + ep_l[2],1])
            x2 = int(v2[0])
            y2 = int(v2[1])

            w_2 = im2[y2 - w_size:y2 + w_size + 1, x2 - w_size:x2 + w_size + 1, :].astype(float)
            
            #using euclidian distance to calculate the distance
            #print(window1)
            diff = w_1 - w_2
            difference = diff * diff 
            sum_diff = np.sum(difference)
            distance = np.sqrt(sum_diff)

            if distance < min_dist:
                min_dist = distance
                pts2[i, 0] = v2[0]
                pts2[i, 1] = v2[1]


    return pts2
