# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json


def stitch(imgmark, N, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    overlapping_array = np.zeros((N,N))
    # Verify if images overlap with each other
    for i in range(N):
        for j in range(N):
            __, isoverlap = match_features(imgs[i],imgs[j])
            overlapping_array[i][j]=isoverlap
        
    # Get image with maximum overlap with other images
    max_overlap_img_index = np.argmax(np.sum(overlapping_array,axis=0))
    panoroma_img = imgs[max_overlap_img_index]
    stitch_img=[max_overlap_img_index]
    temp_array=[]
    for i in range(len(imgs)):
        if i!=max_overlap_img_index:
            if np.any([overlapping_array[i][l] for l in stitch_img]):
                panoroma_img = image_panaroma(panoroma_img,imgs[i])
                stitch_img.append(i)
            elif not np.any([overlapping_array[i][l] for l in stitch_img]):
                temp_array.append(i)
        
    if len(temp_array)!=0: 
        j=0        
        while j<len(temp_array):                                   
            if np.any([overlapping_array[temp_array[j]][l] for l in stitch_img]):                
                panoroma_img = image_panaroma(imgs[temp_array[j]],panoroma_img)                
                stitch_img.append(temp_array[j])                
            j+=1               
    cv2.imwrite(f'./{savepath}',panoroma_img)    
    return overlapping_array

# function used to 
def show_image(img, delay=1000):    
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# A function that computes matches between two input images by generating given number of keypoints.
def match_features(img1,img2): 
    image1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(2000) 
    keypoints1,descriptors1=sift.detectAndCompute(image1,None)  # keypoint and descriptor of img1 using sift
    keypoints2,descriptors2=sift.detectAndCompute(image2,None)  # keypoint and descriptor of img2 using sift
    match_score=[]  
    keypoint_matches=[] 
    descp_match_source=[]
    descp_match_dest=[]  
    descriptors1_index=0 
    for row in descriptors1:                    
        repeated_array = np.tile(row, (descriptors2.shape[0], 1))
        SSD= descriptors2 - repeated_array
        SSD=(SSD) ** 2
        SSD_FINAL=np.sum(SSD, axis=1)
        ssd_index = np.where(SSD_FINAL == np.amin(SSD_FINAL))
        ssd_index=ssd_index[0][0]
        SSD_FINAL.sort()
        if len(SSD_FINAL)>1:
            Ssd_score=SSD_FINAL[0]/SSD_FINAL[1]
        else:
            Ssd_score=SSD_FINAL[0]
        match_score.append(Ssd_score)
        descp_match_source.append(descriptors1_index) 
        descp_match_dest.append(ssd_index) 
        descriptors1_index=descriptors1_index+1
    count_match=0
    i=0
    good_matches_src=[]
    good_matches_dest=[]
    for value in match_score:
        if value < 0.3:
            count_match=count_match+1
            good_matches_src.append(descp_match_source[i])
            good_matches_dest.append(descp_match_dest[i])
        i=i+1

    if count_match>=25:
        source_points = np.float32([ keypoints1[match].pt for match in good_matches_src ]).reshape(-1,1,2)
        dest_points = np.float32([ keypoints2[match].pt for match in good_matches_dest]).reshape(-1,1,2)
        H, mask = cv2.findHomography(source_points, dest_points, cv2.RANSAC,5.0) 
        overlap = 1
    else:   
        H=0
        overlap = 0                       
    return H, overlap

# A function that stitches 2 images together using given number of keypoints    
def image_panaroma(img1, img2):
    
    # get homography between two images by calling match_features function
    H,__ = match_features(img1,img2)    
    height_img1, width_img1 = img1.shape[:2]
    height_img2, width_img2 = img2.shape[:2]
    img2_points = np.float32([[0,0], [0, height_img2],[width_img2, height_img2], [width_img2, 0]]).reshape(-1, 1, 2)
    img1_points = np.float32([[0,0], [0,height_img1], [width_img1,height_img1], [width_img1,0]]).reshape(-1,1,2)

    # Change perspective
    new_points_img1 = cv2.perspectiveTransform(img1_points, H)
    persp_points = np.concatenate((new_points_img1,img2_points), axis=0)
    [x_maximum, y_maximum] = np.int32(persp_points.max(axis=0).ravel() + 0.5)
    [x_minimum, y_minimum] = np.int32(persp_points.min(axis=0).ravel() - 0.5)  
    dist_translation = [-x_minimum,-y_minimum]
    Homography_translation = np.array([[1, 0, dist_translation[0]], [0, 1, dist_translation[1]], [0, 0, 1]])  
    warped_image = cv2.warpPerspective(img1, Homography_translation.dot(H), (x_maximum-x_minimum, y_maximum-y_minimum))     
    warped_image[dist_translation[1]:height_img2+dist_translation[1], dist_translation[0]:width_img2+dist_translation[0]] = img2    
    return warped_image  

if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t3',N=4, savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
