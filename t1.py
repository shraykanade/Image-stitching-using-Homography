#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(img, delay=1000):
    # this function is used to display image
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    image1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()   
    keypoints1,descriptors1=sift.detectAndCompute(image1,None)  # keypoint and descriptor of img1 using sift
    keypoints2,descriptors2=sift.detectAndCompute(image2,None)  # keypoint and descriptor of img2 using sift
    match_score=[]  
    keypoint_matches=[] 
    descp_match_source=[]
    descp_match_dest=[]  
    descriptors1_index=0 
    # for each descriptor of img1 check if its a match with any of the descriptor of img 2
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
    
    # if there are more than or equal to 25 matches in descriptors, then images overlap
    if count_match>=30:
        print('match')  
        source_points = np.float32([ keypoints1[match].pt for match in good_matches_src ]).reshape(-1,1,2)
        dest_points = np.float32([ keypoints2[match].pt for match in good_matches_dest]).reshape(-1,1,2)
        H, mask = cv2.findHomography(source_points, dest_points, cv2.RANSAC,5.0)
        height_img1, width_img1 = img1.shape[:2]
        height_img2, width_img2 = img2.shape[:2]
        img2_points = np.float32([[0,0], [0, height_img2],[width_img2, height_img2], [width_img2, 0]]).reshape(-1, 1, 2)
        img1_points = np.float32([[0,0], [0,height_img1], [width_img1,height_img1], [width_img1,0]]).reshape(-1,1,2)

        # Changing perspective
        new_points_img1 = cv2.perspectiveTransform(img1_points, H)
        persp_points = np.concatenate((new_points_img1,img2_points), axis=0)
        [x_maximum, y_maximum] = np.int32(persp_points.max(axis=0).ravel() + 0.5)
        [x_minimum, y_minimum] = np.int32(persp_points.min(axis=0).ravel() - 0.5)  
        dist_translation = [-x_minimum,-y_minimum]
        Homography_translation = np.array([[1, 0, dist_translation[0]], [0, 1, dist_translation[1]], [0, 0, 1]])  
        warped_image = cv2.warpPerspective(img1, Homography_translation.dot(H), (x_maximum-x_minimum, y_maximum-y_minimum))
        show_image(warped_image)
        padded_img=padding_image(warped_image,img2,x_maximum,x_minimum,y_maximum,y_minimum)
        show_image(padded_img)
        cv2.imwrite(f'.{savepath}',padded_img)
    else:
        print('no overlap')  


def padding_image(warped_image,img2,x_maximum,x_minimum,y_maximum,y_minimum):
    # Performing padding
    height_img2, width_img2 = img2.shape[:2]
    padding_top = -y_minimum
    padding_bottom = y_maximum-height_img2
    padding_left = -x_minimum
    padding_right = x_maximum-width_img2
    img2 = cv2.copyMakeBorder(img2,padding_top,padding_bottom,padding_left,padding_right,borderType=cv2.BORDER_CONSTANT)
    warped_image_h, warped_image_w = warped_image.shape[:2]    
    padding_values = np.zeros(3)
    for i in range(warped_image_h):
        for j in range(warped_image_w):
            padding1=warped_image[i,j,:]
            padding2=img2[i,j,:]
            if not np.array_equal(padding1, padding_values) and np.array_equal(padding2, padding_values):
                warped_image[i, j, :] = padding1
            elif np.array_equal(padding1, padding_values) and not np.array_equal(padding2, padding_values):
                warped_image[i, j, :] = padding2
            elif not np.array_equal(padding1, padding_values) and not np.array_equal(padding2, padding_values):
                if sum(padding1)>sum(padding2):
                    warped_image[i, j, :] = padding1
                else:
                    warped_image[i, j, :] = padding2
            else:
                pass        
    return warped_image              
    

if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = '/task1.png'
    stitch_background(img1, img2, savepath=savepath)

