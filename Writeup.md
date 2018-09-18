# **Finding Lane Lines on the Road**

## Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on my work in a written report


[//]: # (Image References)

[image_org]: ./test_images/solidWhiteRight.jpg "original"
[image_gray]: ./process_images/solidWhiteRight_gray.jpg "Grayscale"
[image_gauss]: ./process_images/solidWhiteRight_gauss.jpg "Gaussian_blur"
[image_canny]: ./process_images/solidWhiteRight_canny.jpg "Canny"
[image_interest]: ./process_images/solidWhiteRight_interest.jpg "region_of_interest"
[image_hough]: ./process_images/solidWhiteRight_hough.jpg "hough_lines"
[image_res]: ./process_images/solidWhiteRight_weighted.jpg "weighted_img"

---

### Reflection

### 1. Describe my pipeline.

My pipeline consisted of 6 steps. The follow image is the original image.

![alt text][image_org]

( 1 ) I converted the images to grayscale, the resault is as follow image.

![alt text][image_gray]

( 2 ) Then I apply Gauss blur on the image, the resault is as follow image.

![alt text][image_gauss]

( 3 ) Then I drow the , the resault is as follow image.

![alt text][image_canny]

( 4 ) And I define the area [(0, imshape[0]),(460, 325), (520, 325), (imshape[1], imshape[0])] as the region of my interest, the resault is as follow image.

![alt text][image_interest]

( 5 ) Then I apply Hough transform on the image, the resault is as follow image.

![alt text][image_hough]

( 6 ) the last, I draw the lane lines on the original image, the resault is as follow image.

![alt text][image_res]


### how I modified the draw_lines() function

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by 4 steps:

( 1 ) First I separate the lines into two groups -- left lanes or right lanes by the positive slop or the negative slope.

( 2 ) Then clean those lines with greater slopes than the threshold.

( 3 ) After I calculate the vertices of the lanes.

( 4 ) At last, the lanes are drawed down.

The modified draw_lines() fuction can be found in LanFinding_lines.py.

### 2. potential shortcomings of my current pipeline


One potential shortcoming would be what would happen when the lane lins are not lines, but curves, just as the lines in video challenge.mp4.


### 3. Suggest possible improvements to my pipeline

A possible improvement would be to identify the lanes are lines or curves.
