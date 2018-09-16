# 引入库
# Import Packages
import cv2
import numpy as np


def gray_scale(img):
    """
    灰度转换
    Applies the Gray scale transform
    :param img:
    :return: grey image
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def gaussian_blur(img, kernel_size):
    """
    高斯滤波
    Applies a Gaussian Noise kernel
    :param img:
    :param kernel_size:
    :return: image after a Gaussian Noise kernel
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    """
    边缘检测
    Applies the Canny transform
    :param img: 
    :param low_threshold: 
    :param high_threshold: 
    :return: image after thr Canny transform
    """
    return cv2.Canny(img, low_threshold, high_threshold)


def region_of_interest(img, vertices):
    """
    区域检测
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.

    :param img:
    :param vertices:
    :return:
    """
    # 定义一个区域
    # 先定义一个空白的图片
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    # if len(img.shape) > 2:
    #    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    #    ignore_mask_color = (255,) * channel_count
    # else:
    #    ignore_mask_color = 255
    ignore_mask_color = 255

    # 将要保留的区域设置为255，不保留的区域设置为0
    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # 接下来进行and操作，保留要保留的区域
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

