# 引入库
# Import Packages
import cv2
import numpy as np
from moviepy.editor import VideoFileClip


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
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # 将要保留的区域设置为255，不保留的区域设置为0
    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # 接下来进行and操作，保留要保留的区域
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    绘制车道线
    Drawing lane lines
    :param img:
    :param lines:
    :param color:
    :param thickness:
    :return:
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    霍夫变换
    Application of Hough transform
    :param img: 灰度图像 image after canny
    :param rho: 参数极径rho以像素值为单位的分辨率. 我们使用 1 像素
    :param theta: 参数极角theta  以弧度为单位的分辨率. 我们使用 1度 (即CV_PI/180)
    :param threshold: 要”检测” 一条直线所需最少的的曲线交点
    :param min_line_len: 能组成一条直线的最少点的数量. 点数量不足的直线将被抛弃.线段的最小长度
    :param max_line_gap: 线段上最近两点之间的阈值
    :return:
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, thickness=8)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    将车道线与原来的图片叠加
    return_img = initial_img * α + img * β + γ

    :param img:
    :param initial_img:
    :param α:
    :param β:
    :param γ:
    :return:
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def process_image(img):
    """
    图片处理管道
    image process pip line
    :param img:
    :return:
    """
    roi_vtx = np.array([[(0, img.shape[0]), (460, 325), (520, 325), (img.shape[1], img.shape[0])]])

    blur_kernel_size = 5  # Gaussian blur kernel size
    canny_low_threshold = 50  # Canny edge detection low threshold
    canny_high_threshold = 150  # Canny edge detection high threshold

    # Hough transform parameters
    rho = 1
    theta = np.pi / 180
    threshold = 15
    min_line_length = 40
    max_line_gap = 20

    gray = gray_scale(img)
    blur_gray = gaussian_blur(gray, blur_kernel_size)
    edges = canny(blur_gray, canny_low_threshold, canny_high_threshold)
    roi_edges = region_of_interest(edges, roi_vtx)
    line_img = hough_lines(roi_edges, rho, theta, threshold, min_line_length, max_line_gap)
    res_img = weighted_img(img, line_img, 0.8, 1, 0)

    return res_img


def process_video(input_video, output_video):
    """
    video pip line
    :param input_video:
    :param output_video:
    :return:
    """
    clip = VideoFileClip(input_video)
    challenge_clip = clip.fl_image(process_image)
    challenge_clip.write_videofile(output_video, audio=False)

