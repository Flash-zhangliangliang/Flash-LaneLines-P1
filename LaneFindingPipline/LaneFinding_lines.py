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


def hough_lane(img, rho, theta, threshold, min_line_len, max_line_gap, y_min):
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
    draw_lanes(line_img, lines, y_min)
    return line_img


def draw_lanes(img, lines, y_min, color=[255, 0, 0], thickness=8):
    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            if k < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)

    if len(left_lines) <= 0 or len(right_lines) <= 0:
        return img

    clean_lines(left_lines, 0.1)
    clean_lines(right_lines, 0.1)
    left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
    left_points = left_points + [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]
    right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]
    right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]

    left_vtx = calc_lane_vertices(left_points, y_min, img.shape[0])
    right_vtx = calc_lane_vertices(right_points, y_min, img.shape[0])

    cv2.line(img, left_vtx[0], left_vtx[1], color, thickness)
    cv2.line(img, right_vtx[0], right_vtx[1], color, thickness)


def clean_lines(lines, threshold):
    slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
    while len(lines) > 0:
        mean = np.mean(slope)
        diff = [abs(s - mean) for s in slope]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slope.pop(idx)
            lines.pop(idx)
        else:
            break


def calc_lane_vertices(point_list, y_min, y_max):
    x = [p[0] for p in point_list]
    y = [p[1] for p in point_list]
    fit = np.polyfit(y, x, 1)
    fit_fn = np.poly1d(fit)

    x_min = int(fit_fn(y_min))
    x_max = int(fit_fn(y_max))

    return [(x_min, y_min), (x_max, y_max)]


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


def process_image_lane(img):
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
    line_img = hough_lane(roi_edges, rho, theta, threshold, min_line_length, max_line_gap, 325)
    res_img = weighted_img(img, line_img, 0.8, 1, 0)

    return res_img


def process_video_lane(input_video, output_video):
    """
    video pip line
    :param input_video:
    :param output_video:
    :return:
    """
    clip = VideoFileClip(input_video)
    challenge_clip = clip.fl_image(process_image_lane)
    challenge_clip.write_videofile(output_video, audio=False)

