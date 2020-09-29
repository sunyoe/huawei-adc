import cv2
import sys
import time
import math
import numpy as np
from ctypes import *

config = {}


class Timer:
    def __init__(self):
        self.t = 0

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(time.time() - self.t)


def get_perspective_mar(height, width):
    '''获取透射变化矩
    '''
    src_points = np.array([[190 * config['resize_p'], 267 * config['resize_p']],
                           [445 * config['resize_p'], 268 * config['resize_p']],
                           [51 * config['resize_p'], 308 * config['resize_p']],
                           [593 * config['resize_p'], 318 * config['resize_p']]], dtype="float32")
    # 俯视图坐标
    x_len = config['calibration']['x_len']
    y_len = x_len * config['calibration']['x_y_p']
    x_left = width / 2 - x_len / 2
    y_down = height - 100 * config['resize_p']
    dst_points = np.array([[x_left, y_down - y_len],
                           [x_left + x_len, y_down - y_len],
                           [x_left, y_down],
                           [x_left + x_len, y_down]], dtype="float32")

    mar = cv2.getPerspectiveTransform(src_points, dst_points)
    mar_reverse = cv2.getPerspectiveTransform(dst_points, src_points)
    return mar, mar_reverse


def perspective_transform(frame_in, mar):
    """透视变换
    """
    height, width = frame_in.shape[0], frame_in.shape[1]
    frame_out = cv2.warpPerspective(frame_in, mar, (width, height), cv2.INTER_LINEAR)
    return frame_out


def base_transform(frame_in):
    """基本变换
    缩放->灰度->降噪->二值化"""

    frame_mid = cv2.resize(frame_in, (int(frame_in.shape[1] * config['resize_p']),
                                      int(frame_in.shape[0] * config['resize_p'])))  # 缩小
    frame_mid = cv2.cvtColor(frame_mid, cv2.COLOR_BGR2GRAY)  # 彩色->灰度
    frame_mid = cv2.erode(frame_mid, np.ones((3, 3), np.uint8), iterations=1)  # 腐蚀降噪
    frame_out = cv2.threshold(frame_mid, config['threshold'], 255, type=cv2.THRESH_BINARY)[1]  # 二值化

    return frame_out


def find_base_point(frame_in):
    """寻找赛道的基准点
    """
    height, width = frame_in.shape[0:2]
    b_point = []
    # left base point
    # 左边下边界跳变点扫描
    # for col in range(1, width // 2):
    #     if frame_in[height - 1, col] == 255:
    #         if frame_in[height - 1, col - 1] == 0:
    #             b_point.append([height - 1, col])
    # # 左边界扫描
    # for row in range(height // 2, height):
    #     if frame_in[row, 0] == 255:
    #         if frame_in[row - 1, 0] == 0:
    #             b_point.append([row, 0])
    # # right base point
    # # 右下边界跳变点扫描
    # for col in range(width // 2, width - 1):
    #     if frame_in[height - 1, col] == 255:
    #         if frame_in[height - 1, col + 1] == 0:
    #             b_point.append([height - 1, col])
    # # 右边界跳变点扫描
    # for row in range(height // 2, height):
    #     if frame_in[row, width - 1] == 255:
    #         if frame_in[row - 1, width - 1] == 0:
    #             b_point.append([row, width - 1])


    # 左边界检测
    left_array = frame_in[int(height/2):,0]
    kernel_up = np.array([[-1],
                          [1],
                          [0]]
                      )
    left_array_1 = cv2.filter2D(left_array, -1, kernel_up)
    k1 = np.transpose(left_array_1.nonzero())
    if len(k1)!=0:
        b_point.append([k1[0,0]+int(height/2),0])
    # 左下半边界检测
    left_bottem = frame_in[height-1,0:int(width/2)]
    kernel_left = np.array([[-1,1,0]])
    left_bottem_array = cv2.filter2D(left_array, -1, kernel_left)
    k2 = np.transpose(left_bottem_array.nonzero())
    if len(k2)!=0:
        b_point.append([height-1,k2[0,0]])
    #右边界扫描
    right_arry = frame_in[int(height/2):,width-1]
    right_array_1 = cv2.filter2D(right_arry, -1, kernel_up)
    k3 = np.transpose(right_array_1.nonzero())
    if len(k3)!=0:
        b_point.append([k3[0,0]+int(height/2),width-1])
    #右下边界扫描
    right_bottem = frame_in[height-1,int(width/2):]
    kernel_right = np.array([[0,1,-1]])
    right_bottem_array = cv2.filter2D(right_arry, -1, kernel_right)
    k4 = np.transpose(right_bottem_array.nonzero())
    if len(k4)!=0:
        b_point.append([height-1,int(width/2)+k4[0,0]])

    return b_point


def cal_point_dis(point1, point2):
    '''计算两个点距离
    param point1:
    param point2:
    return:
    '''
    distance = math.sqrt(math.pow(point1[0] - point2[0], 2) +
                         math.pow(point1[1] - point2[1], 2))
    return distance


def cal_line_dis(line_cal):
    '''
    计算线之间的距离
    :param line_cal:
    :return:
    '''
    distance = 0
    for i in range(len(line_cal) - 1):
        distance += cal_point_dis(line_cal[i], line_cal[i + 1])
    return distance


def cal_line_slope(line_in):
    '''
    计算线段斜率
    :param line_in:
    :return:
    '''
    dx = 0
    for i in range(len(line_in) - 1):
        dx += (line_in[i + 1][1] - line_in[i][1])
    return dx


def next_line_jump(frame_in, point_in, deltx):
    '''
    计算下一个跳变点
    :param frame_in:输入图像
    :param point_in:基准点
    :param deltx:
    :return:
    '''
    current_line = point_in[0]
    current_x = point_in[1]
    next_predict = current_x + deltx
    tmp_x = next_predict + 2 * config['large_dx_err']
    found = False
    # 确定扫描y位置
    if current_line != 0:
        next_line = current_line - 1
    else:
        return None
    # 确定扫描x方向误差限
    if deltx < 0:  # 线条左延伸
        left_max_err = config['large_dx_err']
        right_max_err = config['small_dx_err']
    elif deltx > 0:  # 线条右延伸
        left_max_err = config['small_dx_err']
        right_max_err = config['large_dx_err']
    else:  # 起始位置
        left_max_err = config['large_dx_err']
        right_max_err = config['large_dx_err']
    # 确定扫描x范围
    if next_predict - left_max_err > 1:
        min_x = next_predict - left_max_err
    else:
        min_x = 1
    if next_predict + right_max_err < frame_in.shape[1] - 2:
        max_x = next_predict + right_max_err
    else:
        max_x = frame_in.shape[1] - 2
    # 开始扫描
    for x in range(min_x, max_x):
        if frame_in[next_line][x] == 255:
            if frame_in[next_line][x - 1] == 0 or frame_in[next_line][x + 1] == 0:
                # 寻找距离最近的跳变点
                if abs(x - current_x) < abs(tmp_x - current_x):
                    tmp_x = x
                    found = True
    if found:
        return [next_line, tmp_x]
    else:
        return None


def find_line(frame_in, b_point):
    """根据基本点寻找赛道边沿
    """
    lines = []
    for jump_point in b_point:
        line = [jump_point]
        dx = 0
        while True:
            # 寻找下一行的跳变点
            next_jump_p = next_line_jump(frame_in, jump_point, dx)
            # 若没有跳变点就结束搜索
            if next_jump_p is None:
                lines.append(line)
                break
            # 更新
            line.append(next_jump_p)
            dx = next_jump_p[1] - jump_point[1]
            jump_point = next_jump_p
    return lines


def extract_road(frame_in, lines_in, num):
    height, width = frame_in.shape[0:2]
    # 区分左右线条
    left_lines = []
    right_lines = []
    for ln in lines_in:
        if ln[0][1] < width // 2:
            left_lines.append(ln)
        else:
            right_lines.append(ln)
    # 找出左侧赛道线
    left_line = []
    max_left_dis = 0
    for ln in left_lines:
        dis = cal_line_dis(ln)
        if (dis > max_left_dis and
                dis / cal_point_dis(ln[0], ln[-1]) < config['line_dis_p']):
            left_line = ln
            max_left_dis = dis
    # 找出右侧赛道线
    right_line = []
    max_right_dis = 0
    for ln in right_lines:
        dis = cal_line_dis(ln)
        if (dis > max_right_dis and
                dis / cal_point_dis(ln[0], ln[-1]) < config['line_dis_p']):
            right_line = ln
            max_right_dis = dis
    # 找出最长的赛道线
    if max_left_dis > max_right_dis:
        longest_line = left_line
    else:
        longest_line = right_line
    if num == 1:
        if len(longest_line):
            return [longest_line]
        else:
            return []
    else:
        return [left_line, right_line]


def get_show_frame(frame_in, base_points, lines, roads, road_fit, center, mar_per_rev):
    height, width = frame_in.shape[0:2]
    frame_show = frame_in
    # 显示基本点
    for point in base_points:
        cv2.circle(frame_show, (int(point[1]), int(point[0])), 10, (0, 255, 0), -1)
    # 显示扫描点
    for line in lines:
        for pix in line:
            cv2.circle(frame_show, (int(pix[1]), int(pix[0])), 2, (255, 0, 0), -1)
    # 显示道路点
    for road in roads:
        for pix in road:
            cv2.circle(frame_show, (int(pix[1]), int(pix[0])), 2, (255, 255, 0), -1)
    # 显示逆透视变换的道路fit线
    road_fit = get_perspective_road(road_fit, mar_per_rev)
    for i in range(len(road_fit) - 1):
        cv2.line(frame_show, (int(road_fit[i][1]), int(road_fit[i][0])), (int(road_fit[i + 1][1]), int(road_fit[i + 1][0])),
                 color=(0, 0, 255), thickness=2)
    # 显示逆透视变换的中心点
    center_show = [[y, x] for (y, x) in center if 0 < x < width and 0 < y < height]
    center_show = get_perspective_road(center_show, mar_per_rev)
    for i, pix in enumerate(center_show):
        if i < config['forward_num']:
            cv2.circle(frame_show, (int(pix[1]), int(pix[0])), 2, (0, 255, 255), -1)
        else:
            cv2.circle(frame_show, (int(pix[1]), int(pix[0])), 2, (255, 0, 255), -1)
    return frame_show


def get_vertical_view(frame_in, line, mar):
    frame_sparse = np.zeros_like(frame_in)
    for i in range(len(line) - 1):
        cv2.line(frame_sparse, (line[i][1], line[i][0]), (line[i + 1][1], line[i + 1][0]),
                 color=(255, 255, 255), thickness=1)
    frame_vertical = perspective_transform(frame_sparse, mar)
    return frame_vertical


def get_perspective_road(road, mar):
    src = np.ones((3, 1))
    perspective_road = []
    for point in road:
        src = (point[1], point[0], 1)
        dst = np.matmul(mar, src)
        dst = dst / dst[2]
        perspective_road.append([dst[1], dst[0]])
    return perspective_road


def get_center(road, direction, redis):
    center = []
    road_fit = []

    road_np = np.array(road)
    point_x, point_y = road_np[:, 1], road_np[:, 0]
    param = np.polyfit(point_y, point_x, 2)  # 多项式拟合
    f = np.poly1d(param)  # 多项式方程
    for y in point_y:
        # 多项式拟合结果
        x = f(y)
        # 对三次多项式求导
        # derivative = 3 * param[0] * y * y + 2 * param[1] * y + param[2]
        # 对二次多项式求导
        derivative = 2 * param[0] * y + param[1]
        # 求斜率
        k = -derivative
        # 求截距
        b = y - k * x
        # 求中心点x
        if direction == 0:  # 左侧车道
            resolve_x = x + redis / math.sqrt(1 + k * k)
        else:  # 右侧车道
            resolve_x = x - redis / math.sqrt(1 + k * k)
        resolve_y = k * resolve_x + b
        # 得到中心点
        center_x, center_y = resolve_x, resolve_y
        center.append([center_y, center_x])
        road_fit.append(([y, x]))
    return center, road_fit


def get_error(center, width, num):
    err = 0
    bias = width // 2
    for i, (_, x) in enumerate(center):
        err += (x - bias)
        if i > num:
            break
    return err / num


def test_perspective_transform():
    with Timer():
        frame0 = cv2.imread("image/transform.jpg")
        frame_mid = cv2.resize(frame0, (int(frame0.shape[1] * config['resize_p']),
                                          int(frame0.shape[0] * config['resize_p'])))  # 缩小
        cv2.imshow('test', frame_mid)
        cv2.waitKey()
        mar, _ = get_perspective_mar(frame_mid.shape[0], frame_mid.shape[1])
        frame2 = perspective_transform(frame_mid, mar)  # 透视变换
        cv2.imshow('test', frame2)
        cv2.waitKey()


def init_config():
    config['resize_p'] = 1 / 1.0  # 图像缩放比例
    config['threshold'] = 200  # 二值化阈值
    config['line_dis_p'] = 3.14159 / 2  # 用于去除锐角三角形的比例阈值
    config['calibration'] = {}
    config['calibration']['x_len'] = 200 * config['resize_p']  # 标定参数：缩放像素自由度
    config['calibration']['x_y_p'] = 0.9082  # 标定参数：固定长宽比
    config['small_dx_err'] = int(30 * config['resize_p'])  # 行扫描线时最大的反向准许x偏差像素
    config['large_dx_err'] = int(40 * config['resize_p'])  # 行扫描线时最大的正向准许x偏差像素
    config['forward_num'] = int(100 * config['resize_p'])  # 前瞻像素行数
    config['road_redis'] = int(config['calibration']['x_len'] / 2)  # 道路半径像素个数


if __name__ == "__main__":
    init_config()
    # test_perspective_transform()  # 测试透视变换
    # sys.exit()
    cap = cv2.VideoCapture('test.mp4')
    frame_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * config['resize_p'])
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * config['resize_p'])
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter('output/test.mp4', fourcc, frame_fps, (frame_width, frame_height))
    mar_per, mar_per_rev = get_perspective_mar(frame_height, frame_width)
    while True:
        print("process: %.2f" % (cap.get(cv2.CAP_PROP_POS_FRAMES) / frame_count * 100))  # 进度指示
        success, frame = cap.read()

        if success:
            start = time.time()
            error = 0
            # +================核心================+ #
            frame_bina = base_transform(frame)  # 基本变换
            base_points = find_base_point(frame_bina)  # 寻找最初的赛道点
            lines = find_line(frame_bina, base_points)  # 根据基本点寻找线条
            roads = extract_road(frame_bina, lines, 1)  # 从所有线条中筛选出赛道线
            if len(roads):  # 若成功找到车道线
                road_vertical = get_perspective_road(roads[0], mar_per)  # 根据单线条找到透视变换图中的点
                center, road_fit = get_center(road_vertical, roads[0][0][1], redis=config['road_redis'])  # 多项式拟合获得距离中心线的偏差
                error = get_error(center, frame_width, config['forward_num'])  # 获取前n行平均偏差
            else:  # 若没有找到车道线
                pass
            # +================核心================+ #
            time_use = time.time() - start

            # ########### 效果展示 ############

            if len(roads):
                # 展示透视变换后效果图
                frame_show = np.dstack((frame_bina, frame_bina, frame_bina))
                for p in road_vertical:
                    cv2.circle(frame_show, (int(p[1]), int(p[0])), 2, (255, 255, 0), -1)
                for p in road_fit:
                    cv2.circle(frame_show, (int(p[1]), int(p[0])), 2, (0, 0, 255), -1)
                for i, c in enumerate(center):
                    if i < config['forward_num']:
                        cv2.circle(frame_show, (int(c[1]), int(c[0])), 2, (0, 255, 255), -1)
                    else:
                        cv2.circle(frame_show, (int(c[1]), int(c[0])), 2, (255, 0, 255), -1)
                cv2.putText(frame_show, "fps:%.1f " % (1 / (time_use + 0.0001)) + "err:%.1f" % error,
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow('test', frame_show)  # 视频显示
                cv2.waitKey()
            # 展示原效果图
            frame_show = np.dstack((frame_bina, frame_bina, frame_bina))
            frame_show = get_show_frame(frame_show, base_points, lines, roads, road_fit, center, mar_per_rev)
            cv2.putText(frame_show, "fps:%.1f " % (1 / (time_use + 0.0001)) + "err:%.1f" % error,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('test', frame_show)  # 视频显示
            cv2.waitKey()
            # 视频写入
            # writer.write(frame_show)
        else:
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
