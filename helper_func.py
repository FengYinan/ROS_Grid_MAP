import numpy as np
import math
import cv2 as cv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def prob_to_logodds(prob):
    ## Assuming that prob is a scalar
    prob = np.array(prob,float)
    logOdds = np.log(prob/(1-prob))
    return np.asscalar(logOdds)

def logOdds_to_prob(logOdds):
    ## Assuming that logOdds is a matrix
    p = 1 - 1/(1 + np.exp(logOdds))
    return p

def swap(a,b):
    x,y = b,a
    return x,y

def bresenham(x0, y0, x1, y1):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).

    Input coordinates should be integers.

    The result will contain both the start and the end point.
    """
    dx = int(x1 - x0)
    dy = int(y1 - y0)

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    for x in range(dx + 1):
        yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy

def pose_world_to_map(pntspose,gridSize, map_size):
    pntsWorld = pntspose.copy()
    pntsWorld[-1] = 1
    tran = v2t([map_size/2 * gridSize, map_size/2 * gridSize, np.pi/2])
    pntsWorld = np.dot(tran, pntsWorld)
    pntsMap = pntsWorld[:2] / gridSize
    return np.around(pntsMap[:2]).astype(np.int)

def laser_world_to_map(laserEndPnts, gridSize, map_size):
    tran = v2t([map_size/2 * gridSize, map_size/2 * gridSize, np.pi/2])
    laserEndPnts_map = np.dot(tran, laserEndPnts)
    pntsMap = laserEndPnts_map[:2]/gridSize
    a = np.array(pntsMap[0] < map_size)
    b = np.array(pntsMap[0] >= 0)
    c = np.array(pntsMap[1] < map_size)
    d = np.array(pntsMap[1] >= 0)
    pntsMap = pntsMap[:, a & b & c & d]
    return np.around(pntsMap[:2]).astype(np.int)

def v2t(v):
    x = v[0]
    y = v[1]
    th = v[2]
    trans = np.array([[np.cos(th), -np.sin(th), x],[np.sin(th), np.cos(th), y],[0, 0, 1]])
    return trans

def laser_to_xy(rl, gridSize, subsample=False):
    numBeams = len(rl.ranges)
    maxRange = rl.range_max

    distance = np.array(rl.ranges)
    lins = np.linspace(rl.angle_min, rl.angle_max, numBeams)

    if subsample:
        distance = np.min(distance.reshape([-1,10]),axis=1)
        lins = np.mean(lins.reshape([-1,10]),axis=1)
        numBeams = int(numBeams/10)

    index = np.where(distance<maxRange)

    distance = np.clip(distance, a_max=maxRange+gridSize, a_min=0)
    angle = np.stack([np.cos(lins), np.sin(lins)])

    points = distance * angle
    points = np.concatenate([points, np.ones([1,numBeams])])

    return points, index

def yawFromQuaternion(orientation):
    return math.atan2((2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)),
                      (1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)))

def l2_norm(x, y, goalX, goalY):
    return math.sqrt(pow((goalX - x), 2) + pow((goalY - y), 2))

def dis2d(point1, point2):
    return math.sqrt(pow((point1.x - point2.x), 2) + pow((point1.y - point2.y),2))

def resample(weight, p_list):
    sample = []

    ind = [i for i in range(len(weight))]

    weight = np.array(weight)
    weight = weight / np.sum(weight)
    ind = np.random.choice(ind, len(ind), p=weight)

    for i in ind:
        sample.append(p_list[i])

    return sample

def particle_mean(p_list, map_size):
    pose = np.array([0.,0.,0.])
    map = np.ones((map_size, map_size))

    for p in p_list:
        pose += p.pose
        map += p.grid_map

    pose /= len(p_list)
    map /= len(p_list)

    return pose, map

def laser_to_pixel(scan, image, P):
    # Only detect the range of about 50 degrees straight ahead
    points = laser_to_xy(scan, 0.1)

    #x, y is the pixel position
    brg = []

    for p in points.transpose():
        if np.linalg.norm(p[:2]) < scan.range_max:
            pose = np.array([p[0],p[1],p[2],1])
            l = np.dot(P, pose)
            x = int(l[0]/l[2])
            y = int(l[1]/l[2])
            brg.append(image[x,y])

    return brg

class ShapeAnalysis:
    def __init__(self):
        self.shapes = {'triangle': 0, 'rectangle': 0, 'polygons': 0, 'circles': 0}

    def analysis(self, frame):
        h, w, ch = frame.shape
        result = np.zeros((h, w, ch), dtype=np.uint8)
        # 二值化图像
        print("start to detect lines...\n")
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        cv.imshow("input image", frame)

        out_binary, contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cnt in range(len(contours)):
            # 提取与绘制轮廓
            cv.drawContours(result, contours, cnt, (0, 255, 0), 2)

            # 轮廓逼近
            epsilon = 0.01 * cv.arcLength(contours[cnt], True)
            approx = cv.approxPolyDP(contours[cnt], epsilon, True)

            # 分析几何形状
            corners = len(approx)
            shape_type = ""
            if corners == 3:
                count = self.shapes['triangle']
                count = count+1
                self.shapes['triangle'] = count
                shape_type = "三角形"
            if corners == 4:
                count = self.shapes['rectangle']
                count = count + 1
                self.shapes['rectangle'] = count
                shape_type = "矩形"
            if corners >= 10:
                count = self.shapes['circles']
                count = count + 1
                self.shapes['circles'] = count
                shape_type = "圆形"
            if 4 < corners < 10:
                count = self.shapes['polygons']
                count = count + 1
                self.shapes['polygons'] = count
                shape_type = "多边形"

            # 求解中心位置
            mm = cv.moments(contours[cnt])
            cx = int(mm['m10'] / mm['m00'])
            cy = int(mm['m01'] / mm['m00'])
            cv.circle(result, (cx, cy), 3, (0, 0, 255), -1)

        cv.imshow("Analysis Result", self.draw_text_info(result))
        cv.imwrite("test-result.png", self.draw_text_info(result))
        return self.shapes

    def draw_text_info(self, image):
        c1 = self.shapes['triangle']
        c2 = self.shapes['rectangle']
        c3 = self.shapes['polygons']
        c4 = self.shapes['circles']
        cv.putText(image, "triangle: "+str(c1), (10, 20), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        cv.putText(image, "rectangle: " + str(c2), (10, 40), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        cv.putText(image, "polygons: " + str(c3), (10, 60), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        cv.putText(image, "circles: " + str(c4), (10, 80), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        return image

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def map_process(csv = 'map_test1.csv'):
    t_us = np.loadtxt(csv, delimiter=",")
    t_us = t_us[200:800, 200:800]

    g = np.gradient(t_us)
    g = g[0] + g[1]
    g = normalize(g)
    amax = np.amax(g)
    amin = np.amin(g)
    g = (g - amin) / (amax - amin) * 255
    g = cv.GaussianBlur(g.astype(np.uint8), (3, 3), 1.5)
    cv.imwrite("input_image.png", g)

    t_us = normalize(t_us)
    amax = np.amax(t_us)
    amin = np.amin(t_us)
    t_us = (t_us - amin) / (amax - amin) * 255
    t_us = cv.GaussianBlur(t_us.astype(np.uint8), (3, 3), 1.5)
    t_us = t_us[20:-20, 20:-20]
    th, bw = cv.threshold(t_us, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    edges = cv.Canny(t_us, th / 2, th)
    cv.imwrite("result.png", edges)
    # plt.imshow(edges)
    # plt.show()

    row, column = np.where(edges > 0)
    data = np.stack([row, column], axis=0).transpose()
    kmeans = KMeans(n_clusters=4, random_state=0).fit(data)
    center = np.around(kmeans.cluster_centers_)

    point = np.array([[3, 10, -8, 4.5], [-4, 5, 7, 2.5], [1, 1, 1, 1]])
    tran = v2t([edges.shape[0] / 2 * 0.05, edges.shape[0] / 2 * 0.05, np.pi / 2])
    laserEndPnts_map = np.dot(tran, point)
    # plt.figure()
    # plt.scatter(center.transpose()[0]*0.05, center.transpose()[1]*0.05, label='map')
    # plt.scatter(laserEndPnts_map[0], laserEndPnts_map[1], label='real')
    # plt.legend()
    # plt.savefig('distortion2.jpg')
    # plt.show()
    # print(np.linalg.norm(point[:2] - center.transpose())*0.05)

    tran = np.linalg.inv(tran)
    center *= 0.05
    center = np.concatenate([center, np.ones([center.shape[0], 1])], axis=1)
    laserEndPnts_map2 = np.dot(tran, center.transpose())
    plt.figure()
    plt.scatter(point[0], point[1], label='real')
    plt.scatter(laserEndPnts_map2[0], laserEndPnts_map2[1], label='estimate')
    plt.legend()
    plt.savefig('distortion2.jpg')
    #plt.show()

    # g = cv.cvtColor(g, cv.COLOR_GRAY2BGR)
    # shape = ShapeAnalysis()
    # result, x, y = shape.analysis(g)
    # print(x, y)
