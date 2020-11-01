import cv2
import numpy as np
from sklearn.cluster import dbscan


def find_intersection(o1, p1, o2, p2, frame_size):
    def minus(p1, p2):
        return (p1[0] - p2[0], p1[1] - p2[1])

    x = minus(o2, o1)
    d1 = minus(p1, o1)
    d2 = minus(p2, o2)

    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-8:
        return [False, (0, 0)]
    t1 = (x[0] * d2[1] - x[1] * d2[0]) / cross
    rx = int(o1[0] + d1[0] * t1)
    ry = int(o1[1] + d1[1] * t1)
    tf = False
    if frame_size[1] > rx and rx > 0 and frame_size[0] > ry and ry > 0:
        tf = True
    return [tf, (rx, ry)]

def find_intersections(frame, idx, line_pts):
    # find intersection points
    points = []
    for i in range(len(line_pts)):
        for j in range(len(line_pts)):
            if i >= j:
                continue
            tf, r = find_intersection(line_pts[i][0], line_pts[i][1],
                         line_pts[j][0], line_pts[j][1], frame.shape[:2])
            if tf == True:
               points.append(r)
               cv2.circle(frame, r, 5, (0, 255, 0), -1)
    points = np.array(points)

    # cluster points and find centers
    core, lab = dbscan(points, eps=9, min_samples=6)
    centers = []
    for i in range(np.amax(lab) + 1):
        count = 0
        total = [0, 0]
        for p in range(len(points)):
            if lab[p] == i:
                count += 1
                total[0] += points[p][0]
                total[1] += points[p][1]
        total[0] = int(total[0] / count)
        total[1] = int(total[1] / count)
        centers.append(total)
        cv2.circle(frame, (total[0], total[1]), 10, (255, 0, 0), -1)
    cv2.imwrite(out_dir + '{0:03d}-houghlines-intersections.png'.format(idx), frame)
    return centers

def find_upper_right(centers, y):
    centers_filtered = list(filter(lambda p: p[1] <= y, centers))
    upper_right = max(centers_filtered, key=lambda p: p[0])
    return upper_right

def find_border_points(centers):
    lower_right = max(centers, key=lambda p: p[0])
    lower_left = min(centers, key=lambda p: p[0])
    upper_left = min(centers, key=lambda p: p[1])
    delta_y = 10
    upper_right = find_upper_right(centers, upper_left[1] + delta_y)
    return [upper_left, upper_right, lower_right, lower_left]

def calc_transform_matrix(frame, idx, centers, board):
    # find border points
    border_points = find_border_points(centers)
    src_points = np.array(border_points, np.float32)
    # define destination court border points
    border_x_min, border_x_max = 25, 428
    border_y_min, border_y_max = 27, 248
    dst_points = np.array([(border_x_min, border_y_max),
                           (border_x_min, border_y_min),
                           (border_x_max, border_y_min),
                           (border_x_max, border_y_max)], np.float32)

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(frame, M, board.shape[::-1][1:])
    cv2.imwrite(warp_out_dir + '{0:03d}-warped.png'.format(idx), warped)
    return M

def detect_edges(frame, idx):
    # Convert BGR to HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([102, 73, 145], np.uint8)
    upper_blue = np.array([123, 182, 175], np.uint8)

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(frame_hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imwrite(out_dir + '{0:03d}-frame.png'.format(idx), frame)
    cv2.imwrite(out_dir + '{0:03d}-mask.png'.format(idx), mask)
    # cv2.imwrite(out_dir + '{0:03d}-res.png'.format(idx), res)

    # erosion and dilation
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel_size = 10
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(out_dir + '{0:03d}-mask-erosion-dilation.png'.format(idx), mask)

    # edge detection
    edges = cv2.Canny(mask, 1000, 1500, apertureSize=3)

    cv2.imwrite(out_dir + '{0:03d}-mask-edges.png'.format(idx), edges)

    # calculate Hough lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 50)

    if lines is None:
        return None

    line_pts = []
    n_lines = 60
    for i in range(min(n_lines, len(lines))):
        for rho,theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(frame,(x1, y1), (x2, y2),(0, 0, 255), 2)
            line_pts.append([(x1, y1), (x2, y2)])
    cv2.imwrite(out_dir + '{0:03d}-houghlines.png'.format(idx), frame)
    return line_pts
 
if __name__ == '__main__':
    out_dir = 'out/'
    warp_out_dir = 'warped/'
    frame_path_format = '../video2image/us19-images/{0:03d}.png'
    start, end = 1, 210
    board = cv2.imread('img/tennis-court-board.jpg')


    for i in range(start, end):
        print(i)
        frame_path = frame_path_format.format(i)
        frame = cv2.imread(frame_path)
        line_pts = detect_edges(frame, i)
        if line_pts == None:
            print(frame_path + ': no edges detected')
            continue
        centers = find_intersections(frame, i, line_pts)
        if len(centers) == 0:
            print(frame_path + ': no intersections detected')
            continue
        M = calc_transform_matrix(frame, i, centers, board)