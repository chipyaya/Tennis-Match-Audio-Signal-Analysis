import os
import cv2
import argparse
import numpy as np
from sklearn.cluster import dbscan


def make_out_dir(out_dir, dir_name, path):
    path = os.path.join(out_dir, dir_name, path)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def detect_edges(out_path, prefix, frame):
    # Convert BGR to HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([99, 105, 82], np.uint8)
    upper_blue = np.array([121, 184, 225], np.uint8)

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(frame_hsv, lower_blue, upper_blue)

    # cv2.imwrite(os.path.join(out_path, '{}-frame.png'.format(prefix)), frame)
    cv2.imwrite(os.path.join(out_path, '{}-mask.png'.format(prefix)), mask)

    # erosion and dilation
    kernel_size = 2
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel_size = 2
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # filter out top
    fil = np.zeros(frame.shape[:2], np.uint8)
    for i in range(70, frame.shape[0]):
        for j in range(frame.shape[1]):
            fil[i, j] = 255
    mask = cv2.bitwise_and(mask, mask, mask=fil)

    # edge detection
    edges = cv2.Canny(mask, 1000, 1500, apertureSize=3)

    # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imwrite(os.path.join(out_path, '{}-res.png'.format(prefix)), res)

    cv2.imwrite(os.path.join(out_path, '{}-mask-morph.png'.format(prefix)), mask)
    # cv2.imwrite(os.path.join(out_path, '{}-edges.png'.format(prefix)), edges)

    # calculate Hough lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)

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

    # cv2.imwrite(os.path.join(out_path,
        # '{}-houghlines.png'.format(prefix)), frame)
    return line_pts
 
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

def find_intersections(out_path, prefix, frame, line_pts):
    # find intersection points
    points = []
    for i in range(len(line_pts)):
        for j in range(len(line_pts)):
            if i >= j:
                continue
            tf, p = find_intersection(line_pts[i][0], line_pts[i][1],
                         line_pts[j][0], line_pts[j][1], frame.shape[:2])
            if tf == True:
               points.append(p)
               cv2.circle(frame, p, 5, (0, 255, 0), -1)
    if len(points) == 0:
        return None
    points = np.array(points)

    # cluster points and find centers
    core, lab = dbscan(points, eps=5, min_samples=3)
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
    cv2.imwrite(os.path.join(out_path,
        '{}-intersections.png'.format(prefix)), frame)
    return centers

def find_upper(centers):
    # print('centers:', centers)
    min_y = np.amin(np.array(centers), axis=0)[1]
    upper_left = min(centers, key=lambda p: p[1])
    delta_y = 7
    centers_filtered = list(filter(lambda p: p[1] <= min_y + delta_y, centers))
    upper_left = min(centers_filtered, key=lambda p: p[0])
    upper_right = max(centers_filtered, key=lambda p: p[0])
    return upper_left, upper_right

def find_border_points(centers):
    lower_right = max(centers, key=lambda p: p[0])
    lower_left = min(centers, key=lambda p: p[0])
    upper_left, upper_right = find_upper(centers)
    return [upper_left, upper_right, lower_right, lower_left]

def calc_transform_matrix(out_path, prefix, court, frame, border_points):
    # print('border_points:', border_points)
    # the detected border points are the source
    src_points = np.array(border_points, np.float32)
    # define destination border points on the court court 
    border_x_min, border_x_max = 25, 428
    border_y_min, border_y_max = 27, 248
    dst_points = np.array([(border_x_min, border_y_max),
                           (border_x_min, border_y_min),
                           (border_x_max, border_y_min),
                           (border_x_max, border_y_max)], np.float32)

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(frame, M, court.shape[::-1][1:])
    cv2.imwrite(os.path.join(out_path, '{}-warped.png'.format(prefix)), warped)
    return M

def get_human_position(prefix):
    file_name = '{}_human.txt'.format(prefix)

def get_ball_position(prefix):
    file_name = '{}_ball.txt'.format(prefix)

def transform_and_project(court, M, p, color):
    p = cv2.perspectiveTransform(M, p)
    cv2.circle(court, p, 10, color, -1)

def project(out_path, prefix, court, M, human_pos, ball_pos):
    # for pos in human_pos:
    #     transform_and_project(court, M, pos, (0, 0, 0))
    # transform_and_project(court, M, ball_pos, (255, 0, 0))
    # cv2.imwrite(os.path.join(out_path, '{}-projected.png'.format(prefix)), court)
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--frame_dir', type=str, default='../video2image/us19-images')
    parser.add_argument(
        '--out_dir', type=str, default='results')
    parser.add_argument(
        '--preprocessed_dir', type=str, default='preprocessed')
    parser.add_argument(
        '--projected_dir', type=str, default='projected')
    parser.add_argument(
        '--court_img', type=str, default='court/tennis-court.jpg')
    args = parser.parse_args()

    court = cv2.imread(args.court_img)
    for i, dir_name in enumerate(os.listdir(args.frame_dir)):
        dir_path = os.path.join(args.frame_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            prefix = file_name.split('.')[0]
            if not os.path.isfile(file_path) or not file_name.endswith('.png'):
                continue
            frame = cv2.imread(file_path)
            preprocessed_dir = make_out_dir(
                args.out_dir, dir_name, args.preprocessed_dir)
            projected_dir = make_out_dir(
                args.out_dir, dir_name, args.projected_dir)

            line_pts = detect_edges(preprocessed_dir, prefix, frame)
            if line_pts == None:
                print('{}: no edges detected'.format(file_path))
                continue
            centers = find_intersections(
                preprocessed_dir, prefix, frame, line_pts)
            if centers == None or len(centers) == 0:
                print('{}: no intersection found'.format(file_path))
                continue
            border_points = find_border_points(centers)
            M = calc_transform_matrix(
                preprocessed_dir, prefix, court, frame, border_points)
            human_pos = get_human_position(prefix)
            ball_pos = get_ball_position(prefix)
            project(projected_dir, prefix, court, M, human_pos, ball_pos)
            print('{}: projected'.format(file_path))