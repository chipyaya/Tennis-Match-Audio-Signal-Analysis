import cv2
import numpy as np
from sklearn.cluster import dbscan


# Input: A frame of basketball game
# Output: 
# 1. A transformation matrix that maps frame coordinate to tactic board coordinate
# 2. Warped frame

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

def find_intersections(frame, id, line_pts):
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
    cv2.imwrite('houghlines-intersections.jpg', frame)
    return centers

def calc_transform_matrix(frame, id, centers):
    print(centers)
    print(frame.shape)
    # order centers
    max_x = 0
    min_x = frame.shape[1]
    point0, point2 = -1, -1
    for i in range(len(centers)):
        if centers[i][0] > max_x:
            point0 = i
            max_x = centers[i][0]
        if centers[i][0] < min_x:
            point2 = i
            min_x = centers[i][0]
    max_y = 0
    min_y = frame.shape[0]
    point1, point3 = -1, -1
    for i in range(len(centers)):
        if i == point0 or i == point2:
            continue
        if centers[i][1] > max_y:
            point1 = i
            max_y = centers[i][1]
        if centers[i][1] < min_y:
            point3 = i
            min_y = centers[i][1]

    # define court point and calculate the transformation matrix
    court0, court1, court2, court3 = (190, 0), (190, 328), (0, 328), (0, 0)
    dst_points = np.array([court0, court1, court2, court3], np.float32)
    src_points = np.array([centers[point0], centers[point1], centers[point2], centers[point3]], np.float32)

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(frame, M, (500, 470))
    cv2.imwrite('warped.jpg', warped)
    return M

def detect_edges(frame):
    # Convert BGR to HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([102, 73, 145], np.uint8)
    upper_blue = np.array([123, 182, 175], np.uint8)

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(frame_hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imwrite('frame.jpg', frame)
    cv2.imwrite('mask.jpg', mask)
    cv2.imwrite('res.jpg', res)

    # erosion and dilation
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=3)

    cv2.imwrite('mask-erosion-dilation.jpg', mask)

    # edge detection
    edges = cv2.Canny(mask, 1000, 1500, apertureSize=3)

    cv2.imwrite('mask-edges.jpg', edges)

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
    cv2.imwrite('houghlines.jpg',frame)
    return line_pts
 
if __name__ == '__main__':
    frame_path_format = '../video2image/us19-images/{0:03d}.png'
    start, end = 2, 3

    for i in range(start, end):
        frame_path = frame_path_format.format(i)
        frame = cv2.imread(frame_path)
        line_pts = detect_edges(frame)
        if edges == None:
            print(frame_path + ': no edges detected')
        centers = find_intersections(frame, id, line_pts)
        # calc_transform_matrix(frame, i, centers)