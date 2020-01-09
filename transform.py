import numpy as np
import cv2

def order_points(points):
    """
    This functions returns four corners
    i.e. top-left, top-right, bottom-right, bottom-left
    return: ordered coordinates
    """
    rect = np.zeros((4,2), dtype = "float32")
    
    #top-left point has smallest sum
    #bottom-right point has largest sum
    s = points.sum(axis = 1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    #top-right point has smallest difference
    #bottom-left point has largest difference
    diff = np.diff(points, axis = 1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    
    return rect

def four_point_transform(image, points):
    """
    This functions tranforms the image into top dowm view 
    according to the four points that contain the ROI of image
    :param image:
    :param points:
    """

    rect = order_points(points)
    (tl, tr, br, bl) = rect

    #compute width of new image
    widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1]-bl[1])**2))
    widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1]-tl[1])**2))
    maxWidth = max(int(widthA), int(widthB))

    #compute height of new image
    heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1]-br[1])**2))
    heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1]-bl[1])**2))
    maxHeight = max(int(heightA), int(heightB))

    #set of destination points for "top down view"
    dst = np.array([
        [0, 0],
        [maxWidth-1, 0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]
        ], dtype = "float32")

    #compute perspective transform matrix and apply it
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))

    return warped

