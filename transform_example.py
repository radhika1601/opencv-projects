from transform import four_point_transform
import numpy as np
import argparse
import cv2

#construct argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
ap.add_argument("-c", "--coords", help = "comma seperated list of source points")
args = vars(ap.parse_args())

#load image and source soordinates
image = cv2.imread(args["image"])
points = np.array(eval(args["coords"]), dtype = "float32")

#apply four point transform to get top down view of the image
warped = four_point_transform(image, points)

cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)

