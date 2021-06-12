import numpy as np
import rasterio as rs
from rasterio.plot import show
from rasterio.windows import Window
import xarray as xr
import matplotlib.pyplot as plt
from pyproj import Transformer, CRS
import cv2
import sys
sys.setrecursionlimit(25000)

def find_object(seed, image_matrix, object_list):
    background_color = 255
    line_ini = seed[0] - 1
    line_end = line_ini + 3

    col_ini = seed[1] - 1
    col_end = col_ini + 3
    object_list[seed] = True
    for line in range(line_ini, line_end):
        for column in range(col_ini, col_end):
            if column != seed[1] or line != seed[0]:
                if (line, column) not in object_list.keys():
                    if image_matrix[line][column] < 255:
                        find_object((line, column), image_matrix, object_list)
    return object_list


img = cv2.imread("./data/talhao_101035/interest_b0420201015.tif", -1)

img8 = cv2.convertScaleAbs(img, alpha=0.03)
line_image = np.copy(img8) * 0  # creating a blank to draw lines on

img_eq = cv2.equalizeHist(img8)
img_blur = cv2.GaussianBlur(img8, (5,5), 0 )
img_lap = cv2.Laplacian(img8, cv2.CV_64F)
img_lap = np.uint8(np.absolute(img_lap))
#plt.imshow(img8,cmap = 'gray')

edges = cv2.Canny(img8,10,50)
plt.imshow(edges,cmap = 'gray')
plt.show()


rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 50  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

# Draw the lines on the  image
lines_edges = cv2.addWeighted(img8, 0.8, line_image, 1, 0)
plt.imshow(lines_edges)
plt.show()

