import numpy as np
import rasterio as rs
from rasterio.plot import show
from rasterio.windows import Window
import xarray as xr
import matplotlib.pyplot as plt
from pyproj import Transformer, CRS
import cv2
import sys
import math
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


_rho = 1  # distance resolution in pixels of the Hough grid
_theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 50  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, _rho, _theta, threshold, np.array([]), min_line_length, max_line_gap)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

# Draw the lines on the  image
lines_edges = cv2.addWeighted(img8, 0.8, line_image, 1, 0)
plt.imshow(lines_edges)
plt.show()

_rho = 1  # distance resolution in pixels of the Hough grid
_theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 50  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments



lines = cv2.HoughLines(edges, _rho, _theta, threshold)

for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    # x1 stores the rounded off value of (r * cos(theta) - 1000 * sin(theta))
    x1 = int(x0 + 1000 * (-b))
    # y1 stores the rounded off value of (r * sin(theta)+ 1000 * cos(theta))
    y1 = int(y0 + 1000 * (a))
    # x2 stores the rounded off value of (r * cos(theta)+ 1000 * sin(theta))
    x2 = int(x0 - 1000 * (-b))
    # y2 stores the rounded off value of (r * sin(theta)- 1000 * cos(theta))
    y2 = int(y0 - 1000 * (a))
    cv2.line(img8, (x1, y1), (x2, y2), (0, 0, 255), 2)




# Draw the lines on the  image
plt.imshow(img8)
plt.show()

grid_size = 20
img_grid = np.copy(edges)
height, width = img_grid.shape

for x in range(0, width - 1, grid_size):
    cv2.line(img_grid, (x, 0), (x, height), 255, 1)

for y in range(0, width - 1, grid_size):
    cv2.line(img_grid, (0, y), (width, y), 255, 1)

plt.imshow(img_grid)
plt.show()
img_grid

_rho = 1  # distance resolution in pixels of the Hough grid
_theta = np.pi / 1000  # angular resolution in radians of the Hough grid
threshold = 40  # minimum number of votes (intersections in Hough grid cell)

grid_size = 50
max_rho = math.sqrt(2 * grid_size ** 2)
img_grid = np.copy(edges)
height, width = img_grid.shape
new_image = np.copy(edges)

for line in range(0, height - 1, grid_size):
    for column in range(0, width - 1, grid_size):
        cell = img_grid[line:line + grid_size - 1, column:column + grid_size - 1]
        cell_copy = np.copy(cell)
        lines = cv2.HoughLines(cell, _rho, _theta, threshold)
        if not lines is None:
            for y in lines:
                rho, theta = y[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + max_rho * (-b))
                y1 = int(y0 + max_rho * (a))
                x2 = int(x0 - max_rho * (-b))
                y2 = int(y0 - max_rho * (a))
                cv2.line(cell, (x1, y1), (x2, y2), (255,0,0), 1)
        # Draw the lines on the  image
        img_cp = cv2.addWeighted(cell_copy, 0.8, cell, 1, 0)

        new_image[line:line + grid_size - 1, column:column + grid_size - 1] = img_cp

plt.imshow(new_image)
plt.show()

plt.imshow(edges)
plt.show()