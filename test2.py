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
img_eq = cv2.equalizeHist(img8)
img_blur = cv2.GaussianBlur(img8, (5,5), 0 )
img_lap = cv2.Laplacian(img8, cv2.CV_64F)
img_lap = np.uint8(np.absolute(img_lap))
#plt.imshow(img8,cmap = 'gray')

plt.imshow(img_lap,cmap = 'gray')
plt.show()

edges = cv2.Canny(img_lap,2,50)
plt.imshow(edges,cmap = 'gray')
plt.show()


edges = cv2.Canny(img8,10,50)
plt.imshow(edges,cmap = 'gray')
plt.show()





plt.imshow(img_eq,cmap = 'gray')
edgeseq = cv2.Canny(img_eq,50,200)
plt.imshow(edgeseq,cmap = 'gray')
plt.show()




