import numpy as np
import rasterio as rs
from rasterio.plot import show
from rasterio.windows import Window
import xarray as xr
import matplotlib.pyplot as plt
from pyproj import Transformer, CRS
import cv2
import os
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


def interest_area(image_path, lat, long, square_width):
    #image_path = r'./data/talhao_101035/tci20201010.tif'
    with rs.open(image_path) as rs_img:

        crs_img = CRS.from_user_input(rs.crs.CRS(rs_img.crs))
        crs_WSG84 = CRS("WGS84")
        coord_transformer = Transformer.from_crs(crs_from=crs_WSG84, crs_to=crs_img)
        res = coord_transformer.transform(long, lat)

        x = res[0]
        y = res[1]

        x_factor = rs_img.width / (rs_img.bounds.right - rs_img.bounds.left)
        y_factor = rs_img.height / (rs_img.bounds.top - rs_img.bounds.bottom)

        x_proj = (x - rs_img.bounds.left) * x_factor
        y_proj = (rs_img.bounds.top - y ) * y_factor
        # recortando um quadrado usando a latlong indicada como ponto central
        win_interest = Window(col_off=x_proj - (square_width / 2),
                              row_off=y_proj - (square_width / 2),
                              width=square_width,
                              height=square_width)

        kwargs = rs_img.meta.copy()
        # mantendo os meta dados
        kwargs.update({
            'height': win_interest.height,
            'width': win_interest.width,
            'transform': rs.windows.transform(win_interest, rs_img.transform)})

        win_result = rs_img.read(window=win_interest)
        #show(win_result)


        with rs.open(os.path.split(image_path)[0] + "/interest_"+ os.path.split(image_path)[1], 'w', **kwargs) as dst:
            dst.write(rs_img.read(window=win_interest))

        print("")
    return win_result

# rs.windows.from_bounds(left=-54.5,bottom=-15.5,right=-54, top=-15, transform=img.transform)

# talhão 101035
#talhao_101035 = interest_area(r'./data/talhao_101035/b0420201015.tif', -54.521091,-15.24826, 500)

# talhão 100767
# talhao_100767 = interest_area(r'./data/talhao_100767/b0820201015.tif', -54.773159, -15.228314, 500)

#talhao_103330 = interest_area(r'./data/talhao_103330/b0420201217.tif', -54.93358, -13.25332, 500)

# talhao_102996 = interest_area(r'./data/talhao_102996/b0420201013.tif', -56.0276, -11.74378, 500)

talhao_102939 = interest_area(r'./data/talhao_102939/b0420201105.tif', -59.513273,-14.981855, 500)


print("")


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
line_image = np.copy(img8) * 0
img_eq = cv2.equalizeHist(img8)
img_blur = cv2.GaussianBlur(img8, (5,5), 0 )
img_lap = cv2.Laplacian(img8, cv2.CV_64F)
img_lap = np.uint8(np.absolute(img_lap))

plt.imshow(img8,cmap = 'gray')

# plt.imshow(img_lap,cmap = 'gray')
# plt.show()
#
edges = cv2.Canny(img8,10,50)
plt.imshow(edges,cmap = 'gray')
plt.show()
#
# plt.imshow(img_eq,cmap = 'gray')
# edgeseq = cv2.Canny(img_eq,50,200)
# plt.imshow(edgeseq,cmap = 'gray')
# plt.show()




#
# xrtest = xr.open_rasterio(fp)
# print("")

# fp2 = r'./data/talhao_101035/b0320201010.tif'
# img2 = rs.open(fp2)
#
# print(img2.count)
# print(img2.width, img2.height)
# print(img2.crs)
# show(img2)

# import georaster
# import matplotlib.pyplot as plt
# # Use SingleBandRaster() if image has only one band
# img = georaster.MultiBandRaster('GeoTiff_Image.tif')
# # img.r gives the raster in [height, width, band] format
# # band no. starts from 0
# plt.imshow(img.r[:,:,2])



