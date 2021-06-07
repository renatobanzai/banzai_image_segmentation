import numpy as np
import rasterio as rs
from rasterio.plot import show
from rasterio.windows import Window
import xarray as xr
import matplotlib.pyplot as plt
from pyproj import Transformer, CRS
import cv2


def interest_area(image_path, lat, long, square_width):
    #image_path = r'./data/talhao_101035/tci20201010.tif'
    rs_img = rs.open(image_path)
    cv_img = cv2.imread(image_path)

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

    win_interest = Window(col_off=x_proj - (square_width / 2),
                          row_off=y_proj - (square_width / 2),
                          width=square_width,
                          height=square_width)

    win_result = rs_img.read(window=win_interest)
    show(win_result)
    print("")
    return win_result

# rs.windows.from_bounds(left=-54.5,bottom=-15.5,right=-54, top=-15, transform=img.transform)

#img = cv2.imread("./data/talhao_101035/tci20201015_interest_area.png", 0)
#img = cv2.imread("./data/talhao_101035/ndvi.tiff", 0)
#img = cv2.imread("./data/talhao_100767/tci20201015_interest_area.png", 0)
img = cv2.imread("./data/talhao_101035/interest_tci20201015.tif", 0)
edges = cv2.Canny(img,100,200)
plt.imshow(edges,cmap = 'gray')
#plt.imshow(img,cmap = 'gray')
plt.show()




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



