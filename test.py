import numpy as np
import rasterio as rs
from rasterio.plot import show
from rasterio.windows import Window
import xarray as xr
import matplotlib.pyplot as plt
import pyproj

fp = r'./data/talhao_101035/b0320201010.tif'
img = rs.open(fp)

print(img.count)
print(img.width, img.height)
print(img.crs)


# rs.windows.from_bounds(left=-54.5,bottom=-15.5,right=-54, top=-15, transform=img.transform)

print("")
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



