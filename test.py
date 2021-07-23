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
import fiona
import rasterio.mask
from rasterio.plot import show
from rasterio.windows import Window
import xarray as xr
import matplotlib.pyplot as plt
from pyproj import Transformer, CRS
import cv2
import sys


sys.setrecursionlimit(50000)


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

def mask_vector(image_path, vector_path):
    with fiona.open(vector_path, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with rasterio.open(image_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, invert=True, filled=True)
        out_meta = src.meta
        kwargs = out_meta

    show(out_image)

    with rs.open(os.path.split(image_path)[0] + "/ground_truth_" + os.path.split(image_path)[1], 'w', **kwargs) as dst:
        dst.write(out_image)


count = 0

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
                    if image_matrix[line][column] == 0:
                        # object_list[-1] += 1
                        find_object((line, column), image_matrix, object_list)
    return object_list

def find_object_kernel(seed, image_matrix, object_list):
    m_img = image_matrix[seed[0]-1:seed[0]+2,seed[1]-1:seed[1]+2]
    object_list[seed] = True

    k1 = np.array([[0,1,0],[0,0,1],[0,0,0]])
    k2 = np.array([[0,1,0],[1,0,0],[0,0,0]])
    k3 = np.array([[0,0,0],[0,0,1],[0,1,0]])
    k4 = np.array([[0,0,0],[1,0,0],[0,1,0]])

    remove = []
    if ((m_img * k1).sum() > 250):
        remove.append((seed[0]-1, seed[1]+1))

    if ((m_img * k2).sum() > 250):
        remove.append((seed[0]-1, seed[1]-1))

    if ((m_img * k3).sum() > 250):
        remove.append((seed[0]+1, seed[1]+1))

    if ((m_img * k4).sum() > 250):
        remove.append((seed[0]+1, seed[1]-1))

    line_ini = seed[0] - 1
    line_end = line_ini + 3

    col_ini = seed[1] - 1
    col_end = col_ini + 3

    for line in range(line_ini, line_end):
        for column in range(col_ini, col_end):
            if column != seed[1] or line != seed[0]:
                if (line, column) not in object_list.keys() and (line, column) not in remove:
                    if image_matrix[line][column] == 0:
                        find_object_kernel((line, column), image_matrix, object_list)
    return object_list

def segment_img(img_src_path):
    img_source = cv2.imread(img_src_path, 0)
    for i in range(500):
        img_source[0,i] = 1
        img_source[499, i] = 1
        img_source[i, 0] = 1
        img_source[i, 499] = 1

    img_result = np.copy(img_source) * 0
    obj = {}
    object_list = find_object_kernel((250,250), img_source, obj)
    for key in obj.keys():
        img_result[key[0], key[1]] = 1

    show(img_result)
    cv2.imwrite(os.path.split(img_src_path)[0] + "/seg_" + os.path.split(img_src_path)[1], img_result)

    return ""

def calc_IoU(ground_truth_src_path, result_src_path):
    img_ground_truth = cv2.imread(ground_truth_src_path, 0)
    img_result_path = cv2.imread(result_src_path, 0)

    intersection = np.logical_and(img_ground_truth, img_result_path)
    union = np.logical_or(img_ground_truth, img_result_path)
    iou = np.sum(intersection) / np.sum(union)
    accuracy = (img_result_path==img_ground_truth).sum()/250000
    true_positives = ((img_result_path==1) & (img_ground_truth==1)).sum()
    false_positives = ((img_result_path==1) & (img_ground_truth==0)).sum()
    false_negatives = ((img_result_path==0) & (img_ground_truth==1)).sum()

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * ((precision * recall) / (precision + recall))
    print("iou", iou)
    print("accuracy", accuracy)
    print("precision", precision)
    print("recall", recall)
    print("f1", f1)
    return iou

# rs.windows.from_bounds(left=-54.5,bottom=-15.5,right=-54, top=-15, transform=img.transform)

# talhao_101035 = interest_area(r'./data/machine_learning/tci_pivot.tif', -54.456796,-15.382209, 500)
# talhao_101035 = interest_area(r'./data/machine_learning/b04_pivot.tif', -54.456796,-15.382209, 500)
# talhao_101035 = interest_area(r'./data/machine_learning/b08_pivot.tif', -54.456796,-15.382209, 500)


# talhão 101035
#talhao_101035 = interest_area(r'./data/talhao_101035/b04.tif', -54.521091,-15.24826, 500)

# talhão 100767
# talhao_100767 = interest_area(r'./data/talhao_100767/b08.tif', -54.773159, -15.228314, 500)

#talhao_103330 = interest_area(r'./data/talhao_103330/b0420201217.tif', -54.93358, -13.25332, 500)

# talhao_102996 = interest_area(r'./data/talhao_102996/b0420201013.tif', -56.0276, -11.74378, 500)

#talhao_102939 = interest_area(r'./data/talhao_102939/b0420201105.tif', -59.513273,-14.981855, 500)

#mask_vector("./data/talhao_101035/interest_tci20201015.tif", "./data/talhao_101035/101035.shp.zip")
#mask_vector("./data/talhao_103330/interest_tci20201217.tif", "./data/talhao_103330/103330.shp.zip")
#mask_vector("./data/talhao_102939/interest_b0420201105.tif", "./data/talhao_102939/102939.shp.zip")

# calc_IoU("./data/talhao_101035/seg_ground_truth_interest_tci20201015.tif", "./data/talhao_101035/seg_result.tif")
# iou 0.9469583458195985
# accuracy 0.997876
# precision 0.9904921115870859
# recall 0.9556451612903226
# f1 0.9727566569185778

# calc_IoU("./data/talhao_101035/seg_ground_truth_interest_tci20201015.tif", "./data/talhao_101035/seg_result_hough.tif")
# iou 0.332661087022285
# accuracy 0.947056
# precision 0.39958817829457366
# recall 0.6651209677419355
# f1 0.49924334140435833

# calc_IoU("./data/talhao_101035/seg_ground_truth_interest_tci20201015.tif", "./data/talhao_101035/seg_result_hough_grid.tif")
# iou 0.7625
# accuracy 0.990576
# precision 1.0
# recall 0.7625
# f1 0.8652482269503545

# calc_IoU("./data/talhao_103330/seg_ground_truth_interest_tci20201217.tif", "./data/talhao_103330/seg_result.tif")
# iou 0.9571577071577072
# accuracy 0.997692
# precision 0.9966754291015927
# recall 0.9602234636871508
# f1 0.9781099434728175

# calc_IoU("./data/talhao_103330/seg_ground_truth_interest_tci20201217.tif", "./data/talhao_103330/seg_result_hough.tif")
# iou 0.9270763500931098
# accuracy 0.996084
# precision 1.0
# recall 0.9270763500931098
# f1 0.9621584012987515

# calc_IoU("./data/talhao_103330/seg_ground_truth_interest_tci20201217.tif", "./data/talhao_103330/seg_result_hough_grid.tif")
# iou 0.8524212715389186
# accuracy 0.992052
# precision 0.9966134074331365
# recall 0.8548975791433892
# f1 0.9203319834810152

# calc_IoU("./data/talhao_102939/seg_ground_truth_interest_b0420201105.tif", "./data/talhao_102939/result.tif")
# iou 0.024108
# accuracy 0.024108
# precision 0.024108
# recall 1.0
# f1 0.04708097192874189

# calc_IoU("./data/talhao_102939/seg_ground_truth_interest_b0420201105.tif", "./data/talhao_102939/seg_result_hough.tif")
# iou 0.00016592002654720425
# accuracy 0.975896
# precision 1.0
# recall 0.00016592002654720425
# f1 0.0003317850033178501

# calc_IoU("./data/talhao_102939/seg_ground_truth_interest_b0420201105.tif", "./data/talhao_102939/seg_result_hough_grid.tif")
# iou 0.00016592002654720425
# accuracy 0.975896
# precision 1.0
# recall 0.00016592002654720425
# f1 0.0003317850033178501

# segment_img("./data/talhao_101035/result_hough_grid.tif")
# segment_img("./data/talhao_101035/result_hough.tif")
# segment_img("./data/talhao_101035/result.tif")
# segment_img("./data/talhao_101035/ground_truth_interest_tci20201015.tif")
#
# segment_img("./data/talhao_103330/result_hough.tif")
# segment_img("./data/talhao_103330/result_hough_grid.tif")
# segment_img("./data/talhao_103330/result.tif")
# segment_img("./data/talhao_103330/ground_truth_interest_tci20201217.tif")
#
# segment_img("./data/talhao_102939/result.tif")
# segment_img("./data/talhao_102939/result_hough.tif")
# segment_img("./data/talhao_102939/result_hough_grid.tif")
# segment_img("./data/talhao_102939/ground_truth_interest_b0420201105.tif")

print("")

#

#
#
# img = cv2.imread("./data/talhao_101035/interest_b0420201015.tif", -1)
#
# img8 = cv2.convertScaleAbs(img, alpha=0.03)
# line_image = np.copy(img8) * 0
# img_eq = cv2.equalizeHist(img8)
# img_blur = cv2.GaussianBlur(img8, (5,5), 0 )
# img_lap = cv2.Laplacian(img8, cv2.CV_64F)
# img_lap = np.uint8(np.absolute(img_lap))
#
# plt.imshow(img8,cmap = 'gray')
#
# # plt.imshow(img_lap,cmap = 'gray')
# # plt.show()
# #
# edges = cv2.Canny(img8,10,50)
# plt.imshow(edges,cmap = 'gray')
# plt.show()
# #
# # plt.imshow(img_eq,cmap = 'gray')
# # edgeseq = cv2.Canny(img_eq,50,200)
# # plt.imshow(edgeseq,cmap = 'gray')
# # plt.show()
#
#
#
#
# #
# # xrtest = xr.open_rasterio(fp)
# # print("")
#
# # fp2 = r'./data/talhao_101035/b0320201010.tif'
# # img2 = rs.open(fp2)
# #
# # print(img2.count)
# # print(img2.width, img2.height)
# # print(img2.crs)
# # show(img2)
#
# # import georaster
# # import matplotlib.pyplot as plt
# # # Use SingleBandRaster() if image has only one band
# # img = georaster.MultiBandRaster('GeoTiff_Image.tif')
# # # img.r gives the raster in [height, width, band] format
# # # band no. starts from 0
# # plt.imshow(img.r[:,:,2])
#
#
#
