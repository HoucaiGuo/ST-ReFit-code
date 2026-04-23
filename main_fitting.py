import os
import rasterio
import numpy as np
from os.path import join
from tqdm import trange
from datetime import datetime
from joblib import Parallel, delayed
from scipy.interpolate import make_smoothing_spline
from general_utils import calculate_image_tiles, read_patch_from_image, DOY_str_2_num

np.set_printoptions(suppress=True, precision=5)

site = "IA-T15TVG"
tile = site.split(sep="-")[1]
year = 2023

# CLEAR parameters
ref_cloud_threshold = 30  # T, threshold to select reference images, default is 30
fill_cloud_threshold = 70  # threshold to determine if a patch can be filled without assistance of other patch, default is 70
DOY_cloud_threshold = 70  # threshold to determine if the image on DOY can be filled, default is 70
class_num = 20  # K, number of land-cover classes for Mini Batch K-Means classification, default is 20
common_num = 200  # C, number of common pixels retrieved by the K-D tree, default is 200
similar_num = 20  # S, number of similar pixels for residual compensation, default is 20

# VIPSTF-SW parameters
gap_len_tol = 10
scale_factor = 25  # scale factor between MODIS and Sentinel-2
fusion_similar_win_size = 25
fusion_similar_num = 25

# fitting parameters
lam = 10
filling_min_weight = 0.3
fusion_weight = 0.3
start_DOY = 91
end_DOY = 304

# image parameters
S2_image_suffix = ".jp2"
S2_mask_suffix = ".jp2"  # suffixes of images and masks
image_height = 5490
image_width = 5490
band_num = 6  # dimensions of the images
tile_height = 1100  # 550
tile_width = 1100  # 550  # height and width of the tiles
overlap_height = 0
overlap_width = 0  # height and width of overlapping area between two adjacent tiles
min_value = 0
max_value = 10000  # min and max values of the images

if site == "IA-T15TVG":
    r1, r2, c1, c2 = 23, 12, 23, 12
elif site == "MI-T17TLJ":
    r1, r2, c1, c2 = 24, 11, 25, 10
else:
    r1, r2, c1, c2 = 23, 12, 25, 10

# IO parameters
temp_folder = rf"F:\Sentinel-2-MCD43A4-Seamless\{site}\3F\temp-patch"  # temporary folder
S2_image_folder = rf"F:\Sentinel-2-MCD43A4-Seamless\{site}\data\S2_NBAR"
S2_mask_folder = rf"F:\Sentinel-2-MCD43A4-Seamless\{site}\data\S2_mask"
MCD43A4_folder = rf"F:\Sentinel-2-MCD43A4-Seamless\{site}\3F\MCD43A4_filled"  # folders of images and masks

# ROI parameters:
ROI_path = rf"F:\Sentinel-2-MCD43A4-Seamless\{site}\data\ROI\{site}-ROI.jp2"
ROI_cover_threshold = 80  # threshold to determine if a patch can be filled


def predict_pixel(pixel_vals, pixel_flags, weights,
                  DOYs, pred_DOYs,
                  spline_lambda):
    """
    S2_vals: shape=(band_num, S2_num)
    S2_flags: shape=(S2_num, )
    S2_DOYs: shape=(S2_num, )
    """
    cloud_free_flags = pixel_flags == 0
    weights[cloud_free_flags] = 1.0

    pred = np.empty(shape=(band_num, pred_DOYs.shape[0]), dtype=np.float32)
    for band_idx in range(band_num):
        spline = make_smoothing_spline(DOYs, pixel_vals[band_idx, :], w=weights, lam=spline_lambda)
        pred[band_idx] = spline(pred_DOYs)

    return pred


if __name__ == "__main__":
    """
    preprocessing
    """
    # calculate image tiles
    tile_row_num, tile_col_num, tile_start_rows, tile_end_rows, tile_start_cols, tile_end_cols = \
        calculate_image_tiles(image_height, image_width, tile_height, tile_width, overlap_height, overlap_width)

    info_folder = join(temp_folder, "info")

    # get the day of years (DOYs), names, and cloud covers of all Sentinel-2 images
    S2_DOYs_path = join(info_folder, "S2-DOYs.npy")
    S2_names_path = join(info_folder, "S2-names.npy")
    S2_cloud_covers_path = join(info_folder, f"S2-cloud-covers.npy")
    S2_DOYs = np.load(S2_DOYs_path)
    S2_names = np.load(S2_names_path)
    S2_cloud_covers = np.load(S2_cloud_covers_path)

    # read ROI
    if ROI_path:
        ROI_dataset = rasterio.open(ROI_path)
        ROI = ROI_dataset.read().squeeze()
        ROI_mask = ROI == 1
        ROI_dataset.close()
    else:
        ROI_mask = np.full(shape=(image_height, image_width), fill_value=True, dtype=np.bool_)

    # ROI covers of all tiles
    tile_ROI_covers_path = join(info_folder, f"tile-ROI-covers.npy")
    all_ROI_covers = np.load(tile_ROI_covers_path)

    # cloud covers of all patches
    patch_cloud_covers_path = join(info_folder, f"patch-cloud-covers.npy")
    all_cloud_covers = np.load(patch_cloud_covers_path)

    # time used for filling each patch
    fitting_time_path = join(info_folder, f"time-series-fitting-time.npy")
    if os.path.exists(fitting_time_path):
        all_fitting_time = np.load(fitting_time_path)
    else:
        all_fitting_time = np.zeros(shape=(tile_row_num, tile_col_num), dtype=np.float32)
        np.save(fitting_time_path, all_fitting_time)

    fitting_flags_path = join(info_folder, f"time-series-fitting-flags.npy")
    if os.path.exists(fitting_flags_path):
        all_fitting_flags = np.load(fitting_flags_path)
    else:
        all_fitting_flags = np.zeros(shape=(tile_row_num, tile_col_num), dtype=np.int8)
        np.save(fitting_flags_path, all_fitting_flags)

    print(all_fitting_time)
    print(all_fitting_flags)

    """
    process for each tile
    """
    for tile_row_idx in range(tile_row_num):
        for tile_col_idx in range(tile_col_num):
            if all_ROI_covers[tile_row_idx, tile_col_idx] < ROI_cover_threshold:
                print(f"Tile {tile_row_idx}-{tile_col_idx} has too many non-ROI pixels.")
                continue  # too many non-ROI pixels
            if all_fitting_flags[tile_row_idx, tile_col_idx] == 1:
                print(f"Tile {tile_row_idx}-{tile_col_idx} has been processed.")
                continue

            # tile information
            tile_cloud_covers = all_cloud_covers[tile_row_idx, tile_col_idx, :]
            tile_valid_flags = tile_cloud_covers <= fill_cloud_threshold
            DOY_valid_flags = S2_cloud_covers <= DOY_cloud_threshold
            tile_valid_flags = np.bitwise_and(tile_valid_flags, DOY_valid_flags)
            tile_valid_cloud_covers = tile_cloud_covers[tile_valid_flags]
            tile_valid_DOYs = S2_DOYs[tile_valid_flags]
            tile_valid_names = S2_names[tile_valid_flags]
            # tile_valid_num = np.count_nonzero(tile_valid_flags)

            # print(f"tile {tile_row_idx}-{tile_col_idx}, "
            #       f"valid DOYs {tile_valid_DOYs}")

            # tile information
            tile_row_start = tile_start_rows[tile_row_idx, tile_col_idx]
            tile_row_end = tile_end_rows[tile_row_idx, tile_col_idx]
            tile_col_start = tile_start_cols[tile_row_idx, tile_col_idx]
            tile_col_end = tile_end_cols[tile_row_idx, tile_col_idx]
            tile_actual_height = tile_row_end - tile_row_start
            tile_actual_width = tile_col_end - tile_col_start
            tile_folder = join(temp_folder, f"{tile_row_idx}-{tile_col_idx}")

            # initialize the weights
            weights = np.power(filling_min_weight, tile_valid_cloud_covers / fill_cloud_threshold)

            # check if there is fused image
            fuse_DOYs_path = join(tile_folder, f"fusion-DOYs_tol-{gap_len_tol}.npy")
            if os.path.exists(fuse_DOYs_path):
                fuse_DOYs = np.load(fuse_DOYs_path)
                fuse_num = fuse_DOYs.shape[0]
                tile_valid_DOYs = np.concatenate([tile_valid_DOYs, fuse_DOYs], axis=0)
                sort_indices = np.argsort(tile_valid_DOYs)
                tile_valid_DOYs = tile_valid_DOYs[sort_indices]

                tile_valid_cloud_covers = np.concatenate([tile_valid_cloud_covers,
                                                          np.full(fill_value=-1, shape=(fuse_num,), dtype=np.float32)],
                                                         axis=0)
                tile_valid_cloud_covers = tile_valid_cloud_covers[sort_indices]
                weights = np.concatenate([weights,
                                          np.full(fill_value=fusion_weight, shape=(fuse_num,), dtype=np.float32)],
                                         axis=0)
                weights = weights[sort_indices]

                fuse_names = []
                for fuse_idx in range(fuse_num):
                    fuse_DOY = fuse_DOYs[fuse_idx]
                    yyyydoy = f"{year}{fuse_DOY}"
                    date = datetime.strptime(yyyydoy, "%Y%j")
                    yyyymmdd = date.strftime("%Y%m%d")
                    fuse_name = f"S2_{yyyymmdd}_{fuse_DOY}_{tile}"
                    fuse_names.append(fuse_name)
                fuse_names = np.array(fuse_names)
                tile_valid_names = np.concatenate([tile_valid_names, fuse_names], axis=0)
                tile_valid_names = tile_valid_names[sort_indices]
            tile_valid_num = tile_valid_DOYs.shape[0]
            # print(tile_valid_DOYs)
            # print(tile_valid_cloud_covers)
            # print(weights)
            # print(tile_valid_names)

            print(f"tile {tile_row_idx}-{tile_col_idx}, "
                  f"valid DOYs {tile_valid_DOYs}")

            images = []
            # flag indicating state of the Sentinel-2 pixels
            # 0 is cloud-free,
            # 1 is gap-filled,
            # 2 is fused
            flags = []
            for valid_idx in range(tile_valid_num):
                valid_cloud_cover = tile_valid_cloud_covers[valid_idx]
                valid_name = tile_valid_names[valid_idx]
                if valid_cloud_cover == -1:  # fused
                    image_path = join(tile_folder,
                                      f"{valid_name}_{tile_row_idx}-{tile_col_idx}_"
                                      f"W-{fusion_similar_win_size}_S-{fusion_similar_num}_"
                                      f"tol-{gap_len_tol}_VIPSTF-SW{S2_image_suffix}")
                    dataset = rasterio.open(image_path)
                    profile = dataset.profile
                    image = dataset.read()
                    image = np.transpose(image, (1, 2, 0))
                    flag = np.full(fill_value=2, shape=(tile_actual_height, tile_actual_width), dtype=np.int32)
                elif 0 < valid_cloud_cover <= fill_cloud_threshold:  # gap-filled
                    image_path = join(tile_folder, f"{valid_name}_{tile_row_idx}-{tile_col_idx}_"
                                                   f"T-{ref_cloud_threshold}_K-{class_num}_"
                                                   f"C-{common_num}_S-{similar_num}_CLEAR{S2_image_suffix}")
                    dataset = rasterio.open(image_path)
                    profile = dataset.profile
                    image = dataset.read()
                    image = np.transpose(image, (1, 2, 0))
                    mask_path = join(S2_mask_folder, f"{valid_name}_mask{S2_mask_suffix}")
                    mask, _ = read_patch_from_image(mask_path, tile_row_start, tile_row_end,
                                                    tile_col_start, tile_col_end)
                    mask = mask.squeeze()
                    invalid_mask = mask != 0
                    valid_mask = mask == 0
                    flag = np.empty(shape=mask.shape, dtype=np.float32)
                    flag[valid_mask] = 0
                    flag[invalid_mask] = 1
                else:  # cloud-free
                    image_path = join(tile_folder, f"{valid_name}_{tile_row_idx}-{tile_col_idx}_"
                                                   f"T-{ref_cloud_threshold}{S2_image_suffix}")
                    dataset = rasterio.open(image_path)
                    profile = dataset.profile
                    image = dataset.read()
                    image = np.transpose(image, (1, 2, 0))
                    flag = np.full(fill_value=0, shape=(tile_actual_height, tile_actual_width), dtype=np.int32)
                images.append(image)
                flags.append(flag)
            images = np.stack(images, axis=3)
            flags = np.stack(flags, axis=2)

            # time-series fitting
            images_reshaped = images.reshape(tile_actual_height * tile_actual_width, band_num, tile_valid_num)
            flags_reshaped = flags.reshape(tile_actual_height * tile_actual_width, tile_valid_num)
            pred_DOYs = np.linspace(start_DOY, end_DOY, end_DOY - start_DOY + 1, endpoint=True, dtype=np.int32)
            pred_num = pred_DOYs.shape[0]

            t1 = datetime.now()
            results = (Parallel(n_jobs=-1, backend='loky', timeout=None)
                       (delayed(predict_pixel)
                        (images_reshaped[i, :, :], flags_reshaped[i, :], weights,
                         tile_valid_DOYs, pred_DOYs, lam)
                        for i in trange(tile_actual_height * tile_actual_width)))
            predictions = np.array(results).reshape(tile_actual_height, tile_actual_width, band_num, pred_num)
            predictions[predictions < min_value] = min_value
            predictions[predictions > max_value] = max_value
            t2 = datetime.now()
            time_span = t2 - t1
            total_minutes = time_span.total_seconds() / 60.0
            print(f"\tUsed {total_minutes:.2f} minutes.")

            for pred_idx in range(pred_num):
                prediction = predictions[:, :, :, pred_idx]
                pred_DOY = pred_DOYs[pred_idx]
                save_path = join(tile_folder, f"ST-ReFit_{tile_row_idx}-{tile_col_idx}-{pred_DOY}{S2_image_suffix}")
                prediction = prediction.transpose((2, 0, 1))
                dataset = rasterio.open(save_path, mode='w', **profile)
                dataset.write(prediction)
                dataset.close()

            all_fitting_time[tile_row_idx, tile_col_idx] = total_minutes
            np.save(fitting_time_path, all_fitting_time)
            all_fitting_flags[tile_row_idx, tile_col_idx] = 1
            np.save(fitting_flags_path, all_fitting_flags)
