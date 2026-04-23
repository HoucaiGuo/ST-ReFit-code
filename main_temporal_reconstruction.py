import os
import cv2
import rasterio
import numpy as np
from datetime import datetime
from os.path import join
from tqdm import trange
from general_utils import calculate_image_tiles, DOY_str_2_num
from VIPSTF_SW_utils import *
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

np.set_printoptions(suppress=True, precision=5)

# general parameters
site = "PA-T17TPG"
start_DOY = 91
end_DOY = 304  # [start_DOY, end_DOY]
year = 2023
tile = site.split(sep="-")[1]

# fuse parameters
gap_len_tol = 10
scale_factor = 25  # scale factor between MODIS and Sentinel-2
fusion_similar_win_size = 25
fusion_similar_num = 25

# IO parameters
temp_folder = rf"F:\Sentinel-2-MCD43A4-Seamless\{site}\3F\temp-patch"  # temporary folder
MCD43A4_folder = rf"F:\Sentinel-2-MCD43A4-Seamless\{site}\3F\MCD43A4_filled"  # folders of images and masks

# ROI parameters:
ROI_path = rf"F:\Sentinel-2-MCD43A4-Seamless\{site}\data\ROI\{site}-ROI.jp2"
ROI_cover_threshold = 30  # threshold to determine if a patch can be fused

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

# CLEAR parameters
ref_cloud_threshold = 30  # T, threshold to select reference images, default is 30
fill_cloud_threshold = 70  # threshold to determine if a patch can be filled without assistance of other patch, default is 70
DOY_cloud_threshold = 70  # threshold to determine if the image on DOY can be filled, default is 70
class_num = 20  # K, number of land-cover classes for Mini Batch K-Means classification, default is 20
common_num = 200  # C, number of common pixels retrieved by the K-D tree, default is 200
similar_num = 20  # S, number of similar pixels for residual compensation, default is 20

if site == "IA-T15TVG":
    r1, r2, c1, c2 = 23, 12, 23, 12
elif site == "MI-T17TLJ":
    r1, r2, c1, c2 = 24, 11, 25, 10
else:
    r1, r2, c1, c2 = 23, 12, 25, 10

band_names = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]

if __name__ == "__main__":
    """
    preprocessing
    """
    # calculate image tiles
    tile_row_num, tile_col_num, tile_start_rows, tile_end_rows, tile_start_cols, tile_end_cols = \
        calculate_image_tiles(image_height, image_width, tile_height, tile_width, overlap_height, overlap_width)
    print(f"Total {tile_row_num} * {tile_col_num} tiles!")

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

    """
    process for each tile
    """
    for tile_row_idx in range(tile_row_num):
        for tile_col_idx in range(tile_col_num):
            tile_ROI_cover = all_ROI_covers[tile_row_idx, tile_col_idx]
            if tile_ROI_cover < ROI_cover_threshold:
                continue  # too many non-ROI pixels

            # tile information
            tile_row_start = tile_start_rows[tile_row_idx, tile_col_idx]
            tile_row_end = tile_end_rows[tile_row_idx, tile_col_idx]
            tile_col_start = tile_start_cols[tile_row_idx, tile_col_idx]
            tile_col_end = tile_end_cols[tile_row_idx, tile_col_idx]
            tile_actual_height = tile_row_end - tile_row_start
            tile_actual_width = tile_col_end - tile_col_start
            tile_folder = join(temp_folder, f"{tile_row_idx}-{tile_col_idx}")
            tile_ROI_mask = ROI_mask[tile_row_start:tile_row_end, tile_col_start:tile_col_end]

            # select reference images for spatiotemporal fusion
            tile_cloud_covers = all_cloud_covers[tile_row_idx, tile_col_idx]
            tile_ref_flags = tile_cloud_covers <= fill_cloud_threshold
            DOY_valid_flags = S2_cloud_covers <= DOY_cloud_threshold  # use only DOY_cloud_cover <= 70
            tile_ref_flags = np.bitwise_and(tile_ref_flags, DOY_valid_flags)
            tile_ref_DOYs = S2_DOYs[tile_ref_flags]
            tile_ref_num = tile_ref_DOYs.shape[0]
            tile_ref_names = S2_names[tile_ref_flags]
            tile_ref_cloud_covers = tile_cloud_covers[tile_ref_flags]

            # calculate the length of temporal gaps
            start_DOYs = tile_ref_DOYs[:-1]
            end_DOYs = tile_ref_DOYs[1:]
            valid_start_DOY = start_DOYs[0]
            if valid_start_DOY > start_DOY:
                start_gap_len = valid_start_DOY - start_DOY
                start_gap_fuse_num = np.floor(start_gap_len / gap_len_tol).astype(np.int32)
                start_fuse_DOYs = np.linspace(start_DOY, valid_start_DOY, start_gap_fuse_num + 2, dtype=np.int32)[
                                  :-1]
            else:
                start_fuse_DOYs = np.empty(shape=(0,), dtype=np.int32)
            valid_end_DOY = end_DOYs[-1]
            if valid_end_DOY < end_DOY:
                end_gap_len = end_DOY - valid_end_DOY
                end_gap_fuse_num = np.floor(end_gap_len / gap_len_tol).astype(np.int32)
                end_fuse_DOYs = np.linspace(valid_end_DOY, end_DOY, end_gap_fuse_num + 2, dtype=np.int32)[1:]
            else:
                end_fuse_DOYs = np.empty(shape=(0,), dtype=np.int32)

            # get long temporal gaps
            gap_lens = end_DOYs - start_DOYs
            fuse_DOYs = np.empty(shape=(0,), dtype=np.int32)
            long_gap_flags = gap_lens > gap_len_tol
            long_gap_indices = long_gap_flags.nonzero()[0]
            for long_gap_idx in long_gap_indices:
                long_gap_start = start_DOYs[long_gap_idx]
                long_gap_end = end_DOYs[long_gap_idx]
                long_gap_len = gap_lens[long_gap_idx]
                gap_fuse_num = np.floor(long_gap_len / gap_len_tol).astype(np.int32)
                gap_fuse_DOYs = np.linspace(long_gap_start, long_gap_end, gap_fuse_num + 2, dtype=np.int32)[1:-1]
                fuse_DOYs = np.concatenate([fuse_DOYs, gap_fuse_DOYs], dtype=np.int32)
            tile_fuse_DOYs = np.concatenate([start_fuse_DOYs, fuse_DOYs, end_fuse_DOYs], dtype=np.int32)
            tile_fuse_num = tile_fuse_DOYs.shape[0]
            if tile_fuse_num == 0:
                continue
            tile_fuse_names = []
            for fuse_idx in range(tile_fuse_num):
                fuse_DOY = tile_fuse_DOYs[fuse_idx]
                yyyydoy = f"{year}{fuse_DOY}"
                date = datetime.strptime(yyyydoy, "%Y%j")
                yyyymmdd = date.strftime("%Y%m%d")
                fuse_name = f"S2_{yyyymmdd}_{fuse_DOY}_{tile}"
                tile_fuse_names.append(fuse_name)
            print(f"Tile {tile_row_idx}-{tile_col_idx}, "
                  f"There are {tile_ref_num} reference images, "
                  f"DOYs {tile_ref_DOYs}. \n"
                  f"\tThere are {tile_fuse_num} images that need to be fused. "
                  f"DOYs {tile_fuse_DOYs}")
            print(tile_fuse_names)

            # fusion DOYs
            fuse_DOYs_path = join(tile_folder, f"fusion-DOYs_tol-{gap_len_tol}.npy")
            np.save(fuse_DOYs_path, tile_fuse_DOYs)
            # if not os.path.exists(fuse_DOYs_path):
            #     np.save(fuse_DOYs_path, tile_fuse_DOYs)

            # fusion flags
            fusion_flags_path = join(tile_folder, f"fusion-flags_tol-{gap_len_tol}.npy")
            if os.path.exists(fusion_flags_path):
                tile_fusion_flags = np.load(fusion_flags_path)
            else:
                tile_fusion_flags = np.zeros(shape=(tile_fuse_num,), dtype=np.float32)
                np.save(fusion_flags_path, tile_fusion_flags)

            # time used for fusion
            fusion_time_path = join(tile_folder, f"fusion-time_tol-{gap_len_tol}.npy")
            if os.path.exists(fusion_time_path):
                tile_fusion_time = np.load(fusion_time_path)
            else:
                tile_fusion_time = np.zeros(shape=(tile_fuse_num,), dtype=np.float32)
                np.save(fusion_time_path, tile_fusion_time)

            # read MCD43A4 images
            file_names = os.listdir(MCD43A4_folder)
            M_DOYs = []
            M_images = []
            for file_idx in range(len(file_names)):
                file_name = file_names[file_idx]
                if file_name.endswith(".tif"):
                    splits = file_name.split(sep="_")
                    date = splits[1]
                    DOY_str = splits[2]
                    DOY = DOY_str_2_num(DOY_str)
                    if DOY in tile_ref_DOYs or DOY in tile_fuse_DOYs:
                        M_DOYs.append(DOY)
                        M_image_path = join(MCD43A4_folder, file_name)
                        M_image_dataset = rasterio.open(M_image_path)
                        M_image = M_image_dataset.read()
                        M_image = np.transpose(M_image, (1, 2, 0))
                        M_image = M_image.astype(np.float32)
                        M_images.append(M_image)
            M_images = np.stack(M_images, axis=3)
            M_DOYs = np.array(M_DOYs, dtype=np.uint16)
            print(M_DOYs)
            M_num = M_DOYs.shape[0]
            # subset
            M_row_start = (tile_row_start + r1) // scale_factor
            M_row_end = np.ceil((tile_row_end + r1) / scale_factor).astype(np.int32)
            M_col_start = (tile_col_start + c1) // scale_factor
            M_col_end = np.ceil((tile_col_end + r1) / scale_factor).astype(np.int32)
            M_images = M_images[M_row_start:M_row_end, M_col_start:M_col_end, :, :]

            # resize
            M_resize_rows = M_images.shape[0] * scale_factor
            M_resize_cols = M_images.shape[1] * scale_factor
            M_images = np.stack([cv2.resize(M_images[:, :, :, i],
                                            (M_resize_cols, M_resize_rows),
                                            interpolation=cv2.INTER_CUBIC)
                                 for i in range(M_num)], axis=3)
            # subset to match the extent of Sentinel-2
            M_row_start_1 = (tile_row_start + r1) % scale_factor
            M_row_end_1 = M_row_start_1 + tile_actual_height
            M_col_start_1 = (tile_col_start + c1) % scale_factor
            M_col_end_1 = M_col_start_1 + tile_actual_width
            M_images = M_images[M_row_start_1:M_row_end_1, M_col_start_1:M_col_end_1, :, :]
            # reference images and fuse images
            M_ref_images = M_images[:, :, :, np.isin(M_DOYs, tile_ref_DOYs)]
            M_fuse_images = M_images[:, :, :, np.isin(M_DOYs, tile_fuse_DOYs)]

            # read Sentinel-2 images
            S2_ref_images = []
            for ref_idx in range(tile_ref_num):
                ref_name = tile_ref_names[ref_idx]
                ref_cloud_cover = tile_ref_cloud_covers[ref_idx]
                # identify either the reference image is cloud free or has been filled by CLEAR
                if ref_cloud_cover > 0:
                    S2_ref_path = join(tile_folder, f"{ref_name}_{tile_row_idx}-{tile_col_idx}_"
                                                    f"T-{ref_cloud_threshold}_K-{class_num}_"
                                                    f"C-{common_num}_S-{similar_num}_CLEAR{S2_image_suffix}")
                else:
                    S2_ref_path = join(tile_folder, f"{ref_name}_{tile_row_idx}-{tile_col_idx}_"
                                                    f"T-{ref_cloud_threshold}{S2_image_suffix}")
                S2_ref_dataset = rasterio.open(S2_ref_path)
                patch_profile = S2_ref_dataset.profile
                S2_ref_image = S2_ref_dataset.read()
                S2_ref_image = np.transpose(S2_ref_image, (1, 2, 0))
                S2_ref_image = S2_ref_image.astype(np.float32)
                S2_ref_images.append(S2_ref_image)
            S2_ref_images = np.stack(S2_ref_images, axis=3)

            for fuse_idx in range(tile_fuse_num):
                fuse_DOY = tile_fuse_DOYs[fuse_idx]
                M_fuse_image = M_fuse_images[:, :, :, fuse_idx]

                print(f"\tStart spatiotemporal fusion of the {fuse_idx}th image patch, "
                      f"DOY {fuse_DOY}")
                if tile_fusion_flags[fuse_idx] == 1:
                    print(f"This patch has been fused. Skip!")
                    continue

                t1 = datetime.now()
                F_pred, F_vip, C_vip = \
                    VIPSTF_SW_interpolated(S2_ref_images, M_ref_images, M_fuse_image,
                                           tile_ROI_mask,
                                           similar_win_size=fusion_similar_win_size,
                                           similar_num=fusion_similar_num)
                F_pred[F_pred < min_value] = min_value
                F_pred[F_pred > max_value] = max_value
                t2 = datetime.now()
                time_span = t2 - t1
                total_minutes = time_span.total_seconds() / 60.0
                print(f"\tUsed {total_minutes:.2f} minutes.")

                # save the result
                fuse_name = tile_fuse_names[fuse_idx]
                target_save_path = join(tile_folder,
                                        f"{fuse_name}_{tile_row_idx}-{tile_col_idx}_"
                                        f"W-{fusion_similar_win_size}_S-{fusion_similar_num}_"
                                        f"tol-{gap_len_tol}_"
                                        f"VIPSTF-SW{S2_image_suffix}")
                image_dataset = rasterio.open(target_save_path, mode='w', **patch_profile)
                F_pred = F_pred.transpose((2, 0, 1))
                image_dataset.write(F_pred)
                image_dataset.close()
                print(f"\tSaved {target_save_path}.\n")

                # save the fusion time and flag
                tile_fusion_time[fuse_idx] = total_minutes
                np.save(fusion_time_path, tile_fusion_time)
                tile_fusion_flags[fuse_idx] = 1
                np.save(fusion_flags_path, tile_fusion_flags)
