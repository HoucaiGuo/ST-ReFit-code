import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from general_utils import *
from CLEAR_K_D_Tree_utils import *
from os.path import join
from skimage.morphology import disk, binary_erosion

np.set_printoptions(suppress=True, precision=5)

"""
Gap-filling of Sentinel-2 images using the CLEAR algorithm with a patch-based processing strategy

1. Initialization of the algorithm.
    Get the DOYs and cloud covers, create arrays for saving the computation time.

2. Preliminary gap-filling of all reference image patches through time-series fitting.

3. Gap-filling of all reference image patches using CLEAR.

4. Gap-filling of image patches with cloud cover lower than the fill_cloud_threshold.
"""

# IO parameters
site = "IA-T15TVG"
tile = site.split(sep="-")[1]
temp_folder = rf"F:\Sentinel-2-MCD43A4-Seamless\{site}\3F\temp-patch"  # temporary folder
S2_image_folder = rf"F:\Sentinel-2-MCD43A4-Seamless\{site}\data\S2_NBAR"
S2_mask_folder = rf"F:\Sentinel-2-MCD43A4-Seamless\{site}\data\S2_mask"  # folders of images and masks
S2_image_suffix = ".jp2"
S2_mask_suffix = ".jp2"  # suffixes of images and masks
image_height = 5490
image_width = 5490
band_num = 6  # dimensions of the images
tile_height = 1100
tile_width = 1100  # height and width of the tiles
overlap_height = 0
overlap_width = 0  # height and width of overlapping area between two adjacent tiles
min_value = 0
max_value = 10000  # min and max values of the images

# # ROI parameters:
ROI_path = rf"F:\Sentinel-2-MCD43A4-Seamless\{site}\data\ROI\{site}-ROI.jp2"
ROI_cover_threshold = 30  # threshold to determine if a patch can be filled without assistance of other patch

# CLEAR parameters
ref_cloud_threshold = 30  # T, threshold to select reference images, default is 30
fill_cloud_threshold = 70  # threshold to determine if a patch can be filled without assistance of other patch, default is 70
DOY_cloud_threshold = 70  # threshold to determine if the image on DOY can be filled, default is 70
class_num = 20  # K, number of land-cover classes for Mini Batch K-Means classification, default is 20
common_num = 200  # C, number of common pixels retrieved by the K-D tree, default is 200
similar_num = 20  # S, number of similar pixels for residual compensation, default is 20
batch_max_cloudy_num = 10000  # maximum number of cloudy pixels processed by a single batch, default is 10000

if __name__ == "__main__":
    """
    1. Initialization
    """
    print(f"###### 1. Initialization  ######")
    # make the temp folder
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    # make folder for saving the information
    info_folder = join(temp_folder, "info")
    if not os.path.exists(info_folder):
        os.makedirs(info_folder)

    # get the day of years (DOYs)and names of all Sentinel-2 images
    S2_DOYs_path = join(info_folder, "S2-DOYs.npy")
    S2_names_path = join(info_folder, "S2-names.npy")
    if os.path.exists(S2_DOYs_path):
        DOYs = np.load(S2_DOYs_path)
        S2_names = np.load(S2_names_path)
    else:
        DOYs = []
        S2_names = []
        names = os.listdir(S2_image_folder)
        for name in names:
            if name.endswith(S2_image_suffix):
                name = name.split(sep=".")[0]
                splits = name.split(sep="_")
                DOY_str = splits[2]
                DOY = DOY_str_2_num(DOY_str)
                DOYs.append(DOY)
                S2_names.append(name)
        DOYs = np.array(DOYs, dtype=np.int32)
        DOY_sort_indices = np.argsort(DOYs)  # sort the array by DOYs!!!
        DOYs = DOYs[DOY_sort_indices]
        np.save(S2_DOYs_path, DOYs)
        S2_names = np.array(S2_names)
        S2_names = S2_names[DOY_sort_indices]
        np.save(S2_names_path, S2_names)

    # read ROI
    if ROI_path:
        ROI_dataset = rasterio.open(ROI_path)
        ROI = ROI_dataset.read().squeeze()
        ROI_mask = ROI == 1
        ROI_dataset.close()
    else:
        ROI_mask = np.full(shape=(image_height, image_width), fill_value=True, dtype=np.bool_)
    ROI_pixel_num = np.count_nonzero(ROI_mask)

    # calculate cloud covers of all DOYs
    S2_cloud_covers_path = join(info_folder, "S2-cloud-covers.npy")
    if os.path.exists(S2_cloud_covers_path):
        DOY_cloud_covers = np.load(S2_cloud_covers_path)
    else:
        DOY_cloud_covers = []
        names = os.listdir(S2_mask_folder)
        for name in names:
            if name.endswith(S2_mask_suffix):
                S2_mask_path = os.path.join(S2_mask_folder, name)
                mask_dataset = rasterio.open(S2_mask_path)
                mask = mask_dataset.read().squeeze()
                cloud_mask = mask != 0  # 0 is cloud free, 1 is cloudy, 2 is unobserved
                invalid_mask = np.bitwise_and(cloud_mask, ROI_mask)
                DOY_cloud_cover = np.count_nonzero(invalid_mask) / ROI_pixel_num * 100
                DOY_cloud_covers.append(DOY_cloud_cover)
        DOY_cloud_covers = np.array(DOY_cloud_covers)
        DOY_cloud_covers = DOY_cloud_covers[DOY_sort_indices]
        np.save(S2_cloud_covers_path, DOY_cloud_covers)

    for DOY_idx in range(DOYs.shape[0]):
        print(f"DOY {DOYs[DOY_idx]}, "
              f"cloud-cover {DOY_cloud_covers[DOY_idx]:.2f}%, "
              f"name {S2_names[DOY_idx]}")

    # calculate image tiles
    tile_row_num, tile_col_num, tile_start_rows, tile_end_rows, tile_start_cols, tile_end_cols = \
        calculate_image_tiles(image_height, image_width, tile_height, tile_width, overlap_height, overlap_width)
    print(f"Total {tile_row_num} * {tile_col_num} tiles!")

    # make folders for saving the gap-filled patches
    for tile_row_idx in range(tile_row_num):
        for tile_col_idx in range(tile_col_num):
            tile_folder = join(temp_folder, f"{tile_row_idx}-{tile_col_idx}")
            if not os.path.exists(tile_folder):
                os.makedirs(tile_folder)

    # calculate ROI covers of all tiles
    tile_ROI_covers_path = join(info_folder, f"tile-ROI-covers.npy")
    if os.path.exists(tile_ROI_covers_path):
        all_ROI_covers = np.load(tile_ROI_covers_path)
    else:
        # calculate ROI covers
        all_ROI_covers = np.empty(shape=(tile_row_num, tile_col_num), dtype=np.float32)
        for tile_row_idx in range(tile_row_num):
            for tile_col_idx in range(tile_col_num):
                # tile information
                tile_row_start = tile_start_rows[tile_row_idx, tile_col_idx]
                tile_row_end = tile_end_rows[tile_row_idx, tile_col_idx]
                tile_col_start = tile_start_cols[tile_row_idx, tile_col_idx]
                tile_col_end = tile_end_cols[tile_row_idx, tile_col_idx]
                tile_actual_height = tile_row_end - tile_row_start
                tile_actual_width = tile_col_end - tile_col_start
                tile_ROI_mask = ROI_mask[tile_row_start:tile_row_end, tile_col_start:tile_col_end]
                tile_ROI_pixel_num = np.count_nonzero(tile_ROI_mask)
                tile_ROI_cover = tile_ROI_pixel_num / (tile_actual_height * tile_actual_width) * 100
                all_ROI_covers[tile_row_idx, tile_col_idx] = tile_ROI_cover
        np.save(tile_ROI_covers_path, all_ROI_covers)
    print(all_ROI_covers)

    # calculate cloud covers of all patches
    patch_cloud_covers_path = join(info_folder, f"patch-cloud-covers.npy")
    if os.path.exists(patch_cloud_covers_path):
        all_cloud_covers = np.load(patch_cloud_covers_path)
    else:
        # calculate cloud covers
        all_cloud_covers = np.empty(shape=(tile_row_num, tile_col_num, DOYs.shape[0]), dtype=np.float32)
        for tile_row_idx in range(tile_row_num):
            for tile_col_idx in range(tile_col_num):
                # tile information
                tile_row_start = tile_start_rows[tile_row_idx, tile_col_idx]
                tile_row_end = tile_end_rows[tile_row_idx, tile_col_idx]
                tile_col_start = tile_start_cols[tile_row_idx, tile_col_idx]
                tile_col_end = tile_end_cols[tile_row_idx, tile_col_idx]
                tile_actual_height = tile_row_end - tile_row_start
                tile_actual_width = tile_col_end - tile_col_start
                tile_ROI_mask = ROI_mask[tile_row_start:tile_row_end, tile_col_start:tile_col_end]
                tile_ROI_pixel_num = np.count_nonzero(tile_ROI_mask)
                tile_ROI_cover = tile_ROI_pixel_num / (tile_actual_height * tile_actual_width) * 100

                for DOY_idx in range(DOYs.shape[0]):
                    DOY = DOYs[DOY_idx]
                    DOY_str = DOY_num_2_str(DOY)
                    S2_name = S2_names[DOY_idx]
                    mask_name = f"{S2_name}_mask{S2_mask_suffix}"
                    mask_path = join(S2_mask_folder, mask_name)
                    patch_mask, _ = read_patch_from_image(mask_path, tile_row_start, tile_row_end,
                                                          tile_col_start, tile_col_end)
                    patch_mask = patch_mask.squeeze()
                    patch_cloud_mask = patch_mask != 0  # 0 is cloud free, 1 is cloudy, 2 is unobserved
                    patch_invalid_mask = np.bitwise_and(patch_cloud_mask, tile_ROI_mask)
                    patch_cloud_cover = np.count_nonzero(patch_invalid_mask) / tile_ROI_pixel_num * 100 \
                        if tile_ROI_pixel_num != 0 else 0.0
                    all_cloud_covers[tile_row_idx, tile_col_idx, DOY_idx] = patch_cloud_cover
        np.save(patch_cloud_covers_path, all_cloud_covers)

    # time used for interpolating the reference images of each tile
    fitting_time_path = join(info_folder, f"fitting-time.npy")
    if os.path.exists(fitting_time_path):
        fitting_time = np.load(fitting_time_path)
    else:
        fitting_time = np.zeros(shape=(tile_row_num, tile_col_num), dtype=np.float32)
        np.save(fitting_time_path, fitting_time)

    # flags indicate the time-series fitting of reference images, 1 is interpolated
    fitting_flags_path = join(info_folder, f"fitting-flags.npy")
    if os.path.exists(fitting_flags_path):
        fitting_flags = np.load(fitting_flags_path)
    else:
        fitting_flags = np.zeros(shape=(tile_row_num, tile_col_num), dtype=np.uint8)
        np.save(fitting_flags_path, fitting_flags)

    # time used for filling each patch
    fill_time_path = join(info_folder, f"filling-time.npy")
    if os.path.exists(fill_time_path):
        all_fill_time = np.load(fill_time_path)
    else:
        all_fill_time = np.zeros(shape=(tile_row_num, tile_col_num, DOYs.shape[0]), dtype=np.float32)
        all_fill_time[all_cloud_covers == 0.0] = 0.0
        all_fill_time[all_ROI_covers == 0.0] = 0.0
        np.save(fill_time_path, all_fill_time)

    # flags indicating that if the patch is: non-ROI (-1), cloudy (0), or cloud free (1), or filled by CLEAR (2)
    filled_flag_path = join(info_folder, f"filling-flags.npy")
    if os.path.exists(filled_flag_path):
        all_filled_flags = np.load(filled_flag_path)
    else:
        all_filled_flags = np.zeros(shape=(tile_row_num, tile_col_num, DOYs.shape[0]), dtype=np.int8)
        all_filled_flags[all_cloud_covers == 0.0] = 1
        all_filled_flags[all_ROI_covers == 0.0] = -1  # non-ROI patches also have 0% cloud cover
        np.save(filled_flag_path, all_filled_flags)

    """
    2. Preliminary gap-filling of reference image patches through linear interpolation
    """
    print(f"\n###### 2. Preliminary gap-filling of reference image patches ######")
    # process for each tile
    for tile_row_idx in range(tile_row_num):
        for tile_col_idx in range(tile_col_num):
            print(f"Start interpolation to fill gaps in reference image patches of tile {tile_row_idx}-{tile_col_idx}.")

            if all_ROI_covers[tile_row_idx, tile_col_idx] == 0.0:
                print(f"\tAll pixels in tile {tile_row_idx}-{tile_col_idx} are non-ROI pixels. Skip!")
                continue

            if fitting_flags[tile_row_idx, tile_col_idx] == 1:
                print(f"\tTile {tile_row_idx}-{tile_col_idx} has been fitted!")
                continue

            # tile information
            tile_folder = join(temp_folder, f"{tile_row_idx}-{tile_col_idx}")
            tile_row_start = tile_start_rows[tile_row_idx, tile_col_idx]
            tile_row_end = tile_end_rows[tile_row_idx, tile_col_idx]
            tile_col_start = tile_start_cols[tile_row_idx, tile_col_idx]
            tile_col_end = tile_end_cols[tile_row_idx, tile_col_idx]
            tile_actual_height = tile_row_end - tile_row_start
            tile_actual_width = tile_col_end - tile_col_start
            tile_ROI_mask = ROI_mask[tile_row_start:tile_row_end, tile_col_start:tile_col_end]
            tile_ROI_pixel_num = np.count_nonzero(tile_ROI_mask)
            tile_ROI_cover = tile_ROI_pixel_num / (tile_actual_height * tile_actual_width) * 100

            # read images and masks
            images = []
            masks = []
            for DOY_idx in range(DOYs.shape[0]):
                DOY = DOYs[DOY_idx]
                DOY_str = DOY_num_2_str(DOY)
                S2_name = S2_names[DOY_idx]

                mask_name = f"{S2_name}_mask{S2_mask_suffix}"
                mask_path = join(S2_mask_folder, mask_name)
                patch_mask, _ = read_patch_from_image(mask_path, tile_row_start, tile_row_end,
                                                      tile_col_start, tile_col_end)
                patch_mask = patch_mask.squeeze()
                patch_cloud_mask = patch_mask != 0  # 0 is cloud free, 1 is cloudy, 2 is unobserved
                patch_invalid_mask = np.bitwise_and(patch_cloud_mask, tile_ROI_mask)
                masks.append(patch_invalid_mask)

                image_name = f"{S2_name}{S2_image_suffix}"
                image_path = join(S2_image_folder, image_name)
                image, patch_profile = read_patch_from_image(image_path, tile_row_start, tile_row_end,
                                                             tile_col_start, tile_col_end)
                image = image.astype(np.float32)
                image[~tile_ROI_mask] = 0.0
                images.append(image)
            images = np.stack(images, axis=3)
            masks = np.stack(masks, axis=2)

            tile_cloud_covers = all_cloud_covers[tile_row_idx, tile_col_idx, :].copy()
            DOY_can_fill = DOY_cloud_covers <= DOY_cloud_threshold
            cloud_cover_flags = tile_cloud_covers <= ref_cloud_threshold  # shape = (DOY_num, )
            ref_flags = np.bitwise_and(DOY_can_fill, cloud_cover_flags)

            ref_num = np.count_nonzero(ref_flags)
            ref_indices_ts = ref_flags.nonzero()[0]
            ref_DOYs = DOYs[ref_flags]
            ref_names = S2_names[ref_flags]
            ref_cloud_covers = all_cloud_covers[tile_row_idx, tile_col_idx, ref_flags]
            ref_images = images[:, :, :, ref_flags]
            ref_masks = masks[:, :, ref_flags]
            print(f"\tThere are {ref_num} reference image patches. {ref_DOYs}, {ref_cloud_covers}")

            # interpolation
            t1 = datetime.now()
            interp_images = fill_gaps_in_reference_images(images, masks, DOYs, ref_images, ref_masks, ref_DOYs)
            interp_images[interp_images < min_value] = min_value
            interp_images[interp_images > max_value] = max_value
            t2 = datetime.now()
            time_span = t2 - t1
            total_minutes = time_span.total_seconds() / 60.0
            fitting_time[tile_row_idx, tile_col_idx] = total_minutes
            np.save(fitting_time_path, fitting_time)

            # save the interpolated reference images
            for ref_idx in range(ref_num):
                interp_image = interp_images[:, :, :, ref_idx]
                ref_name = ref_names[ref_idx]
                ref_cloud_cover = ref_cloud_covers[ref_idx]
                if ref_cloud_cover == 0:
                    interp_save_path = join(tile_folder,
                                            f"{ref_name}_{tile_row_idx}-{tile_col_idx}_"
                                            f"T-{ref_cloud_threshold}{S2_image_suffix}")
                else:
                    interp_save_path = join(tile_folder,
                                            f"{ref_name}_{tile_row_idx}-{tile_col_idx}_"
                                            f"T-{ref_cloud_threshold}_interp{S2_image_suffix}")
                interp_dataset = rasterio.open(interp_save_path, mode='w', **patch_profile)
                interp_image = interp_image.transpose((2, 0, 1))
                interp_dataset.write(interp_image)
                interp_dataset.close()

            del interp_images

            fitting_flags[tile_row_idx, tile_col_idx] = 1
            np.save(fitting_flags_path, fitting_flags)

    """
    3. Apply CLEAR to all reference image patches of all tiles
    """
    print(f"\n###### 3. Gap-filling of all reference image patches using CLEAR ######")
    for tile_row_idx in range(tile_row_num):
        for tile_col_idx in range(tile_col_num):
            print(f"Start the gap-filling of reference image patches of tile {tile_row_idx}-{tile_col_idx}.")

            tile_ROI_cover = all_ROI_covers[tile_row_idx, tile_col_idx]
            if tile_ROI_cover == 0.0:
                print(f"\tAll pixels in tile {tile_row_idx}-{tile_col_idx} are non-ROI pixels. Skip!")
                continue
            elif tile_ROI_cover < ROI_cover_threshold:
                print(f"\tTile {tile_row_idx}-{tile_col_idx} has a ROI cover ({tile_ROI_cover:.2f}%) less than the "
                      f"threshold ({ROI_cover_threshold}%). Skip!")
                continue

            # tile information
            tile_folder = join(temp_folder, f"{tile_row_idx}-{tile_col_idx}")
            tile_row_start = tile_start_rows[tile_row_idx, tile_col_idx]
            tile_row_end = tile_end_rows[tile_row_idx, tile_col_idx]
            tile_col_start = tile_start_cols[tile_row_idx, tile_col_idx]
            tile_col_end = tile_end_cols[tile_row_idx, tile_col_idx]
            tile_actual_height = tile_row_end - tile_row_start
            tile_actual_width = tile_col_end - tile_col_start
            tile_ROI_mask = ROI_mask[tile_row_start:tile_row_end, tile_col_start:tile_col_end]

            # select reference images
            tile_cloud_covers = all_cloud_covers[tile_row_idx, tile_col_idx, :].copy()
            DOY_can_fill = DOY_cloud_covers <= DOY_cloud_threshold
            cloud_cover_flags = tile_cloud_covers <= ref_cloud_threshold  # shape = (DOY_num, )
            ref_flags = np.bitwise_and(DOY_can_fill, cloud_cover_flags)
            # ref_flags = tile_cloud_covers <= ref_cloud_threshold  # shape = (DOY_num, )
            ref_num = np.count_nonzero(ref_flags)
            ref_cloud_covers = tile_cloud_covers[ref_flags]
            tile_filled_flags = all_filled_flags[tile_row_idx, tile_col_idx, :]
            ref_filled_flags = tile_filled_flags[ref_flags]
            ref_DOYs = DOYs[ref_flags]
            ref_names = S2_names[ref_flags]
            ref_indices_ts = ref_flags.nonzero()[0]
            print(f"\tThere are {ref_num} reference image patches. {ref_DOYs}, {ref_cloud_covers}")

            if not np.any(ref_filled_flags == 0):  # no existing cloudy image
                print(f"\tAll reference image patches of tile {tile_row_idx}-{tile_col_idx} are cloud-free "
                      f"or have been filled. Skip!")
                continue

            # read interpolated reference images and cloud masks
            ref_images = []
            ref_masks = []
            for ref_idx in range(ref_num):
                ref_name = ref_names[ref_idx]
                mask_name = f"{ref_name}_mask{S2_mask_suffix}"
                mask_path = join(S2_mask_folder, mask_name)
                patch_mask, _ = read_patch_from_image(mask_path, tile_row_start, tile_row_end,
                                                      tile_col_start, tile_col_end)
                patch_mask = patch_mask.squeeze()
                patch_cloud_mask = patch_mask != 0  # 0 is cloud free, 1 is cloudy, 2 is unobserved
                patch_invalid_mask = np.bitwise_and(patch_cloud_mask, tile_ROI_mask)  # consider the ROI mask
                ref_masks.append(patch_invalid_mask)

                if ref_filled_flags[ref_idx] == 2:  # sometimes the process was cancelled
                    ref_path = join(tile_folder, f"{ref_name}_{tile_row_idx}-{tile_col_idx}_"
                                                 f"T-{ref_cloud_threshold}_K-{class_num}_"
                                                 f"C-{common_num}_S-{similar_num}_CLEAR{S2_image_suffix}")
                    ref_cloud_covers[ref_idx] = 0
                else:
                    ref_cloud_cover = ref_cloud_covers[ref_idx]
                    if ref_cloud_cover == 0:
                        ref_path = join(tile_folder,
                                        f"{ref_name}_{tile_row_idx}-{tile_col_idx}_"
                                        f"T-{ref_cloud_threshold}{S2_image_suffix}")
                    else:
                        ref_path = join(tile_folder,
                                        f"{ref_name}_{tile_row_idx}-{tile_col_idx}_"
                                        f"T-{ref_cloud_threshold}_interp{S2_image_suffix}")
                ref_dataset = rasterio.open(ref_path)
                patch_profile = ref_dataset.profile
                ref_image = ref_dataset.read()
                ref_image = np.transpose(ref_image, (1, 2, 0))
                ref_image = ref_image.astype(np.float32)
                ref_image[~tile_ROI_mask] = 0.0
                ref_images.append(ref_image)
            ref_images = np.stack(ref_images, axis=3)
            ref_masks = np.stack(ref_masks, axis=2)

            # gap-filling of reference images and save the results
            for idx in range(ref_num):
                ref_idx = np.argmax(ref_cloud_covers)
                ref_DOY = ref_DOYs[ref_idx]
                ref_name = ref_names[ref_idx]
                if ref_cloud_covers[ref_idx] > 0 and ref_filled_flags[ref_idx] != 2:
                    print(f"\tStart gap-filling of the {ref_idx}th reference patch, DOY: {ref_DOY}, "
                          f"cloud cover: {ref_cloud_covers[ref_idx]:.2f}%.")
                    ref_image = ref_images[:, :, :, ref_idx]
                    ref_mask = ref_masks[:, :, ref_idx]
                    if ref_idx == 0:
                        # image that need to be filled is the first one
                        other_ref_images = ref_images[:, :, :, 1:]
                    elif ref_idx == ref_images.shape[3] - 1:
                        # image that need to be filled is the last one
                        other_ref_images = ref_images[:, :, :, :-1]
                    else:
                        left = ref_images[:, :, :, :ref_idx]
                        right = ref_images[:, :, :, ref_idx + 1:]
                        other_ref_images = np.concatenate([left, right], axis=3)

                    # gap-filling
                    t1 = datetime.now()
                    print(f"\tStart Mini Batch K-Means classification of the pre-filled reference images.")
                    # here we do not use the reference image itself to avoid the impact of inaccurate
                    # time-series fitting results on classification
                    class_map, cluster_centers = classify_reference_images_roi_fast(other_ref_images, tile_ROI_mask,
                                                                                    class_num)
                    class_map = check_class_map_validity(class_map, class_num, cluster_centers, ref_mask)
                    filled_ref_image = fill_single_image_kd_tree_batch(other_ref_images,
                                                                       ref_image, ref_mask,
                                                                       class_map, class_num,
                                                                       common_num, similar_num,
                                                                       batch_max_cloudy_num)[1]
                    filled_ref_image[filled_ref_image < min_value] = min_value
                    filled_ref_image[filled_ref_image > max_value] = max_value
                    t2 = datetime.now()
                    time_span = t2 - t1
                    total_minutes = time_span.total_seconds() / 60.0
                    print(f"\tUsed {total_minutes:.2f} minutes.")

                    # use the gap-filled reference image
                    ref_images[:, :, :, ref_idx] = filled_ref_image
                    ref_cloud_covers[ref_idx] = 0

                    # save the gap-filled image
                    ref_save_path = join(tile_folder,
                                         f"{ref_name}_{tile_row_idx}-{tile_col_idx}_"
                                         f"T-{ref_cloud_threshold}_K-{class_num}_"
                                         f"C-{common_num}_S-{similar_num}_CLEAR{S2_image_suffix}")
                    ref_image_dataset = rasterio.open(ref_save_path, mode='w', **patch_profile)
                    filled_ref_image = filled_ref_image.transpose((2, 0, 1))
                    ref_image_dataset.write(filled_ref_image)
                    ref_image_dataset.close()
                    print(f"\tSaved {ref_save_path}.\n")

                    # index of the ref_idx th image in the whole time-series
                    ref_idx_ts = ref_indices_ts[ref_idx]
                    all_fill_time[tile_row_idx, tile_col_idx, ref_idx_ts] = total_minutes
                    np.save(fill_time_path, all_fill_time)
                    all_filled_flags[tile_row_idx, tile_col_idx, ref_idx_ts] = 2
                    np.save(filled_flag_path, all_filled_flags)

    """
    4. Apply CLEAR to patches with cloud cover higher than the ref_cloud_threshold but lower than the fill_cloud_threshold
    """
    print(f"\n###### 4. Gap-filling of image patches with cloud covers between "
          f"{ref_cloud_threshold}% and {fill_cloud_threshold}% using CLEAR ######")
    for tile_row_idx in range(tile_row_num):
        for tile_col_idx in range(tile_col_num):

            print(f"Start the gap-filling of image patches with cloud covers between "
                  f"({ref_cloud_threshold}%, {fill_cloud_threshold}%] of tile {tile_row_idx}-{tile_col_idx}.")

            tile_ROI_cover = all_ROI_covers[tile_row_idx, tile_col_idx]
            if tile_ROI_cover == 0.0:
                print(f"\tAll pixels in tile {tile_row_idx}-{tile_col_idx} are non-ROI pixels. Skip!")
                continue
            elif tile_ROI_cover < ROI_cover_threshold:
                print(f"\tTile {tile_row_idx}-{tile_col_idx} has a ROI cover ({tile_ROI_cover:.2f}%) less than the "
                      f"threshold ({ROI_cover_threshold}%). Skip!")
                continue

            # tile information
            tile_folder = join(temp_folder, f"{tile_row_idx}-{tile_col_idx}")
            tile_row_start = tile_start_rows[tile_row_idx, tile_col_idx]
            tile_row_end = tile_end_rows[tile_row_idx, tile_col_idx]
            tile_col_start = tile_start_cols[tile_row_idx, tile_col_idx]
            tile_col_end = tile_end_cols[tile_row_idx, tile_col_idx]
            tile_actual_height = tile_row_end - tile_row_start
            tile_actual_width = tile_col_end - tile_col_start
            tile_ROI_mask = ROI_mask[tile_row_start:tile_row_end, tile_col_start:tile_col_end]

            # select images that need to be filled, reference images have already been filled
            tile_cloud_covers = all_cloud_covers[tile_row_idx, tile_col_idx, :].copy()
            non_ref = tile_cloud_covers > ref_cloud_threshold
            # here we do not fill patches with cloud cover higher than the fill_cloud_threshold
            patch_can_fill = tile_cloud_covers <= fill_cloud_threshold
            # need_fill_flags = np.bitwise_and(non_ref, patch_can_fill)

            # # only fill patches in images with cloud cover lower than the threshold
            DOY_can_fill = DOY_cloud_covers <= DOY_cloud_threshold
            need_fill_flags = np.all(np.stack([non_ref, patch_can_fill, DOY_can_fill], axis=1), axis=1)

            if np.count_nonzero(need_fill_flags) == 0:
                print(f"\tNo image patch of tile {tile_row_idx}-{tile_col_idx} needs to be filled. Skip!")
                continue

            tile_filled_flags = all_filled_flags[tile_row_idx, tile_col_idx, :]
            if not np.any(tile_filled_flags[need_fill_flags] == 0):  # no existing cloudy image
                print(
                    f"\tAll image patches with cloud covers between ({ref_cloud_threshold}%, {fill_cloud_threshold}%] "
                    f"of tile {tile_row_idx}-{tile_col_idx} have been filled. Skip!")
                continue

            # select reference images
            # tile_cloud_covers = all_cloud_covers[tile_row_idx, tile_col_idx, :].copy()
            # ref_flags = tile_cloud_covers <= ref_cloud_threshold  # shape = (DOY_num, )
            # select reference images
            tile_cloud_covers = all_cloud_covers[tile_row_idx, tile_col_idx, :].copy()
            DOY_can_fill = DOY_cloud_covers <= DOY_cloud_threshold
            cloud_cover_flags = tile_cloud_covers <= ref_cloud_threshold  # shape = (DOY_num, )
            ref_flags = np.bitwise_and(DOY_can_fill, cloud_cover_flags)
            ref_num = np.count_nonzero(ref_flags)
            ref_cloud_covers = tile_cloud_covers[ref_flags]
            ref_DOYs = DOYs[ref_flags]
            ref_names = S2_names[ref_flags]
            ref_indices_ts = ref_flags.nonzero()[0]
            print(f"\tThere are {ref_num} reference image patches. {ref_DOYs}, {ref_cloud_covers}")

            ref_images = []
            for ref_idx in range(ref_num):
                ref_name = ref_names[ref_idx]
                ref_cloud_cover = ref_cloud_covers[ref_idx]
                # identify either the reference image is cloud free or has been filled by CLEAR
                if ref_cloud_cover > 0:
                    ref_path = join(tile_folder, f"{ref_name}_{tile_row_idx}-{tile_col_idx}_"
                                                 f"T-{ref_cloud_threshold}_K-{class_num}_"
                                                 f"C-{common_num}_S-{similar_num}_CLEAR{S2_image_suffix}")
                else:
                    ref_path = join(tile_folder, f"{ref_name}_{tile_row_idx}-{tile_col_idx}_"
                                                 f"T-{ref_cloud_threshold}{S2_image_suffix}")
                ref_dataset = rasterio.open(ref_path)
                patch_profile = ref_dataset.profile
                ref_image = ref_dataset.read()
                ref_image = np.transpose(ref_image, (1, 2, 0))
                ref_image = ref_image.astype(np.float32)
                ref_image[~tile_ROI_mask] = 0.0
                ref_images.append(ref_image)
            ref_images = np.stack(ref_images, axis=3)

            # Mini Batch Kmeans classification of the gap-filled reference images
            print(f"\tStart Mini Batch KMeans classification of the gap-filled reference image patches.")
            class_map, cluster_centers = classify_reference_images_roi_fast(ref_images, tile_ROI_mask, class_num)

            # fill the target images one by one
            target_indices = need_fill_flags.nonzero()[0]
            for target_idx in target_indices:
                target_DOY = DOYs[target_idx]
                target_name = S2_names[target_idx]

                print(f"\tStart gap-filling of the {target_idx}th image patch, DOY {target_DOY}, "
                      f"cloud cover: {tile_cloud_covers[target_idx]:.2f}%.")
                if all_filled_flags[tile_row_idx, tile_col_idx, target_idx] == 2:
                    print(f"\t\tThis patch has been filled. Skip!")
                    continue

                # read target image and mask
                target_mask_name = f"{target_name}_mask{S2_mask_suffix}"
                target_mask_path = join(S2_mask_folder, target_mask_name)
                target_mask, _ = read_patch_from_image(target_mask_path, tile_row_start, tile_row_end,
                                                       tile_col_start, tile_col_end)
                target_mask = target_mask.squeeze()
                target_mask = target_mask != 0  # 0 is cloud free, 1 is cloudy, 2 is unobserved
                target_mask = np.bitwise_and(target_mask, tile_ROI_mask)  # consider the ROI mask

                target_image_name = f"{target_name}{S2_image_suffix}"
                target_image_path = join(S2_image_folder, target_image_name)
                target_image, target_profile = read_patch_from_image(target_image_path, tile_row_start, tile_row_end,
                                                                     tile_col_start, tile_col_end)
                target_image = target_image.astype(np.float32)
                target_image[~tile_ROI_mask] = 0.0

                """
                Fill the cloudy image
                """
                t1 = datetime.now()
                class_map_ = check_class_map_validity(class_map, class_num, cluster_centers, target_mask)
                filled_target_image = fill_single_image_kd_tree_batch(ref_images,
                                                                      target_image, target_mask,
                                                                      class_map_, class_num,
                                                                      common_num, similar_num,
                                                                      batch_max_cloudy_num)[1]
                filled_target_image[filled_target_image < min_value] = min_value
                filled_target_image[filled_target_image > max_value] = max_value

                t2 = datetime.now()
                time_span = t2 - t1
                total_minutes = time_span.total_seconds() / 60.0
                print(f"\tUsed {total_minutes:.2f} minutes.")

                # save the result
                target_save_path = join(tile_folder,
                                        f"{target_name}_{tile_row_idx}-{tile_col_idx}_"
                                        f"T-{ref_cloud_threshold}_K-{class_num}_"
                                        f"C-{common_num}_S-{similar_num}_CLEAR{S2_image_suffix}")
                image_dataset = rasterio.open(target_save_path, mode='w', **target_profile)
                filled_target_image = filled_target_image.transpose((2, 0, 1))
                image_dataset.write(filled_target_image)
                image_dataset.close()
                print(f"\tSaved {target_save_path}.\n")

                all_fill_time[tile_row_idx, tile_col_idx, target_idx] = total_minutes
                np.save(fill_time_path, all_fill_time)

                # save the fill flag
                all_filled_flags[tile_row_idx, tile_col_idx, target_idx] = 2
                np.save(filled_flag_path, all_filled_flags)

    total_fitting_time = np.sum(fitting_time)
    total_filling_time = np.sum(all_fill_time)
    print(f"Used total {total_fitting_time + total_filling_time} minutes!")
