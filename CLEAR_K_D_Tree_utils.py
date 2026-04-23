import numpy as np
from tqdm import trange
from scipy.interpolate import interp1d
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KDTree
from joblib import Parallel, delayed
import gc


def linear_interpolation(DOYs, vals, valid_flags, DOYs_pred):
    DOYs_fit = DOYs[valid_flags]
    vals_fit = vals[valid_flags]
    valid_num = np.count_nonzero(valid_flags)
    if valid_num == 0:  # no valid observation, use 0 to fill
        F = np.array([0 for i in range(DOYs_pred.shape[0])])
    elif valid_num == 1:  # only one valid observation, use the valid value to fill
        F = np.array([vals_fit[0] for i in range(DOYs_pred.shape[0])])
    else:  # linear interpolation
        f = interp1d(DOYs_fit, vals_fit, kind='linear', fill_value='extrapolate')
        F = f(DOYs_pred)
    return F


def process_pixel_linear_interpolation(DOYs, vals, valid_flags, DOYs_pred):
    # process all bands of the given pixel location
    F = np.empty(shape=(vals.shape[0], DOYs_pred.shape[0]))
    for band_idx in range(vals.shape[0]):
        F[band_idx, :] = linear_interpolation(DOYs, vals[band_idx, :], valid_flags, DOYs_pred)
    return F


def fill_gaps_in_reference_images(images, masks, DOYs, ref_images, ref_masks, ref_DOYs, n_jobs=-1):
    # the value is 1 if the pixel is cloudy in at least one reference image
    any_cloudy_mask = np.any(ref_masks, axis=2).reshape(ref_masks.shape[0] * ref_masks.shape[1])

    # reshape the images and masks for parallel computing
    images = images.reshape(images.shape[0] * images.shape[1], images.shape[2], images.shape[3])
    masks = masks.reshape(masks.shape[0] * masks.shape[1], masks.shape[2])

    # only process when the pixel is cloudy in at least one reference image
    images = images[any_cloudy_mask, :, :]
    masks = masks[any_cloudy_mask, :]

    results = Parallel(n_jobs=n_jobs, backend='loky', timeout=None) \
        (delayed(process_pixel_linear_interpolation)(DOYs, images[i, :, :],
                                                     ~masks[i, :], ref_DOYs)
         for i in trange(images.shape[0]))
    results = np.array(results, dtype=np.float32)

    interp_images_ = ref_images.copy()
    interp_images_ = interp_images_.reshape(interp_images_.shape[0] * interp_images_.shape[1],
                                            interp_images_.shape[2], interp_images_.shape[3])
    interp_images_[any_cloudy_mask, :, :] = results
    interp_images_ = interp_images_.reshape(ref_images.shape[0], ref_images.shape[1],
                                            ref_images.shape[2], ref_images.shape[3])

    interp_images = ref_images.copy()
    for i in range(ref_images.shape[3]):
        ref_mask = ref_masks[:, :, i]
        interp_images[ref_mask, :, i] = interp_images_[ref_mask, :, i]

    # del results
    # del images
    # del interp_images_
    # gc.collect()

    return interp_images


def classify_reference_images_fast(ref_images, class_num):
    """
    Classify the stacked reference images using MiniBatchKMeans.
    """
    X = np.reshape(ref_images,
                   (ref_images.shape[0] * ref_images.shape[1],
                    ref_images.shape[2] * ref_images.shape[3]))
    kmeans = MiniBatchKMeans(n_clusters=class_num, max_iter=1000, random_state=42)
    kmeans.fit(X)
    class_map = kmeans.labels_.reshape((ref_images.shape[0], ref_images.shape[1]))

    return class_map, kmeans.cluster_centers_


def classify_reference_images_roi_fast(ref_images, roi_mask, class_num):
    """
    Classify the stacked reference images using MiniBatchKMeans.
    Pixels outside the ROI is -1, meaning no classification result.
    """
    class_map = np.full(shape=roi_mask.shape, fill_value=-1)
    ref_images_roi = ref_images[roi_mask, :, :]
    X = np.reshape(ref_images_roi,
                   (ref_images_roi.shape[0],
                    ref_images_roi.shape[1] * ref_images_roi.shape[2]))
    kmeans = MiniBatchKMeans(n_clusters=class_num, max_iter=1000, random_state=42)
    kmeans.fit(X)
    class_map[roi_mask] = kmeans.labels_

    return class_map, kmeans.cluster_centers_


def check_class_map_validity(class_map, class_num, class_centers, cloud_mask):
    """
    Check the validity of classification map. If there is no clear pixel within a certain class,
    merge it with the nearest class.
    """
    deleted_classes = []
    for class_idx in range(class_num):
        class_mask = class_map == class_idx
        common_mask = np.all(np.stack([class_mask, ~cloud_mask], axis=2), axis=2)
        if np.count_nonzero(common_mask) < 200:
            deleted_classes.append(class_idx)

            class_center = class_centers[class_idx, :]
            center_distances = np.sum(np.abs(class_centers - class_center), axis=1)

            sorted_indices = np.argsort(center_distances)
            for idx in sorted_indices:
                if idx not in deleted_classes:
                    dst_class = idx
                    class_map[class_mask] = dst_class
                    print(f"{class_idx} ---> {dst_class}")
                    break

    return class_map


def fill_single_image_kd_tree(ref_images,
                              cloudy_image, cloud_mask,
                              class_map, class_num,
                              common_num, similar_num):
    final_prediction = cloudy_image.copy()
    cloud_mask_1 = cloud_mask.copy()
    residuals = np.zeros(shape=cloudy_image.shape, dtype=np.float32)

    """
    Step 1. Class-based linear regression
    """
    all_coefficients = []
    for class_idx in range(class_num):
        class_mask = class_map == class_idx
        class_cloudy_mask = np.all(np.stack([class_mask, cloud_mask_1], axis=2), axis=2)
        # no cloudy pixel in this class
        if np.count_nonzero(class_cloudy_mask) == 0:
            all_coefficients.append(np.zeros(shape=(ref_images.shape[2], ref_images.shape[3]), dtype=np.float32))
            continue
        # row and column indices of the common pixels
        common_mask = np.all(np.stack([class_mask, ~cloud_mask_1], axis=2), axis=2)
        class_coefficients = []
        for band_idx in range(cloudy_image.shape[2]):
            ref_bands = ref_images[:, :, band_idx, :]
            cloudy_band = cloudy_image[:, :, band_idx]

            reg = LinearRegression()

            # shape = (common_num, ref_num)
            X_train = ref_bands[common_mask, :]
            # shape = (common_num, )
            y_train = cloudy_band[common_mask]

            # fit, predict, and calculate the residuals
            reg.fit(X_train, y_train)
            # shape = (class_cloudy_num, ref_num)
            X_pred = ref_bands[class_cloudy_mask, :]
            final_prediction[class_cloudy_mask, band_idx] = reg.predict(X_pred)
            residuals[common_mask, band_idx] = y_train - reg.predict(X_train)

            class_coefficients.append(reg.coef_)
        all_coefficients.append(class_coefficients)
    all_coefficients = np.array(all_coefficients)  # shape = (class_num, band_num, ref_num)
    reg_prediction = final_prediction.copy()

    """
    Step 2. Iterative residual compensation
    """
    # print(f"\tStart iterative residual compensation.")
    # skipped_class_num = 0
    for class_idx in range(class_num):
        class_mask = class_map == class_idx
        class_cloudy_mask = np.all(np.stack([class_mask, cloud_mask_1], axis=2), axis=2)
        # no cloudy pixel in this class
        if np.count_nonzero(class_cloudy_mask) == 0:
            # skipped_class_num += 1
            continue
        # row and column indices of the common pixels
        common_mask = np.all(np.stack([class_mask, ~cloud_mask_1], axis=2), axis=2)
        common_row_indices, common_col_indices = common_mask.nonzero()
        common_pixels = ref_images[common_mask, :, :]
        common_coordinates = np.stack([common_row_indices, common_col_indices], axis=1)

        kd_tree = KDTree(common_coordinates, leaf_size=40, metric="euclidean")
        k = common_num if common_num <= common_coordinates.shape[0] else common_coordinates.shape[0]

        common_residuals = residuals[common_mask, :]
        # temporal_weights = all_coefficients[class_idx - skipped_class_num, :, :]  # shape = (band_num, ref_num)
        temporal_weights = all_coefficients[class_idx, :, :]  # shape = (band_num, ref_num)

        cloudy_num = np.count_nonzero(class_cloudy_mask)
        cloudy_row_indices, cloudy_col_indices = class_cloudy_mask.nonzero()
        for cloudy_idx in trange(cloudy_num, position=0, leave=True):
            target_row_idx = cloudy_row_indices[cloudy_idx]
            target_col_idx = cloudy_col_indices[cloudy_idx]
            cloudy_coordinates = np.expand_dims(np.stack([target_row_idx, target_col_idx], axis=0), axis=0)
            indices = kd_tree.query(cloudy_coordinates, k=k, return_distance=False)
            indices = indices.squeeze()

            """
            consider spectral difference
            """
            common_values = common_pixels[indices, :, :]  # shape = (common_num, band_num, ref_num)
            target_values = ref_images[target_row_idx, target_col_idx, :, :]  # shape = (band_num, ref_num)

            # calculate the time-series absolute difference (TSAD)
            TSADs = np.sum(np.abs(common_values - target_values) *
                           np.abs(temporal_weights), axis=(1, 2))  # shape = (common_num, )

            similar_indices = np.argsort(TSADs)[:similar_num]
            similar_TSADs = TSADs[similar_indices]
            similar_TSADs_norm = ((similar_TSADs - similar_TSADs.min()) /
                                  (similar_TSADs.max() - similar_TSADs.min() + 0.00001) + 1)

            selected_row_indices = common_row_indices[indices]
            selected_col_indices = common_col_indices[indices]

            similar_row_indices = selected_row_indices[similar_indices]
            similar_col_indices = selected_col_indices[similar_indices]

            similar_distances = np.sqrt(np.square(similar_row_indices - target_row_idx) +
                                        np.square(similar_col_indices - target_col_idx))
            similar_distances_norm = ((similar_distances - similar_distances.min()) /
                                      (similar_distances.max() - similar_distances.min() + 0.00001) + 1)

            similar_weights = ((1 / (similar_distances_norm * similar_TSADs_norm)) /
                               np.sum((1 / (similar_distances_norm * similar_TSADs_norm))))

            selected_residuals = common_residuals[indices, :]
            similar_residuals = selected_residuals[similar_indices, :]
            residual = np.sum(np.stack([similar_weights for i in range(cloudy_image.shape[2])],
                                       axis=1) * similar_residuals,
                              axis=0)
            final_prediction[target_row_idx, target_col_idx, :] += residual

    return reg_prediction, final_prediction


def fill_single_image_kd_tree_batch(ref_images,
                                    cloudy_image, cloud_mask,
                                    class_map, class_num,
                                    common_num, similar_num,
                                    batch_max_cloudy_num=10000):
    """
    Fill a cloudy image using CLEAR.
    """
    final_prediction = cloudy_image.copy()
    cloud_mask_1 = cloud_mask.copy()
    residuals = np.zeros(shape=cloudy_image.shape, dtype=np.float32)

    """
    Step 1. Class-based linear regression
    """
    all_coefficients = []
    for class_idx in range(class_num):
        class_mask = class_map == class_idx
        class_cloudy_mask = np.all(np.stack([class_mask, cloud_mask_1], axis=2), axis=2)
        # no cloudy pixel in this class
        if np.count_nonzero(class_cloudy_mask) == 0:
            continue
        # row and column indices of the common pixels
        common_mask = np.all(np.stack([class_mask, ~cloud_mask_1], axis=2), axis=2)
        class_coefficients = []
        for band_idx in range(cloudy_image.shape[2]):
            ref_bands = ref_images[:, :, band_idx, :]
            cloudy_band = cloudy_image[:, :, band_idx]

            reg = LinearRegression()

            # shape = (common_num, ref_num)
            X_train = ref_bands[common_mask, :]
            # shape = (common_num, )
            y_train = cloudy_band[common_mask]

            # fit, predict, and calculate the residuals
            reg.fit(X_train, y_train)
            # shape = (class_cloudy_num, ref_num)
            X_pred = ref_bands[class_cloudy_mask, :]
            final_prediction[class_cloudy_mask, band_idx] = reg.predict(X_pred)
            residuals[common_mask, band_idx] = y_train - reg.predict(X_train)

            class_coefficients.append(reg.coef_)
        all_coefficients.append(class_coefficients)
    all_coefficients = np.array(all_coefficients)
    reg_prediction = final_prediction.copy()

    """
    Step 2. Iterative residual compensation
    """
    skipped_class_num = 0
    for class_idx in trange(class_num):
        class_mask = class_map == class_idx
        class_cloudy_mask = np.all(np.stack([class_mask, cloud_mask_1], axis=2), axis=2)  # cloudy pixels in this class
        cloudy_num = np.count_nonzero(class_cloudy_mask)
        if cloudy_num == 0:  # no cloudy pixel in this class, skip
            skipped_class_num += 1
            continue

        # row and column indices of the common pixels
        common_mask = np.all(np.stack([class_mask, ~cloud_mask_1], axis=2), axis=2)
        common_row_indices, common_col_indices = common_mask.nonzero()
        common_coordinates = np.stack([common_row_indices, common_col_indices], axis=1)

        # construct the K-D tree using the coordinates of all common pixels
        kd_tree = KDTree(common_coordinates, leaf_size=40, metric="euclidean")
        k = common_num if common_num <= common_coordinates.shape[0] else common_coordinates.shape[0]

        # all common pixels and the corresponding residuals
        common_pixels = ref_images[common_mask, :, :]  # shape = (class_common_num, band_num, ref_num)
        common_residuals = residuals[common_mask, :]

        # temporal weights of the reference images
        temporal_weights = all_coefficients[class_idx - skipped_class_num, :, :]  # shape = (band_num, ref_num)

        # row and column indices of the cloudy pixels
        cloudy_row_indices, cloudy_col_indices = class_cloudy_mask.nonzero()

        # batch processing of the cloudy pixels
        for batch_start in range(0, cloudy_num, batch_max_cloudy_num):
            batch_end = min(batch_start + batch_max_cloudy_num, cloudy_num)
            batch_cloudy_num = batch_end - batch_start  # number of cloudy pixels in current batch

            # row and column indices of the cloudy pixels in current batch
            batch_cloudy_row_indices = cloudy_row_indices[batch_start:batch_end]
            batch_cloudy_col_indices = cloudy_col_indices[batch_start:batch_end]
            batch_cloudy_coordinates = np.stack([batch_cloudy_row_indices, batch_cloudy_col_indices], axis=1)

            # query for the k-nearest common pixels for all cloudy pixels in current batch using the K-D tree,
            # also return the euclidean distances
            distances, indices = kd_tree.query(batch_cloudy_coordinates,
                                               k=k, return_distance=True)  # shapes = (cloudy_num, common_num)

            # get values and residuals of the k-nearest common pixels for all cloudy pixels in current batch,
            # automatically broadcast, shapes = (batch_cloudy_num, common_num, band_num, ref_num)
            selected_common_vals = common_pixels[indices]
            selected_common_residuals = common_residuals[indices]

            # values of all cloudy pixels in current batch, shape = (batch_cloudy_num, band_num, ref_num)
            target_vals = ref_images[batch_cloudy_row_indices, batch_cloudy_col_indices, :, :]

            # calculate TSADs between the cloudy pixels in current batch and their corresponding common pixels,
            # automatically broadcast
            spectral_diffs = np.abs(selected_common_vals - target_vals[:, None, :, :])
            weighted_diffs = spectral_diffs * temporal_weights[None, None, :, :]
            TSADs = np.sum(weighted_diffs, axis=(2, 3))  # shape = (batch_cloudy_num, common_num)

            # select similar pixels for all cloudy pixels in current batch, similar pixels are defined as
            # common pixels that having the smallest TSADs
            partition_indices = np.argpartition(TSADs, similar_num, axis=1)  # shape = (batch_cloudy_num, common_num)
            similar_indices = partition_indices[:, :similar_num]  # shape = (batch_cloudy_num, similar_num)

            # TSADs values of the similar pixels
            cloudy_pixel_indices = np.arange(batch_cloudy_num)[:,
                                   None]  # indices along the cloudy pixel dimension (0 ... cloudy_num-1)
            similar_TSADs = TSADs[cloudy_pixel_indices, similar_indices]  # (batch_cloudy_num, similar_num)
            similar_TSADs_min = similar_TSADs.min(axis=1, keepdims=True)
            similar_TSADs_max = similar_TSADs.max(axis=1, keepdims=True)
            similar_TSADs_norm = ((similar_TSADs - similar_TSADs_min) /
                                  (similar_TSADs_max - similar_TSADs_min + 0.00001) + 1)

            # spatial distances of the similar pixels
            similar_distances = distances[cloudy_pixel_indices, similar_indices]  # (batch_cloudy_num, similar_num)
            similar_distances_min = similar_distances.min(axis=1, keepdims=True)
            similar_distances_max = similar_distances.max(axis=1, keepdims=True)
            similar_distances_norm = ((similar_distances - similar_distances_min) /
                                      (similar_distances_max - similar_distances_min + 0.00001) + 1)

            # calculate the weights of all similar pixels
            similar_Is_reciprocal = 1.0 / (
                        similar_distances_norm * similar_TSADs_norm)  # (batch_cloudy_num, similar_num)
            similar_weights = similar_Is_reciprocal / np.sum(similar_Is_reciprocal, axis=1,
                                                             keepdims=True)  # (batch_cloudy_num, similar_num)

            # calculate the weighted residuals for all cloudy pixels in current batch
            similar_residuals = selected_common_residuals[cloudy_pixel_indices, similar_indices,
                                :]  # (batch_cloudy_num, similar_num, band_num)
            weighted_residuals = np.sum(similar_weights[:, :, None] * similar_residuals,
                                        axis=1)  # (batch_cloudy_num, band_num)

            # compensate for the residuals
            final_prediction[batch_cloudy_row_indices, batch_cloudy_col_indices, :] += weighted_residuals

    return reg_prediction, final_prediction
