import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import trange


def select_similar_pixels(F_vip, similar_win_size=25, similar_num=25):
    # calculate distances from the central pixel to neighboring pixels
    neighbor_rows = np.linspace(start=0, stop=similar_win_size - 1, num=similar_win_size)
    neighbor_cols = np.linspace(start=0, stop=similar_win_size - 1, num=similar_win_size)
    xx, yy = np.meshgrid(neighbor_rows, neighbor_cols, indexing='ij')
    central_row = similar_win_size // 2
    central_col = similar_win_size // 2
    distances = np.sqrt(np.square(xx - central_row) + np.square(yy - central_col))
    distances = distances.flatten()

    pad_width = similar_win_size // 2
    F_vip_pad = np.pad(F_vip,
                       pad_width=((pad_width, pad_width),
                                  (pad_width, pad_width),
                                  (0, 0)),
                       mode="reflect")
    F_vip_similar_weights = np.empty(shape=(F_vip.shape[0], F_vip.shape[1], similar_num),
                                     dtype=np.float32)
    F_vip_similar_indices = np.empty(shape=(F_vip.shape[0], F_vip.shape[1], similar_num),
                                     dtype=np.uint32)

    for row_idx in trange(F_vip.shape[0]):
        for col_idx in range(F_vip.shape[1]):
            central_pixel_vals = F_vip[row_idx, col_idx, :]
            neighbor_pixel_vals = F_vip_pad[row_idx:row_idx + similar_win_size,
                                  col_idx:col_idx + similar_win_size, :]
            D = np.mean(np.abs(neighbor_pixel_vals - central_pixel_vals), axis=2).flatten()
            similar_indices = np.argsort(D)[:similar_num]

            similar_distances = 1 + distances[similar_indices] / (similar_win_size // 2)
            similar_weights = (1 / similar_distances) / np.sum(1 / similar_distances)

            F_vip_similar_indices[row_idx, col_idx, :] = similar_indices
            F_vip_similar_weights[row_idx, col_idx, :] = similar_weights

    return F_vip_similar_indices, F_vip_similar_weights


def VIPSTF_SW_interpolated(F_TS, C_TS_cubic, C_tp_cubic, ROI_mask, similar_win_size=25, similar_num=25):
    F_row_num, F_col_num, band_num, ts_num = F_TS.shape

    # reshape data for regression
    C_TS_cubic_reshaped = C_TS_cubic.reshape(F_row_num * F_col_num, band_num, ts_num)
    C_tp_cubic_reshaped = C_tp_cubic.reshape(F_row_num * F_col_num, band_num)
    F_TS_reshaped = F_TS.reshape(F_row_num * F_col_num, band_num, ts_num)
    ROI_mask_reshaped = ROI_mask.reshape(F_row_num * F_col_num, )

    # construct the VIP
    C_vip = np.empty(shape=(F_row_num, F_col_num, band_num), dtype=np.float32)
    F_vip = np.empty(shape=(F_row_num, F_col_num, band_num), dtype=np.float32)
    for band_idx in range(band_num):
        # regression model fitting using the interpolated images
        reg = LinearRegression(fit_intercept=True, positive=True)
        X_train = C_TS_cubic_reshaped[ROI_mask_reshaped, band_idx, :]
        y_train = C_tp_cubic_reshaped[ROI_mask_reshaped, band_idx]
        reg.fit(X_train, y_train)

        # predict the VIP
        X_pred = C_TS_cubic_reshaped[:, band_idx, :]
        C_vip[:, :, band_idx] = reg.predict(X_pred).reshape(F_row_num, F_col_num)
        X_pred = F_TS_reshaped[:, band_idx, :]
        F_vip[:, :, band_idx] = reg.predict(X_pred).reshape(F_row_num, F_col_num)

    # temporal change
    C_tp_cubic[~ROI_mask, :] = 0
    C_vip[~ROI_mask, :] = 0
    F_vip[~ROI_mask, :] = 0
    C_changes = C_tp_cubic - C_vip

    # select similar pixels
    print("Start similar pixels selection.")
    F_vip_similar_indices, F_vip_similar_weights = select_similar_pixels(F_vip, similar_win_size, similar_num)

    # predict
    print("Start prediction.")
    pad_width = similar_win_size // 2
    C_changes_pad = np.pad(C_changes, pad_width=((pad_width, pad_width),
                                                 (pad_width, pad_width), (0, 0)), mode="reflect")
    F_pred = F_vip.copy()
    for row_idx in trange(F_vip.shape[0]):
        for col_idx in range(F_vip.shape[1]):
            neighbor_changes = C_changes_pad[row_idx:row_idx + similar_win_size,
                               col_idx:col_idx + similar_win_size, :]
            similar_indices = F_vip_similar_indices[row_idx, col_idx, :]
            similar_changes = neighbor_changes.reshape(similar_win_size ** 2, -1)[similar_indices, :]
            similar_weights = F_vip_similar_weights[row_idx, col_idx, :]
            # F_change = np.sum(similar_changes * similar_weights, axis=0)
            F_changes = np.sum(np.stack([similar_weights for i in range(band_num)],
                                        axis=1) * similar_changes,
                               axis=0)

            F_pred[row_idx, col_idx, :] += F_changes

    return F_pred, F_vip, C_vip
