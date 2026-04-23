import re
import numpy as np
import rasterio
from rasterio.windows import Window
from datetime import datetime


def calculate_image_tiles(image_height, image_width,
                          tile_height, tile_width,
                          overlap_height, overlap_width):
    stride_row = tile_height - overlap_height
    stride_col = tile_width - overlap_width

    # 初步计算块数
    tile_row_num = np.ceil((image_height - overlap_height) / stride_row).astype(np.int32)
    tile_col_num = np.ceil((image_width - overlap_width) / stride_col).astype(np.int32)

    # 检查最后一行是否需要合并
    last_row_height = image_height - (tile_row_num - 1) * stride_row
    if last_row_height <= tile_height // 2 and tile_row_num > 1:
        tile_row_num -= 1

    # 检查最后一列是否需要合并
    last_col_width = image_width - (tile_col_num - 1) * stride_col
    if last_col_width <= tile_width // 2 and tile_col_num > 1:
        tile_col_num -= 1

    tile_start_rows = np.empty(shape=(tile_row_num, tile_col_num), dtype=np.int32)
    tile_end_rows = np.empty(shape=(tile_row_num, tile_col_num), dtype=np.int32)
    tile_start_cols = np.empty(shape=(tile_row_num, tile_col_num), dtype=np.int32)
    tile_end_cols = np.empty(shape=(tile_row_num, tile_col_num), dtype=np.int32)
    for row_idx in range(tile_row_num):
        for col_idx in range(tile_col_num):
            # 计算当前块的左上角位置
            tile_start_row = row_idx * stride_row
            tile_start_col = col_idx * stride_col

            # 计算块的实际尺寸（考虑图像边界裁剪）
            if row_idx != tile_row_num - 1:
                tile_end_row = tile_start_row + tile_height
            else:
                tile_end_row = image_height
            if col_idx != tile_col_num - 1:
                tile_end_col = tile_start_col + tile_width
            else:
                tile_end_col = image_width

            tile_start_rows[row_idx, col_idx] = tile_start_row
            tile_end_rows[row_idx, col_idx] = tile_end_row
            tile_start_cols[row_idx, col_idx] = tile_start_col
            tile_end_cols[row_idx, col_idx] = tile_end_col

    return tile_row_num, tile_col_num, tile_start_rows, tile_end_rows, tile_start_cols, tile_end_cols


def read_patch_from_image(image_path, patch_row_start, patch_row_end, patch_col_start, patch_col_end):
    with rasterio.open(image_path) as src:
        patch_height = patch_row_end - patch_row_start
        patch_width = patch_col_end - patch_col_start

        patch_window = Window(
            col_off=patch_col_start,
            row_off=patch_row_start,
            width=patch_width,
            height=patch_height
        )

        # 读取数据 (保持原始波段顺序)
        patch_data = src.read(window=patch_window)

        # 创建元数据副本
        patch_profile = src.profile.copy()

        # 更新元数据
        patch_profile.update({
            'height': patch_height,
            'width': patch_width,
            'transform': src.window_transform(patch_window),
        })

        # # 更新地理边界
        # patch_profile['bounds'] = src.window_bounds(patch_window)

        patch_data = patch_data.transpose((1, 2, 0))

        return patch_data, patch_profile


def get_S2_DOY(S2_name):
    pattern = re.compile(r"S2[AB]_(\d{8})")
    match = pattern.search(S2_name)
    date = match.group(1)
    DOY =datetime.strptime(date, "%Y%m%d").timetuple().tm_yday

    return DOY


def DOY_str_2_num(DOY_str):
    """
    '001' --> 1; '041' --> 41; '121' --> 121
    """
    if DOY_str.startswith("00"):
        DOY = eval(DOY_str[2])
    elif DOY_str.startswith("0"):
        DOY = eval(DOY_str[1:])
    else:
        DOY = eval(DOY_str)

    return DOY


def DOY_num_2_str(DOY_num):
    if 0 <= DOY_num < 10:
        DOY_str = f"00{DOY_num}"
    elif 10 <= DOY_num < 100:
        DOY_str = f"0{DOY_num}"
    else:
        DOY_str = f"{DOY_num}"

    return DOY_str


def color_composite(image, bands_idx):
    image = np.stack([image[:, :, i] for i in bands_idx], axis=2)
    return image


def linear_pct_stretch(img, pct=2, max_out=1, min_out=0):
    def gray_process(gray):
        truncated_down = np.percentile(gray, pct)
        truncated_up = np.percentile(gray, 100 - pct)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out
        gray[gray < min_out] = min_out
        gray[gray > max_out] = max_out
        return gray

    bands = []
    for band_idx in range(img.shape[2]):
        band = img[:, :, band_idx]
        band_strch = gray_process(band)
        bands.append(band_strch)
    img_pct_strch = np.stack(bands, axis=2)
    return img_pct_strch


def color_composite_ma(image, bands_idx):
    image = np.ma.stack([image[:, :, i] for i in bands_idx], axis=2)
    return image


def linear_pct_stretch_ma(img, pct=2, max_out=1, min_out=0):
    def gray_process(gray):
        truncated_down = np.percentile(gray, pct)
        truncated_up = np.percentile(gray, 100 - pct)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out
        gray[gray < min_out] = min_out
        gray[gray > max_out] = max_out
        return gray

    out = img.copy()
    for band_idx in range(img.shape[2]):
        band = img.data[:, :, band_idx]
        mask = img.mask[:, :, band_idx]
        band_strch = gray_process(band[~mask])
        out.data[:, :, band_idx][~mask] = band_strch
    return out


def set_axis_size(axis, width, height):
    """ w, h: width, height in inches """
    l = axis.figure.subplotpars.left
    r = axis.figure.subplotpars.right
    t = axis.figure.subplotpars.top
    b = axis.figure.subplotpars.bottom
    figw = float(width) / (r - l)
    figh = float(height) / (t - b)
    axis.figure.set_size_inches(figw, figh)


def set_axis_visibility(axis):
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines["top"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["right"].set_visible(False)


def decode_bit(value, bit):
    """提取某个位的值（0或1）"""
    return (value // (2 ** bit)) % 2


def decode_bits(value, start_bit, num_bits):
    """提取多个位组成的整数值"""
    return (value // (2 ** start_bit)) % (2 ** num_bits)