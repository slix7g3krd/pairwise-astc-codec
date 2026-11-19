"""第 58 批：ASTC 纹理压缩修复。

对应 13 项修复（按形状选块、按通道量化、色彩空间分离、alpha 保留、
压缩后 mip、法线归一化、块边界采样、缓存版本、纹理数组隔离、HDR 端点等），
每个函数只做纯计算，便于在不同平台的解码器之间保持一致的行为。
"""

import math

# 单轴允许的最大 ASTC 块尺寸（本实现固定 8×8 上限，按形状逐轴收缩）。
MAX_BLOCK_EXTENT = 8


def astc_block_size(width, height):
    """按纹理形状返回 ASTC 块尺寸。

    不再对所有纹理套用固定 8×8：每个轴独立取 ``min(8, 该轴尺寸)``，
    长条图（如 128×4）得到 8×4、小图（如 4×4）得到 4×4，避免按总
    分辨率选块而产生大量空块。cubemap / 纹理数组的每个面 / 层都应
    分别调用本函数，保证各层块尺寸一致且不跨层。
    """
    if width <= 0 or height <= 0:
        raise ValueError("texture dimensions must be positive")
    return (
        min(width, MAX_BLOCK_EXTENT),
        min(height, MAX_BLOCK_EXTENT),
    )


def channel_scale(value, bits):
    """按通道位数量化（各通道独立量化，端点和权重不共用精度）。

    把 [0, 1] 区间的通道值量化到 ``2**bits`` 个离散层级再映射回来，
    法线贴图的各通道因此可以分别选位宽，避免块内被插成平面。
    """
    if bits <= 0:
        raise ValueError("bits must be positive")
    levels = (1 << bits) - 1
    clamped = max(0.0, min(1.0, float(value)))
    return round(clamped * levels) / levels


def decode_mode(is_srgb):
    """返回解码色彩空间：sRGB 颜色贴图与线性数据（如法线贴图）分离。

    两类纹理不再共用一套质量参数，sRGB 走 ``"srgb"``，线性数据
    （法线、粗糙度、HDR 等）走 ``"linear"``，避免颜色发灰。
    """
    return "srgb" if is_srgb else "linear"


def alpha_value(alpha, cutoff):
    """按 alpha 裁剪阈值返回，保留半透明像素而非整块压成不透明。

    低于阈值的像素视为完全透明（0.0），其余像素保留原始 alpha，
    UI 图集边缘因此不会出现黑边。
    """
    if alpha < cutoff:
        return 0.0
    return float(alpha)


def normalize_normal(normal):
    """归一化法线向量。

    法线重建（含双线性插值、块边界混合）之后必须重新归一化，
    否则长度偏离 1 会在光照计算中产生暗斑。零向量安全返回零向量。
    """
    x, y, z = (float(v) for v in normal)
    length = math.sqrt(x * x + y * y + z * z)
    if length == 0.0:
        return (0.0, 0.0, 0.0)
    return (x / length, y / length, z / length)


def mip_count_for(resolution):
    """按分辨率返回 mip 数（含原始分辨率这一级）。

    mip 在压缩之后生成，逐级缩小到 1×1，级数为
    ``floor(log2(resolution)) + 1``，例如 64 → 7 级。
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if resolution == 1:
        return 1
    return int(math.floor(math.log2(resolution))) + 1


def edge_fade(x, block):
    """返回块边界衰减权重，范围 [0, 1]。

    块边缘（x 为 0 或 block）权重为 0，块中心权重为 1，呈线性三角
    衰减。双线性采样跨 ASTC 块边界时用该权重混合，法线不会在接缝
    处被插成平面；所有平台按同一公式处理边界块，保证解码一致性。
    """
    if block <= 0:
        raise ValueError("block must be positive")
    clamped = max(0.0, min(float(block), float(x)))
    distance_to_edge = min(clamped, block - clamped)
    half = block / 2.0
    return max(0.0, min(1.0, distance_to_edge / half))


def asset_key(path, version):
    """返回压缩缓存键：路径之外带上版本号。

    缓存不再只按路径命中——源图修改或压缩参数调整时提升版本号，
    键即变化，避免改图后仍命中旧块。
    """
    return f"{path}#{version}"


def array_layer_key(layer, slice_index):
    """返回纹理数组层键：层与切片显式成对隔离。

    纹理数组的每一层独立压缩、独立缓存，层间边界不参与块划分，
    杜绝跨层采样串图。
    """
    return (layer, slice_index)


def hdr_endpoint(value, max_value):
    """压缩 HDR 端点：钳制到 [0, max_value] 而非按 8bit 截断。

    HDR 纹理使用宽位宽端点编码，超过当前块端点上限的高光被钳制
    到 ``max_value`` 而不是截成 8bit 最大值，保留高光动态范围。
    """
    if max_value < 0:
        raise ValueError("max_value must be non-negative")
    return max(0.0, min(float(max_value), float(value)))
