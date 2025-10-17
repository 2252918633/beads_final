# methods.py
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.cluster import KMeans
import math
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# 导入 photo2pixel 模块
try:
    from photo2pixel.models.module_pixel_effect import PixelEffectModule
    from photo2pixel.utils.img_common_util import convert_image_to_tensor, convert_tensor_to_image
    PHOTO2PIXEL_AVAILABLE = True
except ImportError:
    PHOTO2PIXEL_AVAILABLE = False
# backend/methods.py

import pymysql
import os
from contextlib import contextmanager

# ==================== 数据库连接 ====================

@contextmanager
def get_db_connection():
    """数据库连接上下文管理器"""
    conn = pymysql.connect(
        host=os.getenv('MYSQL_HOST', 'beads_mysql'),   # ✅ 只写主机名，不带端口
        port=int(os.getenv('MYSQL_PORT', 3306)), # ✅ 端口单独传
        user=os.getenv('MYSQL_USER', 'beads_user'),
        password=os.getenv('MYSQL_PASSWORD', '225291'),
        database=os.getenv('MYSQL_DATABASE', 'beads'),
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    try:
        yield conn
    finally:
        conn.close()

# ==================== 月度重置 ====================

def check_and_reset_monthly():
    """检查并执行月度重置（自动调用存储过程）"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.callproc('check_and_reset_monthly')
                result = cursor.fetchone()
                if result:
                    print(f"[月度检查] {result.get('result', 'OK')}")
            conn.commit()
    except Exception as e:
        print(f"[警告] 月度检查失败: {e}")

# ==================== 验证码验证（登录时） ====================

def validate_code(code: str) -> dict:
    """
    验证验证码是否有效（仅验证，不扣减次数）
    
    Returns:
        {
            'valid': bool,          # 是否有效
            'remaining': int,       # 剩余次数
            'message': str          # 提示信息
        }
    """
    # 先执行月度检查
    check_and_reset_monthly()
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT code, times FROM verification_codes WHERE code = %s",
                    (code,)
                )
                result = cursor.fetchone()
                
                if not result:
                    return {
                        'valid': False,
                        'remaining': 0,
                        'message': '验证码不存在'
                    }
                
                remaining = result['times']
                
                if remaining <= 0:
                    return {
                        'valid': False,
                        'remaining': 0,
                        'message': '验证码次数已用完，将在下月初重置为100次'
                    }
                
                return {
                    'valid': True,
                    'remaining': remaining,
                    'message': f'验证成功，本月还可使用 {remaining} 次'
                }
                
    except Exception as e:
        print(f"[错误] 验证码验证失败: {e}")
        return {
            'valid': False,
            'remaining': 0,
            'message': f'系统错误: {str(e)}'
        }

# ==================== 扣减次数（生成图纸时） ====================

def consume_code_usage(code: str) -> dict:
    """
    扣减验证码使用次数
    
    Returns:
        {
            'success': bool,        # 是否成功
            'remaining': int,       # 剩余次数
            'message': str          # 提示信息
        }
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # 查询当前次数
                cursor.execute(
                    "SELECT times FROM verification_codes WHERE code = %s",
                    (code,)
                )
                result = cursor.fetchone()
                
                if not result:
                    return {
                        'success': False,
                        'remaining': 0,
                        'message': '验证码不存在'
                    }
                
                current_times = result['times']
                
                if current_times <= 0:
                    return {
                        'success': False,
                        'remaining': 0,
                        'message': '次数已用完，无法生成图纸'
                    }
                
                # 扣减次数
                cursor.execute(
                    "UPDATE verification_codes SET times = times - 1 WHERE code = %s AND times > 0",
                    (code,)
                )
                conn.commit()
                
                new_remaining = current_times - 1
                
                return {
                    'success': True,
                    'remaining': new_remaining,
                    'message': f'生成成功，剩余 {new_remaining} 次'
                }
                
    except Exception as e:
        print(f"[错误] 扣减次数失败: {e}")
        return {
            'success': False,
            'remaining': 0,
            'message': f'系统错误: {str(e)}'
        }

# ==================== 查询剩余次数 ====================

def get_code_remaining(code: str) -> int:
    """查询验证码剩余次数"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT times FROM verification_codes WHERE code = %s",
                    (code,)
                )
                result = cursor.fetchone()
                return result['times'] if result else 0
    except Exception as e:
        print(f"[错误] 查询剩余次数失败: {e}")
        return 0

# 保留原有的 load_valid_codes() 作为兼容（可选）
def load_valid_codes():
    """兼容性函数：从数据库加载所有有效验证码"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT code FROM verification_codes WHERE times > 0")
                results = cursor.fetchall()
                return set(row['code'] for row in results)
    except Exception as e:
        print(f"[警告] 无法从数据库加载验证码: {e}")
        # 降级到文件加载
        import json
        try:
            with open('code.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get('code', []))
        except:
            return set()
def enhance_lines(input_path: str, output_path: str, line_strength: int = 3):
    """
    增强图像线条（增粗、保持黑色、保留原图颜色）

    参数:
        input_path (str): 输入图片路径
        output_path (str): 输出图片路径
        line_strength (int): 线条粗细程度（建议范围 1~6）
    """
    # --- 参数校正 ---
    if line_strength < 1:
        line_strength = 1
    if line_strength > 10:
        line_strength = 10

    thickness = line_strength
    iterations = max(1, line_strength // 2)  # 线条越粗，膨胀次数适当增加
    edge_low, edge_high = 50, 150

    # --- 读取图像 ---
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"❌ 无法读取图片: {input_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 检测边缘 ---
    edges = cv2.Canny(gray, edge_low, edge_high)

    # --- 膨胀线条（加粗） ---
    kernel = np.ones((thickness, thickness), np.uint8)
    edges_bold = cv2.dilate(edges, kernel, iterations=iterations)

    # --- 反相得到黑线掩码 ---
    mask = cv2.bitwise_not(edges_bold)

    # --- 保留原图颜色 + 黑色线条 ---
    img_bold = cv2.bitwise_and(img, img, mask=mask)
    black_lines = np.zeros_like(img)
    final = cv2.addWeighted(img_bold, 1, black_lines, 1, 0)

    # --- 保存结果 ---
    cv2.imwrite(output_path, final)
    print(f"✅ 已保存增强线条图像: {output_path}（线条强度: {line_strength}）")
# 加载拼豆颜色库
def load_bead_colors():
    """加载 color.json 中的拼豆颜色"""
    try:
        with open('color.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['bead_palette']
    except:
        return []

# 加载验证码
def load_valid_codes():
    try:
        with open('code.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('code', [])
    except:
        return []

def hex_to_rgb(hex_color):
    """将十六进制颜色转为RGB"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def hex_from_rgb(rgb):
    """将RGB转为十六进制颜色"""
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

def rgb_to_lab(rgb):
    """将RGB转换为Lab色彩空间（手动实现，更准确）"""
    # 归一化RGB值
    r, g, b = [x / 255.0 for x in rgb]
    
    # RGB to XYZ
    def rgb_to_xyz_helper(c):
        if c > 0.04045:
            return ((c + 0.055) / 1.055) ** 2.4
        return c / 12.92
    
    r = rgb_to_xyz_helper(r)
    g = rgb_to_xyz_helper(g)
    b = rgb_to_xyz_helper(b)
    
    # 转换到XYZ (使用D65光源)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    
    # XYZ to Lab
    # D65标准光源参考白点
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    
    def xyz_to_lab_helper(t):
        if t > 0.008856:
            return t ** (1/3)
        return (7.787 * t) + (16 / 116)
    
    fx = xyz_to_lab_helper(x / xn)
    fy = xyz_to_lab_helper(y / yn)
    fz = xyz_to_lab_helper(z / zn)
    
    L = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    return L, a, b

def color_distance_advanced(rgb1, rgb2):
    """使用改进的Lab色彩距离（考虑人眼感知权重）"""
    L1, a1, b1 = rgb_to_lab(rgb1)
    L2, a2, b2 = rgb_to_lab(rgb2)
    
    # Delta E 2000 的简化版本，考虑亮度和色度的权重
    dL = L1 - L2
    da = a1 - a2
    db = b1 - b2
    
    # 计算色度差
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    dC = C1 - C2
    
    # 计算色调差
    dH_squared = da**2 + db**2 - dC**2
    dH = math.sqrt(max(0, dH_squared))
    
    # 加权因子（给亮度差更大的权重）
    kL = 1.0
    kC = 1.0
    kH = 1.0
    
    # 计算加权距离
    distance = math.sqrt(
        (dL / kL) ** 2 +
        (dC / kC) ** 2 +
        (dH / kH) ** 2
    )
    
    return distance
def advanced_color_quantization(img, n_colors=16):
    """高级颜色量化，使用K-means聚类减少颜色数量"""
    import cv2
    from sklearn.cluster import KMeans
    
    # 转换为numpy数组
    data = np.array(img).reshape(-1, 3).astype(np.float32)
    
    if len(data) == 0:
        return img
    
    # 如果颜色数已经很少，直接返回
    unique_colors = len(np.unique(data.view(np.dtype((np.void, data.dtype.itemsize*data.shape[1])))))
    if unique_colors <= n_colors:
        return img
    
    # 使用K-means聚类
    kmeans = KMeans(n_clusters=min(n_colors, unique_colors), random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_.astype(np.uint8)
    
    # 重建图像
    quantized_data = centers[labels].reshape(img.size[1], img.size[0], 3)
    return Image.fromarray(quantized_data)
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import colorsys

def compress_saturation(rgb, factor=0.9):
    h, l, s = colorsys.rgb_to_hls(*(x/255 for x in rgb))
    s = min(1.0, s * factor)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r*255), int(g*255), int(b*255))

def find_closest_bead_color(rgb, bead_colors):
    """最高准确度拼豆色匹配（含饱和度压缩 + CIEDE2000）"""
    if not bead_colors:
        return None
    print(1111111111)
    # 校正输入颜色
    rgb = compress_saturation(rgb, 0.9)

    # 转换目标色到 Lab（注意 sRGB 非线性）
    target_color = convert_color(
        sRGBColor(rgb[0], rgb[1], rgb[2], is_upscaled=True, is_linear=False),
        LabColor
    )

    min_distance = float("inf")
    closest_bead = None

    for bead in bead_colors:
        bead_rgb = hex_to_rgb(bead['hex'])
        bead_lab = convert_color(
            sRGBColor(bead_rgb[0], bead_rgb[1], bead_rgb[2], is_upscaled=True, is_linear=False),
            LabColor
        )
        distance = delta_e_cie2000(target_color, bead_lab)

        if distance < min_distance:
            min_distance = distance
            closest_bead = bead

    return closest_bead


def kmeans_then_map_to_beads(img, k, bead_colors):
    """先KMeans聚类，再映射到拼豆颜色"""
    arr = np.array(img.convert('RGB'))
    h, w, c = arr.shape
    data = arr.reshape((-1, 3)).astype(np.float32)
    
    print(f"KMeans 聚类中... 样本数={len(data)}, 聚类数={k}")
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
    labels = kmeans.fit_predict(data)
    centers = np.clip(kmeans.cluster_centers_.astype(np.uint8), 0, 255)
    
    # 关键：把每个聚类中心映射到最接近的拼豆颜色
    bead_centers = []
    for center in centers:
        bead = find_closest_bead_color(tuple(center), bead_colors)
        if bead:
            bead_rgb = hex_to_rgb(bead['hex'])
            bead_centers.append(bead_rgb)
        else:
            bead_centers.append(tuple(center))
    
    bead_centers = np.array(bead_centers)
    
    # 用映射后的拼豆颜色重新绘制图像
    new_img = bead_centers[labels].reshape((h, w, 3))
    return Image.fromarray(new_img.astype(np.uint8))

def map_image_to_bead_colors(img, bead_colors):
    """将图像的每个像素映射到最接近的拼豆颜色"""
    arr = np.array(img.convert('RGB'))
    h, w, _ = arr.shape
    
    # 创建新的图像数组
    new_arr = np.zeros_like(arr)
    
    for y in range(h):
        for x in range(w):
            rgb = tuple(arr[y, x])
            bead = find_closest_bead_color(rgb, bead_colors)
            if bead:
                new_rgb = hex_to_rgb(bead['hex'])
                new_arr[y, x] = new_rgb
    
    return Image.fromarray(new_arr)

def image_to_bead_palette_counts(img, bead_colors):
    """统计图像中使用的拼豆颜色及数量，返回拼豆编号"""
    arr = np.array(img.convert('RGB'))
    h, w, _ = arr.shape
    bead_counter = {}
    
    for y in range(h):
        for x in range(w):
            rgb = tuple(arr[y, x])
            bead = find_closest_bead_color(rgb, bead_colors)
            if bead:
                code = bead['code']
                if code in bead_counter:
                    bead_counter[code] += 1
                else:
                    bead_counter[code] = 1
    
    # 按数量排序
    items = sorted(bead_counter.items(), key=lambda x: -x[1])
    return items  # 返回 [(code, count), ...] 格式

def add_coordinates_to_board(board_img, beads_width, beads_height, pixel_per_bead):
    """在图纸上添加坐标轴"""
    margin_top = 30
    margin_left = 40
    
    new_width = board_img.width + margin_left
    new_height = board_img.height + margin_top
    new_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    
    new_img.paste(board_img, (margin_left, margin_top))
    
    draw = ImageDraw.Draw(new_img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 14)
        except:
            font = ImageFont.load_default()
    
    # 添加水平坐标
    for x in range(beads_width):
        x_pos = margin_left + x * pixel_per_bead + pixel_per_bead // 2
        label = str(x + 1)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text((x_pos - text_width // 2, (margin_top - text_height) // 2), 
                  label, fill=(0, 0, 0), font=font)
    
    # 添加垂直坐标
    for y in range(beads_height):
        y_pos = margin_top + y * pixel_per_bead + pixel_per_bead // 2
        label = str(y + 1)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text(((margin_left - text_width) // 2, y_pos - text_height // 2), 
                  label, fill=(0, 0, 0), font=font)
    
    return new_img
# 替换为简单的尺寸检查函数
def check_image_size(img, max_width=2048, max_height=2048):
    """
    检查图像尺寸是否超过限制
    返回: (is_valid, error_message)
    """
    width, height = img.size
    
    if width > max_width or height > max_height:
        return False, f"图片尺寸过大 ({width}×{height})，请使用 {max_width}×{max_height} 以内的图片"
    
    return True, None

def pixel_segment(input_image_path, pixels_per_row, output_image_path, 
                  param_num_bins=4, param_kernel_size=7):
    """
    使用分割像素算法对图像进行像素化处理
    
    参数:
        input_image_path (str): 输入图像路径
        pixels_per_row (int): 目标宽度的像素块数量
        output_image_path (str): 输出图像路径
        param_num_bins (int): 颜色分级数量，默认4
        param_kernel_size (int): 卷积核大小，默认11
    """
    if not PHOTO2PIXEL_AVAILABLE:
        print("[警告] photo2pixel 模块未正确导入，使用简单像素化")
        # 降级到简单像素化
        img = Image.open(input_image_path).convert('RGB')
        w, h = img.size
        pixel_height = max(1, int(h * pixels_per_row / w))
        small = img.resize((pixels_per_row, pixel_height), Image.LANCZOS)
        pixel_img = small.resize((w, h), Image.NEAREST)
        pixel_img.save(output_image_path, format='PNG')
        return
    
    try:
        # 1. 读取图像
        img = Image.open(input_image_path).convert('RGB')
        w, h = img.size
        
        # 2. 计算像素大小（pixel_size）
        # pixel_size 表示每个像素块的大小
        pixel_size = max(1, w // pixels_per_row)
        
        # 确保 pixel_size 合理
        if pixel_size < 4:
            pixel_size = 4
        if pixel_size > 32:
            pixel_size = 32
        
        # 3. 将图像转换为 tensor
        img_pt = convert_image_to_tensor(img)
        
        # 4. 创建 PixelEffectModule 并处理
        model = PixelEffectModule()
        model.eval()
        
        with torch.no_grad():
            result_rgb_pt = model(
                img_pt,
                param_num_bins=param_num_bins,
                param_kernel_size=param_kernel_size,
                param_pixel_size=pixel_size
            )
        
        # 5. 将 tensor 转换回图像
        result_img = convert_tensor_to_image(result_rgb_pt)
        
        # 6. 保存结果
        result_img.save(output_image_path, format='PNG')
        
        print(f"[像素分割] 完成: {input_image_path} -> {output_image_path}")
        print(f"[像素分割] 参数: pixels_per_row={pixels_per_row}, pixel_size={pixel_size}")
        
    except Exception as e:
        print(f"[错误] pixel_segment 处理失败: {e}")
        # 发生错误时降级到简单像素化
        img = Image.open(input_image_path).convert('RGB')
        w, h = img.size
        pixel_height = max(1, int(h * pixels_per_row / w))
        small = img.resize((pixels_per_row, pixel_height), Image.LANCZOS)
        pixel_img = small.resize((w, h), Image.NEAREST)
        pixel_img.save(output_image_path, format='PNG')
        print(f"[降级] 使用简单像素化完成")
"""
拼豆色号转换器 - Mard品牌
将RGB颜色转换为最接近的Mard拼豆色号
"""

from typing import Tuple, Dict, Optional
from functools import lru_cache


class MardColorMatcher:
    """Mard拼豆色号匹配器"""
    
    # Mard调色板数据 (色号: RGB十六进制颜色)
    MARD_PALETTE = {
        'A1': 'fff5ca', 'A2': 'ffffcc', 'A3': 'fff297', 'A4': 'fff651',
        'A5': 'ffdb4d', 'A6': 'fcc73b', 'A7': 'fe8443', 'A8': 'e5c62d',
        'A9': 'fbaa72', 'A10': 'fd9443', 'A11': 'ffcf7c', 'A12': 'fdbc9e',
        'A14': 'ff7443', 'A16': 'fff49e', 'A17': 'fde676', 'A18': 'ffb67b',
        'A19': 'fa9285', 'A20': 'fee27e',
        
        'B1': 'eaf149', 'B2': 'b6eb45', 'B3': '9af5a8', 'B4': '51ff51',
        'B5': '6fd363', 'B6': '79ebc7', 'B7': '3bb48b', 'B8': '13904c',
        'B9': '28523a', 'B10': 'ace1d9', 'B11': '52641d', 'B12': '206a4f',
        'B13': 'c7ff85', 'B14': 'd0ed39', 'B15': '205726', 'B16': 'ccffaa',
        'B17': 'b6c130', 'B18': 'e4f954', 'B19': '00d0a1', 'B20': 'c4ffd7',
        'B21': '157c74', 'B22': '0d5349', 'B23': '32451d', 'B24': 'e4fb9f',
        'B25': '5c9083', 'B26': 'a9a44e',
        
        'C1': 'e0ffe2', 'C2': 'b9f6ed', 'C3': 'a8e1fc', 'C4': '84d2ff',
        'C5': '24b9e4', 'C6': '78bdf8', 'C7': '3c8aec', 'C8': '195dc4',
        'C9': '2e26cc', 'C10': '30d9ed', 'C11': '00c4d0', 'C12': '1a3758',
        'C13': 'b4d7ff', 'C14': 'e0fff9', 'C15': '00cbd2', 'C16': '0a5491',
        'C17': '7be7f4', 'C18': '264257', 'C19': '259bb9', 'C20': '157cbf',
        'C21': 'deeefe', 'C22': '7fc0d4',
        
        'D1': 'b9cffe', 'D2': '9ca6db', 'D3': '224eb1', 'D4': '2e4770',
        'D5': 'c56fb8', 'D6': 'aa76e4', 'D7': '7e4dc7', 'D8': 'e0cdff',
        'D9': 'c9befd', 'D10': '351d4f', 'D11': 'c1c2ff', 'D12': 'daafe4',
        'D13': 'c64dae', 'D14': '9735b3', 'D15': '46297d', 'D16': 'e3e4ff',
        'D17': 'c9d9fd', 'D18': 'b87bd5', 'D19': 'e8cbfa', 'D20': 'a73ae9',
        'D21': '8b339b', 'D22': '595093', 'D23': 'efe7fc', 'D24': '7f77ea',
        'D25': '423bc0',
        
        'E1': 'ffd1cc', 'E2': 'ffcaea', 'E3': 'fb8dc5', 'E4': 'ee78ac',
        'E5': 'f653a6', 'E6': 'fd2f81', 'E7': 'a2176a', 'E8': 'ffd8e8',
        'E9': 'ea8cda', 'E10': 'bf3974', 'E11': 'fee7e2', 'E12': 'fcafdd',
        'E13': 'a61284', 'E14': 'fdd2bf', 'E15': 'f3d1d7', 'E16': 'fff3eb',
        'E17': 'fae8f8', 'E18': 'fdd1ea', 'E19': 'f6d3f3', 'E20': 'f4d7e9',
        'E21': 'd6b3b9', 'E22': 'c685b0', 'E23': 'a089a3',
        
        'F1': 'fda098', 'F2': 'fa6a62', 'F3': 'ec4a59', 'F4': 'ff452c',
        'F5': 'ff0000', 'F6': 'b53710', 'F7': '811530', 'F8': 'bb082f',
        'F9': 'e67088', 'F10': '8c420c', 'F11': '6f3329', 'F12': 'f9425e',
        'F13': 'd14b30', 'F14': 'fdaca9', 'F15': 'dd0c2c', 'F16': 'fddedc',
        'F17': 'fbaf9f', 'F18': 'e0804d', 'F19': 'c64b60',
        
        'G1': 'fee5c7', 'G2': 'fcceb7', 'G3': 'f9c2a4', 'G4': 'dcb99a',
        'G5': 'f2a76d', 'G6': 'ee9867', 'G7': '91664e', 'G8': '4d312c',
        'G9': 'eabb85', 'G10': 'bf9143', 'G11': 'e4c896', 'G12': 'ddb986',
        'G13': 'd19058', 'G14': '856a54', 'G15': 'f5f3dc', 'G16': 'f2dfce',
        'G17': '5d5249', 'G18': 'ffedd9', 'G19': 'eca450', 'G20': 'b0623e',
        'G21': 'cb906e',
        
        'H2': 'ffffff', 'H3': 'aeaeae', 'H4': '979296', 'H5': '606060',
        'H6': '353537', 'H7': '000000', 'H8': 'f7e6f0', 'H9': 'e6e3dc',
        'H10': 'e5e2f3', 'H11': 'cecdcb', 'H12': 'fff2e2', 'H13': 'ece1cb',
        'H14': 'c5ced8', 'H15': '9facc7', 'H16': '241d1a', 'H17': 'f6f4f5',
        'H18': 'fffdf1', 'H19': 'f8f3eb', 'H20': 'a0acb1',
        
        'M1': 'ccd7c9', 'M2': '91aa94', 'M3': '718898', 'M4': 'dacec2',
        'M5': 'dbd6b6', 'M6': 'c8b999', 'M7': 'c1aca7', 'M8': 'b09699',
        'M9': 'ab9787', 'M10': 'bb9fb9', 'M11': '9b809f', 'M12': '5a474c',
        'M13': 'dda89a', 'M14': 'c0725d', 'M15': '86848a',
    }
    
    def __init__(self):
        """初始化颜色匹配器"""
        # 预处理调色板，将十六进制转为RGB
        self.palette_rgb = {}
        for name, hex_color in self.MARD_PALETTE.items():
            self.palette_rgb[name] = self._hex_to_rgb(hex_color)
    
    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """将十六进制颜色转换为RGB元组"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    @staticmethod
    def _rgb_to_hex(r: int, g: int, b: int) -> str:
        """将RGB转换为十六进制颜色"""
        return f"{r:02x}{g:02x}{b:02x}"
    
    @staticmethod
    def _color_distance(r1: int, g1: int, b1: int, 
                       r2: int, g2: int, b2: int) -> float:
        """
        计算两个RGB颜色之间的感知色差
        使用加权欧几里得距离公式，考虑人眼对不同颜色的敏感度
        
        参考: https://stackoverflow.com/a/9085524
        """
        rmean = (r1 + r2) / 2
        rd = r1 - r2
        gd = g1 - g2
        bd = b1 - b2
        
        # 加权计算，绿色权重最高，红蓝根据平均红色值动态调整
        return (((512 + rmean) * rd * rd) / 256 + 
                4 * gd * gd + 
                ((767 - rmean) * bd * bd) / 256)
    
    @lru_cache(maxsize=1024)
    def find_closest_color(self, r: int, g: int, b: int) -> Dict[str, any]:
        """
        查找最接近的Mard色号
        
        Args:
            r: 红色分量 (0-255)
            g: 绿色分量 (0-255)
            b: 蓝色分量 (0-255)
            
        Returns:
            字典包含: {
                'name': 色号名称 (如 'F5'),
                'hex': 十六进制颜色值 (如 'ff0000'),
                'rgb': RGB元组 (255, 0, 0),
                'distance': 色差值
            }
        """
        min_distance = float('inf')
        closest_color = None
        closest_name = None
        
        # 遍历调色板找到最小色差
        for name, (cr, cg, cb) in self.palette_rgb.items():
            distance = self._color_distance(r, g, b, cr, cg, cb)
            
            # 如果完全匹配，直接返回
            if distance == 0:
                return {
                    'name': name,
                    'hex': self.MARD_PALETTE[name],
                    'rgb': (cr, cg, cb),
                    'distance': 0
                }
            
            if distance < min_distance:
                min_distance = distance
                closest_color = (cr, cg, cb)
                closest_name = name
        
        return {
            'name': closest_name,
            'hex': self.MARD_PALETTE[closest_name],
            'rgb': closest_color,
            'distance': min_distance
        }
    
    def find_closest_color_from_hex(self, hex_color: str) -> Dict[str, any]:
        """
        从十六进制颜色查找最接近的Mard色号
        
        Args:
            hex_color: 十六进制颜色，如 '#FF0000' 或 'FF0000'
            
        Returns:
            与find_closest_color相同的字典
        """
        r, g, b = self._hex_to_rgb(hex_color)
        return self.find_closest_color(r, g, b)
    
    def convert_image_colors(self, image_rgb_array) -> list:
        """
        转换整个图像的颜色到Mard色号
        
        Args:
            image_rgb_array: 图像的RGB数组 (高度 x 宽度 x 3)
            
        Returns:
            色号名称的二维数组
        """
        import numpy as np
        
        if not isinstance(image_rgb_array, np.ndarray):
            image_rgb_array = np.array(image_rgb_array)
        
        height, width = image_rgb_array.shape[:2]
        result = []
        
        for y in range(height):
            row = []
            for x in range(width):
                r, g, b = image_rgb_array[y, x, :3]
                color_info = self.find_closest_color(int(r), int(g), int(b))
                row.append(color_info['name'])
            result.append(row)
        
        return result
    
    def get_color_statistics(self, color_names: list) -> Dict[str, int]:
        """
        统计色号使用数量
        
        Args:
            color_names: 色号名称列表或二维数组
            
        Returns:
            字典 {色号: 数量}，按数量降序排列
        """
        from collections import Counter
        
        # 如果是二维数组，展平
        if isinstance(color_names[0], list):
            flat_list = [name for row in color_names for name in row]
        else:
            flat_list = color_names
        
        counter = Counter(flat_list)
        # 按数量降序排序
        return dict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
