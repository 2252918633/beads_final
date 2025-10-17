# api.py
from flask import Flask, request, render_template, send_file, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import os
import csv
import datetime
from functools import wraps
import zipfile
import tempfile
import sys

# 添加photo2pxil_api到路径
import sys
import os
from werkzeug.middleware.proxy_fix import ProxyFix

# ... 其他导入 ...

# 在文件顶部，导入部分之后添加：

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
photo2pixel_path = os.path.join(project_root, 'photo2pxil_api')
sys.path.insert(0, photo2pixel_path)

from methods import (
    validate_code,          # 新增：验证验证码
    consume_code_usage,     # 新增：扣减次数
    get_code_remaining,    
    load_bead_colors, 
    load_valid_codes,
    kmeans_then_map_to_beads,
    map_image_to_bead_colors,
    image_to_bead_palette_counts,
    add_coordinates_to_board,
    advanced_color_quantization,
    pixel_segment,
    MardColorMatcher        # ✅ 添加 Mard 颜色匹配器
)

try:
    from photo2pixel_api import pixel_effect, edge_enhance
    PHOTO2PIXEL_AVAILABLE = True
    print("✓ photo2pixel_api 加载成功 (两个方法: pixel_effect, edge_enhance)")
except ImportError as e:
    PHOTO2PIXEL_AVAILABLE = False
    print(f"警告: photo2pixel_api未找到 - {e}")

app = Flask(__name__, template_folder='../frontend')
app.secret_key = 'your-secret-key-here-change-this'  # 生产环境请改成随机密钥

# ✅ Session 配置（确保每次关闭浏览器都需要重新登录）
app.config['SESSION_COOKIE_NAME'] = 'beads_session'
app.config['SESSION_COOKIE_HTTPONLY'] = True  # 防止 XSS 攻击
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # 防止 CSRF 攻击
app.config['PERMANENT_SESSION_LIFETIME'] = 600  # Session 1小时后过期
app.config['SESSION_REFRESH_EACH_REQUEST'] = True  # 每次请求刷新过期时间
app.secret_key = 'your-secret-key-here-change-this'

# 🔥 新增：让Flask感知反向代理的HTTPS
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)


# 文件夹配置
UPLOAD_FOLDER = "uploads"
OUT_FOLDER = "outputs"
TEMP_FOLDER = "temp_processing"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# 加载配置数据
VALID_CODES = load_valid_codes()
BEAD_COLORS = load_bead_colors()
MARD_MATCHER = MardColorMatcher()  # ✅ 添加这一行

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # ✅ 检查是否已认证
        if not session.get('authenticated'):
            if request.path.startswith('/api/'):
                return jsonify({'success': False, 'error': '请先登录', 'need_login': True}), 401
            else:
                return redirect(url_for('login'))
        
        # ✅ 额外检查：验证码是否存在（防止 session 伪造）
        user_code = session.get('user_code')
        if not user_code:
            session.clear()
            if request.path.startswith('/api/'):
                return jsonify({'success': False, 'error': '登录已失效，请重新登录', 'need_login': True}), 401
            else:
                return redirect(url_for('login'))
        
        return f(*args, **kwargs)
    return decorated_function

def calculate_color_stats(pixel_img):
    """统计图像中的颜色并匹配最接近的拼豆颜色"""
    pixels = np.array(pixel_img).reshape(-1, 3)
    unique_colors, unique_counts = np.unique(pixels, axis=0, return_counts=True)
    total_pixels = pixels.shape[0]
    
    color_map = {}
    for color, count in zip(unique_colors, unique_counts):
        # ✅ 使用 MardColorMatcher
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        matched = MARD_MATCHER.find_closest_color(r, g, b)
        
        key = matched['name']  # 使用 Mard 色号作为 key
        if key in color_map:
            color_map[key]['count'] += int(count)
        else:
            color_map[key] = {
                'code': matched['name'],
                'name': matched['name'],
                'hex': '#' + matched['hex'],
                'count': int(count)
            }
    
    stats = list(color_map.values())
    stats.sort(key=lambda x: -x['count'])
    
    for item in stats:
        item['percentage'] = f"{(item['count'] / total_pixels * 100):.1f}%"
    
    return stats

# ==================== 登录路由 ====================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        code = request.form.get('code', '').strip()
        
        if not code:
            return render_template('login.html', error='请输入验证码')
        
        # 使用新的验证逻辑
        result = validate_code(code)
        
        if result['valid']:
            # ✅ 清除旧的 session，重新开始
            session.clear()
            
            # ✅ 设置新的 session（不设置 permanent）
            session['authenticated'] = True
            session['user_code'] = code
            session['remaining_times'] = result['remaining']
            # ❌ 不要设置：session.permanent = True
            
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error=result['message'])
    
    # GET 请求
    if session.get('authenticated'):
        # 如果已登录，重定向到首页
        return redirect(url_for('index'))
    
    return render_template('login.html')

# 路由：退出登录
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    # 获取最新的剩余次数
    user_code = session.get('user_code')
    if user_code:
        remaining = get_code_remaining(user_code)
        session['remaining_times'] = remaining
    else:
        remaining = session.get('remaining_times', 0)
    
    return render_template('index.html', remaining_times=remaining)

# API: 处理步骤1 - 上传图片和背景去除
@app.route('/api/process-step1', methods=['POST'])
@login_required
def process_step1():
    try:
        f = request.files.get('file')
        if not f:
            return jsonify({'success': False, 'error': '没有上传文件'})
        
        # 保存原始文件
        filename = secure_filename(f.filename)
        ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        session['timestamp'] = ts
        
        original_path = os.path.join(TEMP_FOLDER, f"{ts}_original.png")
        f.save(original_path)
        
       # 处理图片
        img = Image.open(original_path)
        
        # 保存原图URL供前端显示
        session['original_image'] = original_path
        
     
        img_processed = img.convert('RGB')
        
        # 保存处理后的图片
        step1_path = os.path.join(TEMP_FOLDER, f"{ts}_step1.png")
        img_processed.save(step1_path, format='PNG')
        
        session['step1_image'] = step1_path
        
        return jsonify({
            'success': True,
            'image_url': url_for('send_file_from_path', p=step1_path)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
# API: 线条增强预览
@app.route('/api/enhance-lines-preview', methods=['POST'])
@login_required
def enhance_lines_preview():
    """实时预览线条增强效果"""
    try:
        # 获取上传的图片
        f = request.files.get('file')
        if not f:
            return jsonify({'success': False, 'error': '没有上传文件'})
        
        # 获取线条强度参数
        line_strength = int(request.form.get('line_strength', 3))
        
        # 保存临时文件
        import time
        ts = int(time.time() * 1000)
        temp_input = os.path.join(TEMP_FOLDER, f"{ts}_input.png")
        temp_output = os.path.join(TEMP_FOLDER, f"{ts}_enhanced.png")
        
        # 保存原始图片
        img = Image.open(f)
        img.save(temp_input, format='PNG')
        
        # 调用线条增强函数
        from methods import enhance_lines
        enhance_lines(temp_input, temp_output, line_strength=line_strength)
        
        # 返回增强后的图片URL
        return jsonify({
            'success': True,
            'image_url': url_for('send_file_from_path', p=temp_output, t=ts)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})
@app.route('/api/process-step2', methods=['POST'])
@login_required
def process_step2():
    try:
        if not PHOTO2PIXEL_AVAILABLE:
            return jsonify({'success': False, 'error': '像素画功能不可用'})
        
        step1_image = session.get('step1_image')
        if not step1_image or not os.path.exists(step1_image):
            return jsonify({'success': False, 'error': '请先完成步骤1'})
        
        data = request.get_json()
        pixels_per_row = data.get('pixels_per_row', 25)
        algorithm = data.get('algorithm', 'pixel_segment')
        edge_thresh = data.get('edge_thresh', 80)
        
        # 日志输出
        print("=" * 70)
        print(f"[API] 步骤2 - 像素画生成")
        print(f"[API] 收到参数: pixels_per_row={pixels_per_row}, algorithm={algorithm}, edge_thresh={edge_thresh}")
        
        # 准备输出路径
        ts = session.get('timestamp')
        import time
        cache_buster = int(time.time() * 1000) % 10000
        step2_path = os.path.join(TEMP_FOLDER, f"{ts}_step2_{cache_buster}.png")
        
        # 根据算法调用对应的方法
        if algorithm == 'original':
            # 简单像素化（非AI）
            print(f"[API] 使用算法: 简单像素化")
            img = Image.open(step1_image)
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            else:
                img = img.convert('RGB')
            
            w, h = img.size
            pixel_height = max(1, int(h * pixels_per_row / w))
            small = img.resize((pixels_per_row, pixel_height), Image.LANCZOS)
            pixel_img = small.resize((w, h), Image.NEAREST)
            pixel_img.save(step2_path, format='PNG')
            
        elif algorithm == 'pixel_effect':
            # 像素化效果（颜色平滑）
            print(f"[API] 使用算法: 像素化效果（颜色平滑）")
            pixel_effect(
                input_image_path=step1_image,
                pixels_per_row=pixels_per_row,
                output_image_path=step2_path
            )
            
        elif algorithm == 'pixel_segment':
            # 分割像素效果（色块分明）
            print(f"[API] 使用算法: 分割像素效果（色块分明）")
            pixel_segment(
                input_image_path=step1_image,
                pixels_per_row=pixels_per_row,
                output_image_path=step2_path
            )
            
        elif algorithm == 'edge_detect':
            # 边缘强化效果
            print(f"[API] 使用算法: 边缘强化效果（强度={edge_thresh}）")
            edge_enhance(
                input_image_path=step1_image,
                pixels_per_row=pixels_per_row,
                edge_strength=edge_thresh,
                output_image_path=step2_path
            )
            
        else:
            return jsonify({'success': False, 'error': f'未知算法: {algorithm}'})
        
        print(f"[API] 像素画已保存: {step2_path}")
        print("=" * 70)
        
        # 保存像素画信息到session
        session['step2_image'] = step2_path
        session['pixels_per_row'] = pixels_per_row
        session['algorithm'] = algorithm
        
        return jsonify({
            'success': True,
            'image_url': url_for('send_file_from_path', p=step2_path, t=cache_buster)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})
# API: 处理步骤3 - 生成初步图纸
@app.route('/api/process-step3', methods=['POST'])
@login_required
def process_step3():
    try:
        step2_image = session.get('step2_image')
        if not step2_image or not os.path.exists(step2_image):
            return jsonify({'success': False, 'error': '请先完成步骤2'})
        
        # 加载像素画（步骤2生成的是原图尺寸的像素画）
        pixel_img_large = Image.open(step2_image).convert('RGB')
        w_large, h_large = pixel_img_large.size
        
        # 获取步骤2的尺寸参数
        pixels_per_row = session.get('pixels_per_row', 50)
        
        # 将像素画缩小到实际的像素块数量（用于颜色统计）
        pixel_height = max(1, int(h_large * pixels_per_row / w_large))
        pixel_img_small = pixel_img_large.resize((pixels_per_row, pixel_height), Image.NEAREST)
        w0, h0 = pixel_img_small.size  # 这才是真实的像素块数量
        
        # 计算网格大小（每个像素块/拼豆的显示大小）
        pixel_per_bead = max(15, min(40, 1000 // pixels_per_row))
        
        # 使用缩小的像素画来绘制网格
        board_width = w0 * pixel_per_bead
        board_height = h0 * pixel_per_bead
        board = pixel_img_small.resize((board_width, board_height), Image.NEAREST)
        
        # 绘制网格线
        bd_arr = np.array(board)
        for y in range(0, board_height, pixel_per_bead):
            if y < board_height:
                bd_arr[y:min(y+1, board_height), :, :] = [220, 220, 220]  # 浅灰色网格
        for x in range(0, board_width, pixel_per_bead):
            if x < board_width:
                bd_arr[:, x:min(x+1, board_width), :] = [220, 220, 220]
        board = Image.fromarray(bd_arr)
        
        # 保存文件
        ts = session.get('timestamp')
        base = f"{ts}_draft"
        pixel_path_display = os.path.join(OUT_FOLDER, base + "_pixel_display.png")  # 原图尺寸用于显示
        pixel_path_small = os.path.join(OUT_FOLDER, base + "_pixel_small.png")     # 缩小版用于统计
        board_path = os.path.join(OUT_FOLDER, base + "_board.png")
        
        pixel_img_large.save(pixel_path_display, format='PNG')  # 保存原图尺寸的像素画
        pixel_img_small.save(pixel_path_small, format='PNG')    # 保存缩小版
        board.save(board_path, format='PNG')
        
        # 保存到session
        session['draft_data'] = {
            'pixel_path_display': pixel_path_display,  # 用于显示的大图
            'pixel_path': pixel_path_small,            # 用于颜色统计的小图
            'board_path': board_path,
            'image_width': w0,
            'image_height': h0,
            'pixels_per_row': pixels_per_row,
            'pixel_per_bead': pixel_per_bead
        }
        
        # 返回JSON而不是跳转页面
        return jsonify({
            'success': True,
            'pixel_url': url_for('send_file_from_path', p=pixel_path_display),
            'board_url': url_for('send_file_from_path', p=board_path),
            'width': w0,
            'height': h0
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})
@app.route('/api/apply-clustering', methods=['POST'])
@login_required
def apply_clustering():
    try:
        data = request.get_json()
        algorithm = data.get('algorithm', 'none')
        color_count = int(data.get('color_count', 16))
        
        print("=" * 70)
        print(f"[API] 开始应用聚类算法: {algorithm}, 颜色数: {color_count}")
        
        # 【补充这部分代码】获取草稿数据
        draft_data = session.get('draft_data')
        if not draft_data:
            return jsonify({'success': False, 'error': '未找到草稿数据，请重新生成'})
        
        # 加载像素图小图（用于颜色处理）
        pixel_img_small_path = draft_data['pixel_path']
        pixel_img = Image.open(pixel_img_small_path).convert('RGB')
        
        # 获取图像尺寸
        w0, h0 = pixel_img.size
        
        if algorithm == 'none':
            # 不聚类 - 保留原始颜色，但要标注色号
            print(f"[API] 不聚类处理（保留原色+标注色号）")
            thumb = pixel_img.copy()
            
            # 统计所有颜色
            pixels = np.array(thumb).reshape(-1, 3)
            unique_colors, unique_counts = np.unique(pixels, axis=0, return_counts=True)
            
            # 按数量排序
            sorted_indices = np.argsort(-unique_counts)
            
            # 创建颜色到色号的映射字典（使用 MardColorMatcher）
            color_to_code = {}
            color_stats = []
            total_beads = w0 * h0
            
            for idx in sorted_indices:
                rgb = unique_colors[idx]
                count = unique_counts[idx]
                percentage = (count / total_beads) * 100
                
                # 实际颜色值
                hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
                
                # ✅ 替换为
                matched = MARD_MATCHER.find_closest_color(int(rgb[0]), int(rgb[1]), int(rgb[2]))
                color_to_code[color_key] = matched['name']

                color_stats.append({
                    'code': matched['name'],
                    'name': matched['name'],
                    'hex': hex_color,
                    'bead_hex': '#' + matched['hex'],
                    'count': int(count),
                    'percentage': f'{percentage:.1f}%'
                })
            
        elif algorithm == 'smart':
            # 智能聚类 - 先聚类，再匹配色号
            print(f"[API] 智能聚类（聚类后匹配色号）")
            
            thumb = advanced_color_quantization(pixel_img, color_count)
            
            pixels = np.array(thumb).reshape(-1, 3)
            unique_colors, unique_counts = np.unique(pixels, axis=0, return_counts=True)
            
            sorted_indices = np.argsort(-unique_counts)
            
            # 创建颜色到色号的映射（使用 MardColorMatcher）
            color_to_code = {}
            color_stats = []
            total_beads = w0 * h0
            
            for idx in sorted_indices:
                rgb = unique_colors[idx]
                count = unique_counts[idx]
                percentage = (count / total_beads) * 100
                
                hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
                
                # ✅ 替换为
                matched = MARD_MATCHER.find_closest_color(int(rgb[0]), int(rgb[1]), int(rgb[2]))
                color_to_code[color_key] = matched['name']

                color_stats.append({
                    'code': matched['name'],
                    'name': matched['name'],
                    'hex': hex_color,
                    'bead_hex': '#' + matched['hex'],
                    'count': int(count),
                    'percentage': f'{percentage:.1f}%'
                })
        else:
            return jsonify({'success': False, 'error': '未知的算法'})
        
        
        # 生成最终图纸（带色号标注）
        pixel_per_bead = draft_data['pixel_per_bead']
        board_width = w0 * pixel_per_bead
        board_height = h0 * pixel_per_bead
        board = thumb.resize((board_width, board_height), Image.NEAREST)
        
        # 绘制网格线
        bd_arr = np.array(board)
        for y in range(0, board_height, pixel_per_bead):
            if y < board_height:
                bd_arr[y:min(y+1, board_height), :, :] = [0, 0, 0]
        for x in range(0, board_width, pixel_per_bead):
            if x < board_width:
                bd_arr[:, x:min(x+1, board_width), :] = [0, 0, 0]
        board = Image.fromarray(bd_arr)
        
        # 【新增】在每个格子中标注色号
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(board)
        
        # 尝试加载字体
        try:
            # 尝试不同的字体路径
            font_size = max(8, pixel_per_bead // 3)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        # 在每个格子中绘制色号
        thumb_arr = np.array(thumb)
        for y in range(h0):
            for x in range(w0):
                # 获取该位置的颜色
                pixel_color = tuple(thumb_arr[y, x])
                bead_code = color_to_code.get(pixel_color, '?')
                
                # 计算格子中心位置
                center_x = x * pixel_per_bead + pixel_per_bead // 2
                center_y = y * pixel_per_bead + pixel_per_bead // 2
                
                # 计算文字大小和位置（居中）
                bbox = draw.textbbox((0, 0), bead_code, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                text_x = center_x - text_width // 2
                text_y = center_y - text_height // 2
                
                # 判断背景颜色深浅，选择文字颜色
                brightness = (pixel_color[0] * 0.299 + pixel_color[1] * 0.587 + pixel_color[2] * 0.114)
                text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
                
                # 绘制文字
                draw.text((text_x, text_y), bead_code, fill=text_color, font=font)
        
        # 添加坐标
        board = add_coordinates_to_board(board, w0, h0, pixel_per_bead)
        
        # 保存文件
        ts = session.get('timestamp')
        base = f"{ts}_final_{algorithm}"
        thumb_path = os.path.join(OUT_FOLDER, base + "_thumb.png")
        board_path = os.path.join(OUT_FOLDER, base + "_board.png")
        
        # 保存CSV
        csv_path = os.path.join(OUT_FOLDER, base + "_colors.csv")
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['色号', '名称', '实际色值', '拼豆色值(参考)', '数量', '占比'])
            for stat in color_stats:
                writer.writerow([
                    stat['code'], 
                    stat['name'], 
                    stat['hex'], 
                    stat.get('bead_hex', stat['hex']),
                    stat['count'], 
                    stat['percentage']
                ])
        
        thumb.save(thumb_path, format='PNG')
        board.save(board_path, format='PNG')
        
        print(f"[API] 颜色统计完成，共 {len(color_stats)} 种颜色")
        print("=" * 70)
        
        # 保存到session
        session['final_data'] = {
            'thumb_path': thumb_path,
            'board_path': board_path,
            'csv_path': csv_path,
            'color_stats': color_stats[:50],
            'total_colors': len(color_stats),
            'algorithm': '智能聚类' if algorithm == 'smart' else '不聚类',
            'image_width': w0,
            'image_height': h0
        }
        
        return jsonify({'success': True})
        
    except Exception as e:
        import traceback
        print(f"[API] apply_clustering 错误: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})
# 路由：文件服务
@app.route('/file')
@login_required  #
def send_file_from_path():
    p = request.args.get('p')
    dl = request.args.get('dl', None)
    if not p or not os.path.exists(p):
        return "File not found", 404
    return send_file(p, as_attachment=bool(dl))

# 路由：批量下载
@app.route('/download_all')
@login_required
def download_all():
    """提供所有生成文件的打包下载"""
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    
    try:
        with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
            if os.path.exists(OUT_FOLDER):
                files = os.listdir(OUT_FOLDER)
                files.sort(key=lambda x: os.path.getmtime(os.path.join(OUT_FOLDER, x)), reverse=True)
                
                added_files = 0
                for file in files[:30]:
                    file_path = os.path.join(OUT_FOLDER, file)
                    if os.path.isfile(file_path):
                        zipf.write(file_path, file)
                        added_files += 1
        
        if added_files > 0:
            return send_file(temp_zip.name, 
                           as_attachment=True, 
                           download_name=f'拼豆图纸_批量下载_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.zip',
                           mimetype='application/zip')
        else:
            return "没有找到可下载的文件", 404
            
    except Exception as e:
        return f"打包失败: {str(e)}", 500
    finally:
        try:
            os.unlink(temp_zip.name)
        except:
            pass
# 路由：显示图纸
# 路由：显示图纸
@app.route('/draft')
@login_required
def show_draft():
    draft_data = session.get('draft_data')
    if not draft_data:
        return redirect(url_for('index'))
    
    # 构建URL - 使用大图用于显示
    pixel_url = url_for('send_file_from_path', p=draft_data['pixel_path_display'])
    board_url = url_for('send_file_from_path', p=draft_data['board_path'])
    
    return render_template(
        'draft.html',
        pixel_url=pixel_url,
        board_url=board_url,
        image_width=draft_data['image_width'],
        image_height=draft_data['image_height']
    )

# 路由：显示最终结果
@app.route('/result')
@login_required
def show_result():
    final_data = session.get('final_data')
    if not final_data:
        return redirect(url_for('show_draft'))
    
    # 构建URL
    thumb_url = url_for('send_file_from_path', p=final_data['thumb_path'])
    board_url = url_for('send_file_from_path', p=final_data['board_path'])
    download_thumb = url_for('send_file_from_path', p=final_data['thumb_path'], dl=1)
    download_board = url_for('send_file_from_path', p=final_data['board_path'], dl=1)
    download_csv = url_for('send_file_from_path', p=final_data['csv_path'], dl=1)
    
    return render_template(
        'result.html',
        thumb_url=thumb_url,
        board_url=board_url,
        color_stats=final_data['color_stats'],
        total_colors=final_data['total_colors'],
        algorithm=final_data['algorithm'],
        image_width=final_data['image_width'],
        image_height=final_data['image_height'],
        download_thumb=download_thumb,
        download_board=download_board,
        download_csv=download_csv
    )

@app.route('/api/get-pixel-data', methods=['GET'])
@login_required
def get_pixel_data():
    """获取像素数据用于编辑"""
    try:
        draft_data = session.get('draft_data')
        if not draft_data:
            return jsonify({'success': False, 'error': '未找到数据'})
        
        # 读取小图
        pixel_img = Image.open(draft_data['pixel_path']).convert('RGB')
        pixels = np.array(pixel_img)
        
        # 转换为列表格式 [[[r,g,b], [r,g,b], ...], ...]
        pixel_data = pixels.tolist()
        
        # 返回拼豆色库
        bead_colors = BEAD_COLORS
        
        return jsonify({
            'success': True,
            'pixel_data': pixel_data,
            'bead_colors': bead_colors,
            'width': draft_data['image_width'],
            'height': draft_data['image_height']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/save-edited-pattern', methods=['POST'])
@login_required
def save_edited_pattern():
    """保存用户编辑后的像素图"""
    try:
        data = request.get_json()
        pixel_data = data.get('pixel_data')
        color_stats = data.get('color_stats')  # ✅ 接收前端的统计信息
        
        # 转换回图像
        img_array = np.array(pixel_data, dtype=np.uint8)
        pixel_img = Image.fromarray(img_array)
        
        # 获取原始draft_data
        draft_data = session.get('draft_data')
        pixel_per_bead = draft_data['pixel_per_bead']
        w0 = draft_data['image_width']
        h0 = draft_data['image_height']
        
        # 重新生成带网格的图纸
        h, w = img_array.shape[:2]
        board_width = w * pixel_per_bead
        board_height = h * pixel_per_bead
        board = pixel_img.resize((board_width, board_height), Image.NEAREST)
        
        # 绘制网格
        bd_arr = np.array(board)
        for y in range(0, board_height, pixel_per_bead):
            bd_arr[y:min(y+1, board_height), :, :] = [0, 0, 0]
        for x in range(0, board_width, pixel_per_bead):
            bd_arr[:, x:min(x+1, board_width), :] = [0, 0, 0]
        board = Image.fromarray(bd_arr)
        
        # 添加色号标注
        from PIL import ImageDraw, ImageFont
        
        draw = ImageDraw.Draw(board)
        
        # 加载字体
        try:
            font_size = max(8, pixel_per_bead // 3)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        # 创建颜色到色号的映射并标注（使用 MardColorMatcher）
        color_to_code = {}
        for y in range(h0):
            for x in range(w0):
                pixel_color = tuple(img_array[y, x])
                
                if pixel_color not in color_to_code:
                    matched = MARD_MATCHER.find_closest_color(
                        int(pixel_color[0]), 
                        int(pixel_color[1]), 
                        int(pixel_color[2])
                    )
                    color_to_code[pixel_color] = matched['name']

                bead_code = color_to_code[pixel_color]
                
                # 计算格子中心位置
                center_x = x * pixel_per_bead + pixel_per_bead // 2
                center_y = y * pixel_per_bead + pixel_per_bead // 2
                
                # 计算文字位置（居中）
                bbox = draw.textbbox((0, 0), bead_code, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                text_x = center_x - text_width // 2
                text_y = center_y - text_height // 2
                
                # 判断背景颜色深浅，选择文字颜色
                brightness = (pixel_color[0] * 0.299 + pixel_color[1] * 0.587 + pixel_color[2] * 0.114)
                text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
                
                # 绘制文字
                draw.text((text_x, text_y), bead_code, fill=text_color, font=font)
        
        # 添加坐标
        board = add_coordinates_to_board(board, w0, h0, pixel_per_bead)
        
        # 保存新图纸
        ts = session.get('timestamp')
        import time
        cache_buster = int(time.time() * 1000) % 10000
        new_board_path = os.path.join(OUT_FOLDER, f"{ts}_edited_board_{cache_buster}.png")
        board.save(new_board_path, format='PNG')
        
        # 保存小图
        new_pixel_path = os.path.join(OUT_FOLDER, f"{ts}_edited_pixel_{cache_buster}.png")
        pixel_img.save(new_pixel_path, format='PNG')
        
        # 更新draft_data
        draft_data['pixel_path'] = new_pixel_path
        draft_data['board_path'] = new_board_path
        session['draft_data'] = draft_data
        
        # ✅ 不重新统计，直接使用前端传来的统计信息
        # color_stats = calculate_color_stats(pixel_img)  # 删除这行
        
        # ✅ 保存CSV（使用前端统计）
        if color_stats:
            csv_path = os.path.join(OUT_FOLDER, f"{ts}_edited_colors_{cache_buster}.csv")
            import csv
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['色号', '名称', '色值', '数量', '占比'])
                for stat in color_stats:
                    writer.writerow([
                        stat['code'], 
                        stat['name'], 
                        stat['hex'], 
                        stat['count'], 
                        stat['percentage']
                    ])
        
        return jsonify({
            'success': True,
            'board_url': url_for('send_file_from_path', p=new_board_path, t=cache_buster),
            'download_url': url_for('send_file_from_path', p=new_board_path, dl=1),
            'csv_download_url': url_for('send_file_from_path', p=csv_path, dl=1) if color_stats else None,  # ← 添加这行
            'color_stats': color_stats
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})
# API: 检查session状态
@app.route('/api/check-session', methods=['GET'])
@login_required
def check_session():
    step1_image = session.get('step1_image')
    step2_image = session.get('step2_image')
    draft_data = session.get('draft_data')
    
    result = {
        'has_step1': step1_image is not None and os.path.exists(step1_image) if step1_image else False,
        'has_step2': step2_image is not None and os.path.exists(step2_image) if step2_image else False,
        'has_step3': draft_data is not None
    }
    
    if result['has_step1']:
        result['step1_url'] = url_for('send_file_from_path', p=step1_image)
    
    if result['has_step2']:
        result['step2_url'] = url_for('send_file_from_path', p=step2_image)
        result['pixels_per_row'] = session.get('pixels_per_row', 50)
        result['algorithm'] = session.get('algorithm', 'photo2pixel')
    
    if result['has_step3']:
        result['step3_pixel_url'] = url_for('send_file_from_path', p=draft_data['pixel_path_display'])
        result['step3_board_url'] = url_for('send_file_from_path', p=draft_data['board_path'])
        result['step3_width'] = draft_data['image_width']
        result['step3_height'] = draft_data['image_height']
    
    return jsonify(result)
@app.route('/api/generate-pattern', methods=['POST'])
@login_required
def generate_pattern():
    try:
        # 获取用户验证码
        user_code = session.get('user_code')
        if not user_code:
            return jsonify({'success': False, 'error': '未找到验证码，请重新登录'})
        
        # 检查剩余次数
        remaining = get_code_remaining(user_code)
        if remaining <= 0:
            return jsonify({
                'success': False, 
                'error': '本月使用次数已用完，将在下月初重置为100次'
            })
        
        # 获取参数
        f = request.files.get('file')
        if not f:
            return jsonify({'success': False, 'error': '没有上传文件'})

        bead_width = int(request.form.get('bead_width', 30))
        use_cluster = request.form.get('use_cluster') == 'true'
        line_strength = int(request.form.get('line_strength', 0))
        use_line_enhance = request.form.get('use_line_enhance') == 'true'

        print("=" * 70)
        print(f"[一体化生成] 用户: {user_code}, 剩余次数: {remaining}")
        print(f"[参数] 宽度={bead_width}, 聚类={use_cluster}, 线条增强={use_line_enhance}, 强度={line_strength}")

        # 2. 生成时间戳
        ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        session['timestamp'] = ts
        
        # 3. 保存原始文件
        filename = secure_filename(f.filename)
        original_path = os.path.join(TEMP_FOLDER, f"{ts}_original.png")
        f.save(original_path)
        
        # 4. 加载图片
        # 4. 加载并检查图片尺寸
        try:
            img = Image.open(original_path)
            img = img.convert('RGB')
            
            # 简单尺寸检查
            from methods import check_image_size
            is_valid, error_msg = check_image_size(img, max_width=2048, max_height=2048)
            
            if not is_valid:
                print(f"[错误] {error_msg}")
                return jsonify({'success': False, 'error': error_msg})
            
            print(f"[处理] 图像尺寸检查通过: {img.size}")

        except Exception as e:
            print(f"[错误] 图像处理失败: {str(e)}")
            return jsonify({'success': False, 'error': f'图像格式不支持或文件损坏: {str(e)}'})

        # 5. 应用线条增强（✅ 正确：在img定义之后）
        if use_line_enhance and line_strength > 0:
            print(f"[处理] 应用线条增强，强度={line_strength}")
            temp_before = os.path.join(TEMP_FOLDER, f"{ts}_before_enhance.png")
            temp_after = os.path.join(TEMP_FOLDER, f"{ts}_after_enhance.png")
            img.save(temp_before, format='PNG')
            
            from methods import enhance_lines
            enhance_lines(temp_before, temp_after, line_strength=line_strength)
            
            img = Image.open(temp_after)
            img = img.convert('RGB')  # 确保格式一致

        # 6. 继续后续处理...
        step1_path = os.path.join(TEMP_FOLDER, f"{ts}_processed.png")
        img.save(step1_path, format='PNG')

        # 5. 生成像素画（固定使用分割像素算法）
        pixel_path_large = os.path.join(TEMP_FOLDER, f"{ts}_pixel_large.png")

        pixel_segment(
                input_image_path=step1_path,
                pixels_per_row=bead_width,
                output_image_path=pixel_path_large
        )
           

        # 6. 缩小到实际像素块数量
        pixel_img_large = Image.open(pixel_path_large).convert('RGB')  # 现在转RGB是安全的
        w_large, h_large = pixel_img_large.size
        pixel_height = max(1, int(h_large * bead_width / w_large))
        pixel_img = pixel_img_large.resize((bead_width, pixel_height), Image.NEAREST)
        w0, h0 = pixel_img.size
        
        print(f"[处理] 像素画尺寸: {w0}×{h0}")
        
        # 7. 颜色处理（是否聚类）
        if use_cluster:
            print(f"[处理] 应用智能颜色聚类...")
            pixel_img = advanced_color_quantization(pixel_img, n_colors=16)
        else:
            print(f"[处理] 不聚类，保留原始颜色")
        
        # 8. 生成图纸
        pixel_per_bead = max(15, min(40, 1000 // bead_width))
        board_width = w0 * pixel_per_bead
        board_height = h0 * pixel_per_bead
        board = pixel_img.resize((board_width, board_height), Image.NEAREST)
        
        # 9. 绘制网格
        bd_arr = np.array(board)
        for y in range(0, board_height, pixel_per_bead):
            if y < board_height:
                bd_arr[y:min(y+1, board_height), :, :] = [200, 200, 200]  # 浅灰色网格
        for x in range(0, board_width, pixel_per_bead):
            if x < board_width:
                bd_arr[:, x:min(x+1, board_width), :] = [200, 200, 200]
        board = Image.fromarray(bd_arr)
        
        # 10. 标注色号
        from PIL import ImageDraw, ImageFont
        
        draw = ImageDraw.Draw(board)
        
        # 加载字体
        try:
            font_size = max(8, pixel_per_bead // 3)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        # 创建颜色映射并标注（使用 MardColorMatcher）
        pixel_arr = np.array(pixel_img)
        color_to_code = {}
        
        for y in range(h0):
            for x in range(w0):
                pixel_color = tuple(pixel_arr[y, x])
                
                # 找最接近的拼豆色号
                # ✅ 替换为
                if pixel_color not in color_to_code:
                    matched = MARD_MATCHER.find_closest_color(
                        int(pixel_color[0]),
                        int(pixel_color[1]),
                        int(pixel_color[2])
                    )
                    color_to_code[pixel_color] = matched['name']

                bead_code = color_to_code[pixel_color]
                
                # 计算格子中心位置
                center_x = x * pixel_per_bead + pixel_per_bead // 2
                center_y = y * pixel_per_bead + pixel_per_bead // 2
                
                # 计算文字位置
                bbox = draw.textbbox((0, 0), bead_code, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_x = center_x - text_width // 2
                text_y = center_y - text_height // 2
                
                # 根据背景亮度选择文字颜色
                brightness = (pixel_color[0] * 0.299 + pixel_color[1] * 0.587 + pixel_color[2] * 0.114)
                text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
                
                # 绘制色号
                draw.text((text_x, text_y), bead_code, fill=text_color, font=font)
        
        # 11. 添加坐标
        board = add_coordinates_to_board(board, w0, h0, pixel_per_bead)
        
        # 12. 计算颜色统计
        print(f"[处理] 计算颜色统计...")
        color_stats = calculate_color_stats(pixel_img)
        
        # 13. 保存文件
        base = f"{ts}_final"
        thumb_path = os.path.join(OUT_FOLDER, base + "_thumb.png")
        board_path = os.path.join(OUT_FOLDER, base + "_board.png")
        pixel_small_path = os.path.join(OUT_FOLDER, base + "_pixel.png")
        csv_path = os.path.join(OUT_FOLDER, base + "_colors.csv")
        
        pixel_img.save(pixel_small_path, format='PNG')
        board.save(board_path, format='PNG')
        
        # 保存CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['色号', '名称', '色值', '数量', '占比'])
            for stat in color_stats:
                writer.writerow([
                    stat['code'],
                    stat['name'],
                    stat['hex'],
                    stat['count'],
                    stat['percentage']
                ])
        
        print(f"[完成] 生成成功！共 {len(color_stats)} 种颜色")
        print("=" * 70)
        
        # 14. 保存到session
        session['draft_data'] = {
            'pixel_path': pixel_small_path,
            'pixel_path_display': pixel_small_path,
            'board_path': board_path,
            'image_width': w0,
            'image_height': h0,
            'pixels_per_row': bead_width,
            'pixel_per_bead': pixel_per_bead
        }
        
        session['final_data'] = {
            'thumb_path': pixel_small_path,
            'board_path': board_path,
            'csv_path': csv_path,
            'color_stats': color_stats[:50],
            'total_colors': len(color_stats),
            'algorithm': '智能聚类' if use_cluster else '不聚类',
            'image_width': w0,
            'image_height': h0
        }
        
         # ✅ 图纸生成成功后，扣减次数
        consume_result = consume_code_usage(user_code)
        
        if consume_result['success']:
            # 更新 session 中的剩余次数
            session['remaining_times'] = consume_result['remaining']
            print(f"[扣减成功] {consume_result['message']}")
        else:
            print(f"[警告] 扣减失败: {consume_result['message']}")
        
        # 返回成功（包含剩余次数）
        return jsonify({
            'success': True,
            'remaining_times': consume_result['remaining'],
            'message': consume_result['message']
        })
        
    except Exception as e:
        import traceback
        print(f"[错误] 生成失败: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})
# ==================== 新增：查询剩余次数 API ====================

@app.route('/api/get-remaining', methods=['GET'])
@login_required
def get_remaining():
    """获取当前用户剩余次数"""
    user_code = session.get('user_code')
    if not user_code:
        return jsonify({'success': False, 'error': '未登录'})
    
    remaining = get_code_remaining(user_code)
    session['remaining_times'] = remaining
    
    return jsonify({
        'success': True,
        'remaining': remaining,
        'code': user_code[-4:]  # 只显示后4位
    })
if __name__ == '__main__':
    # 开发模式
    app.run(host="0.0.0.0", port=5001, debug=True)
else:
    # 生产模式配置
    app.config['DEBUG'] = False
    app.config['TESTING'] = False
    # 可以添加更多生产环境配置
