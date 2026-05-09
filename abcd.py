import os
import random
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np


def ensure_dir(path):
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)


def load_image(path):
    """加载图片并转换为RGBA模式"""
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")
    return Image.open(path).convert("RGBA")


def detect_frame_bounds(frame_img):
    """检测车框图片中黑色边框的内边界和外边界"""
    img_array = np.array(frame_img)
    # 检测深色像素（放宽阈值）
    dark_mask = (img_array[:, :, 0] < 80) & (img_array[:, :, 1] < 80) & (img_array[:, :, 2] < 80)
    dark_coords = np.where(dark_mask)
    
    if len(dark_coords[0]) == 0:
        h, w = img_array.shape[:2]
        # 默认框内区域为图片的30%-70%
        inner = (int(w*0.15), int(h*0.15), int(w*0.85), int(h*0.85))
        outer = (int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9))
        return inner, outer
    
    y_min, y_max = dark_coords[0].min(), dark_coords[0].max()
    x_min, x_max = dark_coords[1].min(), dark_coords[1].max()
    
    # 外边界：深色区域的最外边缘
    outer = (x_min, y_min, x_max, y_max)
    
    # 内边界：尝试检测边框内侧（寻找深色到浅色的过渡）
    # 从外边界向内收缩一定比例来估算内边界
    border_thickness = min(x_max - x_min, y_max - y_min) * 0.05
    inner = (
        int(x_min + border_thickness),
        int(y_min + border_thickness),
        int(x_max - border_thickness),
        int(y_max - border_thickness)
    )
    
    return inner, outer


def create_colorful_background(size):
    """创建五颜六色的背景图"""
    img = Image.new("RGBA", size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    bg_type = random.choice(['solid', 'gradient', 'checker', 'stripes', 'dots'])
    
    if bg_type == 'solid':
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255), 255)
        draw.rectangle([0, 0, size[0], size[1]], fill=color)
        
    elif bg_type == 'gradient':
        color1 = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255), 255)
        color2 = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255), 255)
        for i in range(size[1]):
            ratio = i / size[1]
            r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            draw.line([(0, i), (size[0], i)], fill=(r, g, b, 255))
            
    elif bg_type == 'checker':
        color1 = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255), 255)
        color2 = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200), 255)
        cell_size = random.randint(20, 50)
        for y in range(0, size[1], cell_size):
            for x in range(0, size[0], cell_size):
                if (x // cell_size + y // cell_size) % 2 == 0:
                    draw.rectangle([x, y, x + cell_size, y + cell_size], fill=color1)
                else:
                    draw.rectangle([x, y, x + cell_size, y + cell_size], fill=color2)
                    
    elif bg_type == 'stripes':
        color1 = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255), 255)
        color2 = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200), 255)
        stripe_width = random.randint(10, 30)
        is_horizontal = random.choice([True, False])
        if is_horizontal:
            for y in range(0, size[1], stripe_width):
                if (y // stripe_width) % 2 == 0:
                    draw.rectangle([0, y, size[0], y + stripe_width], fill=color1)
                else:
                    draw.rectangle([0, y, size[0], y + stripe_width], fill=color2)
        else:
            for x in range(0, size[0], stripe_width):
                if (x // stripe_width) % 2 == 0:
                    draw.rectangle([x, 0, x + stripe_width, size[1]], fill=color1)
                else:
                    draw.rectangle([x, 0, x + stripe_width, size[1]], fill=color2)
                    
    else:  # dots
        bg_color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200), 255)
        dot_color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255), 255)
        draw.rectangle([0, 0, size[0], size[1]], fill=bg_color)
        spacing = random.randint(30, 60)
        dot_radius = random.randint(5, 15)
        for y in range(spacing // 2, size[1], spacing):
            for x in range(spacing // 2, size[0], spacing):
                draw.ellipse([x - dot_radius, y - dot_radius, 
                            x + dot_radius, y + dot_radius], fill=dot_color)
    
    return img


def scale_car_to_canvas(car_img, canvas_size, scale_range=(0.6, 0.8)):
    """缩放车辆"""
    canvas_w, canvas_h = canvas_size
    scale_ratio = random.uniform(scale_range[0], scale_range[1])
    target_dim = max(canvas_w, canvas_h) * scale_ratio
    
    car_w, car_h = car_img.size
    car_max_dim = max(car_w, car_h)
    resize_factor = target_dim / car_max_dim
    
    new_w = int(car_w * resize_factor)
    new_h = int(car_h * resize_factor)
    
    return car_img.resize((new_w, new_h), Image.LANCZOS)


def scale_frame_to_car(frame_img, car_size, ratio=1.2):
    """缩放车框"""
    car_w, car_h = car_size
    car_max_dim = max(car_w, car_h)
    target_frame_max = car_max_dim * ratio
    
    frame_w, frame_h = frame_img.size
    frame_max_dim = max(frame_w, frame_h)
    resize_factor = target_frame_max / frame_max_dim
    
    new_w = int(frame_w * resize_factor)
    new_h = int(frame_h * resize_factor)
    
    return frame_img.resize((new_w, new_h), Image.LANCZOS)


def random_rotate(img, min_angle=0, max_angle=360):
    """随机旋转"""
    angle = random.uniform(min_angle, max_angle)
    return img.rotate(angle, Image.BICUBIC, expand=True)


def safe_randint(a, b):
    if a > b:
        a, b = b, a
    if a == b:
        return a
    return random.randint(a, b)


def paste_on_background(background, foreground, position):
    temp = background.copy()
    temp.paste(foreground, position, foreground)
    return temp


def check_car_position(car_bbox, frame_inner, frame_outer):
    """判断车辆相对于边框的位置
    car_bbox: (x, y, x+w, y+h) 车辆在画布上的边界框
    frame_inner: (left, top, right, bottom) 内边框
    frame_outer: (left, top, right, bottom) 外边框
    
    返回: 'inside', 'outside', 'overlap'
    """
    cx1, cy1, cx2, cy2 = car_bbox
    fi1, fi2, fi3, fi4 = frame_inner
    fo1, fo2, fo3, fo4 = frame_outer
    
    # 检查是否完全在内边框内
    if cx1 >= fi1 and cy1 >= fi2 and cx2 <= fi3 and cy2 <= fi4:
        return 'inside'
    
    # 检查是否有任意一边超出外边框
    if cx1 < fo1 or cy1 < fo2 or cx2 > fo3 or cy2 > fo4:
        return 'outside'
    
    # 不超出外边框但超出内边框
    return 'overlap'


def generate_examples(background_path, car_path, output_dir, count_per_type=10, canvas_size=(400, 400)):
    """生成图片"""
    output_dir = Path(output_dir)
    
    inside_dir = output_dir / "inside"
    outside_dir = output_dir / "outside"
    overlap_dir = output_dir / "overlap"
    
    ensure_dir(inside_dir)
    ensure_dir(outside_dir)
    ensure_dir(overlap_dir)

    original_frame_bg = load_image(Path(background_path))
    original_car = load_image(Path(car_path))
    
    canvas_w, canvas_h = canvas_size
    
    print(f"画布: {canvas_w}x{canvas_h}")
    print(f"原始车框: {original_frame_bg.size}")
    print(f"原始车辆: {original_car.size}")
    print(f"车框=车辆x1.2, 车辆占画布60%-80%")
    print("-" * 50)
    
    inside_count = 0
    outside_count = 0
    overlap_count = 0

    for i in range(1, count_per_type + 1):
        
        for case_type in ['inside', 'outside', 'overlap']:
            success = False
            for attempt in range(200):  # 增加尝试次数
                try:
                    # 重新随机缩放和旋转
                    car_scaled = scale_car_to_canvas(original_car, canvas_size, (0.6, 0.8))
                    car_rotated = random_rotate(car_scaled, 0, 360)
                    car_w, car_h = car_rotated.size
                    
                    frame_bg_scaled = scale_frame_to_car(original_frame_bg, (car_w, car_h), ratio=1.2)
                    frame_w, frame_h = frame_bg_scaled.size
                    
                    frame_x = (canvas_w - frame_w) // 2
                    frame_y = (canvas_h - frame_h) // 2
                    
                    # 检测内边框和外边框（相对于车框，转为画布坐标）
                    inner_rel, outer_rel = detect_frame_bounds(frame_bg_scaled)
                    frame_inner = (
                        frame_x + inner_rel[0],
                        frame_y + inner_rel[1],
                        frame_x + inner_rel[2],
                        frame_y + inner_rel[3]
                    )
                    frame_outer = (
                        frame_x + outer_rel[0],
                        frame_y + outer_rel[1],
                        frame_x + outer_rel[2],
                        frame_y + outer_rel[3]
                    )
                    
                    # 根据类型生成位置
                    if case_type == 'inside':
                        pos = make_inside_pos(car_rotated, frame_inner, canvas_size)
                    elif case_type == 'outside':
                        pos = make_outside_pos(car_rotated, frame_outer, canvas_size)
                    else:  # overlap
                        pos = make_overlap_pos(car_rotated, frame_inner, frame_outer, canvas_size)
                    
                    if pos is None:
                        continue
                    
                    # 验证位置是否符合要求
                    car_bbox = (pos[0], pos[1], pos[0] + car_w, pos[1] + car_h)
                    actual_type = check_car_position(car_bbox, frame_inner, frame_outer)
                    
                    if actual_type != case_type:
                        continue  # 不符合要求，重新尝试
                    
                    # 创建画布
                    canvas = Image.new("RGBA", canvas_size, (255, 255, 255, 255))
                    canvas.paste(frame_bg_scaled, (frame_x, frame_y), frame_bg_scaled)
                    result = paste_on_background(canvas, car_rotated, pos)
                    
                    # 保存
                    target_dir = inside_dir if case_type == 'inside' else (outside_dir if case_type == 'outside' else overlap_dir)
                    result.save(target_dir / f"car_{case_type}_{i:03d}.png")
                    
                    # 彩色版本
                    colorful = create_colorful_background(canvas_size)
                    colorful.paste(frame_bg_scaled, (frame_x, frame_y), frame_bg_scaled)
                    paste_on_background(colorful, car_rotated, pos).save(target_dir / f"car_{case_type}_color_{i:03d}.png")
                    
                    if case_type == 'inside':
                        inside_count += 2
                    elif case_type == 'outside':
                        outside_count += 2
                    else:
                        overlap_count += 2
                    
                    success = True
                    break
                    
                except Exception as e:
                    continue
            
            if not success:
                print(f"  ✗ 第{i}组 {case_type} 失败")
        
        print(f"\r进度: {i}/{count_per_type} (内:{inside_count} 外:{outside_count} 叠:{overlap_count})", end="")
    
    print(f"\n\n完成! 共{inside_count+outside_count+overlap_count}张")
    print(f"inside: {inside_count} | outside: {outside_count} | overlap: {overlap_count}")


def make_inside_pos(car_img, frame_inner, canvas_size):
    """生成inside位置：车辆完全在内外边框内（即完全在内边框内）"""
    canvas_w, canvas_h = canvas_size
    car_w, car_h = car_img.size
    fl, ft, fr, fb = frame_inner
    
    margin = 5  # 增加边距确保不碰到内边框
    
    min_x = fl + margin
    max_x = fr - car_w - margin
    min_y = ft + margin
    max_y = fb - car_h - margin
    
    if max_x < min_x or max_y < min_y:
        # 如果内边框太小，尝试更小的边距
        min_x = fl
        max_x = fr - car_w
        min_y = ft
        max_y = fb - car_h
        
        if max_x < min_x or max_y < min_y:
            return None
    
    x = safe_randint(min_x, max_x)
    y = safe_randint(min_y, max_y)
    
    return (x, y)


def make_outside_pos(car_img, frame_outer, canvas_size):
    """生成outside位置：车辆任意一边超出外边框"""
    canvas_w, canvas_h = canvas_size
    car_w, car_h = car_img.size
    fl, ft, fr, fb = frame_outer
    
    positions = []
    
    # 左边超出：车辆的右边界在外边框左边界的右侧，但左边界超出
    if car_w < fr:  # 确保车辆能部分在框内
        x = safe_randint(fl - car_w + 10, fl - 1)  # 车辆左边界在外边框左边界的左边
        if x >= 0 and x + car_w > fl:  # 确保超出且不超出画布
            y = safe_randint(max(0, ft - car_h + 10), min(canvas_h - car_h, fb - 10))
            if 0 <= y <= canvas_h - car_h:
                positions.append((max(0, x), y))
    
    # 右边超出
    if car_w < canvas_w - fl:
        x = safe_randint(fr - car_w + 1, fr - 10)  # 车辆右边界在外边框右边界的右边
        if x + car_w <= canvas_w and x < fr:
            y = safe_randint(max(0, ft - car_h + 10), min(canvas_h - car_h, fb - 10))
            if 0 <= y <= canvas_h - car_h:
                positions.append((x, y))
    
    # 上边超出
    if car_h < fb:
        y = safe_randint(ft - car_h + 10, ft - 1)
        if y >= 0 and y + car_h > ft:
            x = safe_randint(max(0, fl - car_w + 10), min(canvas_w - car_w, fr - 10))
            if 0 <= x <= canvas_w - car_w:
                positions.append((x, max(0, y)))
    
    # 下边超出
    if car_h < canvas_h - ft:
        y = safe_randint(fb - car_h + 1, fb - 10)
        if y + car_h <= canvas_h and y < fb:
            x = safe_randint(max(0, fl - car_w + 10), min(canvas_w - car_w, fr - 10))
            if 0 <= x <= canvas_w - car_w:
                positions.append((x, y))
    
    # 角落超出（同时超出两边）
    # 左上角
    if fl - car_w + 10 >= 0 and ft - car_h + 10 >= 0:
        x = safe_randint(max(0, fl - car_w + 10), fl - 1)
        y = safe_randint(max(0, ft - car_h + 10), ft - 1)
        if x + car_w > fl and y + car_h > ft:
            positions.append((x, y))
    
    # 右上角
    if fr + 10 <= canvas_w and ft - car_h + 10 >= 0:
        x = safe_randint(fr - car_w + 1, min(fr - 10, canvas_w - car_w))
        y = safe_randint(max(0, ft - car_h + 10), ft - 1)
        if x < fr and y + car_h > ft:
            positions.append((x, y))
    
    # 左下角
    if fl - car_w + 10 >= 0 and fb + 10 <= canvas_h:
        x = safe_randint(max(0, fl - car_w + 10), fl - 1)
        y = safe_randint(fb - car_h + 1, min(fb - 10, canvas_h - car_h))
        if x + car_w > fl and y < fb:
            positions.append((x, y))
    
    # 右下角
    if fr + 10 <= canvas_w and fb + 10 <= canvas_h:
        x = safe_randint(fr - car_w + 1, min(fr - 10, canvas_w - car_w))
        y = safe_randint(fb - car_h + 1, min(fb - 10, canvas_h - car_h))
        if x < fr and y < fb:
            positions.append((x, y))
    
    if positions:
        return random.choice(positions)
    return None


def make_overlap_pos(car_img, frame_inner, frame_outer, canvas_size):
    """生成overlap位置：车辆不超出外边框但超出内边框"""
    canvas_w, canvas_h = canvas_size
    car_w, car_h = car_img.size
    fi_l, fi_t, fi_r, fi_b = frame_inner
    fo_l, fo_t, fo_r, fo_b = frame_outer
    
    positions = []
    
    # 策略：让车辆的某一边在内边框和外边框之间
    
    # 左边在内边框和外边框之间
    if fo_l < fi_l and car_w <= fo_r - fo_l:
        x = safe_randint(fo_l, fi_l - 1)
        if x >= 0:
            y = safe_randint(fo_t, fo_b - car_h)
            if fo_t <= y and y + car_h <= fo_b:
                # 验证：不超出外边框但超出内边框
                if x < fi_l and x >= fo_l:
                    positions.append((x, y))
    
    # 右边在内边框和外边框之间
    if fo_r > fi_r and car_w <= fo_r - fo_l:
        x = safe_randint(fi_r - car_w + 1, fo_r - car_w)
        if x + car_w <= fo_r and x >= fo_l:
            y = safe_randint(fo_t, fo_b - car_h)
            if fo_t <= y and y + car_h <= fo_b:
                if x + car_w > fi_r and x + car_w <= fo_r:
                    positions.append((x, y))
    
    # 上边在内边框和外边框之间
    if fo_t < fi_t and car_h <= fo_b - fo_t:
        y = safe_randint(fo_t, fi_t - 1)
        if y >= 0:
            x = safe_randint(fo_l, fo_r - car_w)
            if fo_l <= x and x + car_w <= fo_r:
                if y < fi_t and y >= fo_t:
                    positions.append((x, y))
    
    # 下边在内边框和外边框之间
    if fo_b > fi_b and car_h <= fo_b - fo_t:
        y = safe_randint(fi_b - car_h + 1, fo_b - car_h)
        if y + car_h <= fo_b and y >= fo_t:
            x = safe_randint(fo_l, fo_r - car_w)
            if fo_l <= x and x + car_w <= fo_r:
                if y + car_h > fi_b and y + car_h <= fo_b:
                    positions.append((x, y))
    
    # 如果以上方法都没找到合适位置，尝试随机放置并验证
    if not positions:
        for _ in range(50):
            x = safe_randint(fo_l, fo_r - car_w)
            y = safe_randint(fo_t, fo_b - car_h)
            
            if x >= 0 and y >= 0 and x + car_w <= canvas_w and y + car_h <= canvas_h:
                car_bbox = (x, y, x + car_w, y + car_h)
                if check_car_position(car_bbox, frame_inner, frame_outer) == 'overlap':
                    positions.append((x, y))
    
    if positions:
        return random.choice(positions)
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成车辆与车框位置示例")
    parser.add_argument("--background", default="background.png")
    parser.add_argument("--car", default="car.png")
    parser.add_argument("--output", default="output_images")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--canvas", type=int, default=400)

    args = parser.parse_args()
    
    try:
        generate_examples(
            background_path=args.background,
            car_path=args.car,
            output_dir=args.output,
            count_per_type=args.count,
            canvas_size=(args.canvas, args.canvas)
        )
    except FileNotFoundError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()