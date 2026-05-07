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
    """检测车框图片中黑色边框的内边界"""
    img_array = np.array(frame_img)
    # 检测深色像素（放宽阈值）
    dark_mask = (img_array[:, :, 0] < 80) & (img_array[:, :, 1] < 80) & (img_array[:, :, 2] < 80)
    dark_coords = np.where(dark_mask)
    
    if len(dark_coords[0]) == 0:
        h, w = img_array.shape[:2]
        # 默认框内区域为图片的30%-70%
        return int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)
    
    y_min, y_max = dark_coords[0].min(), dark_coords[0].max()
    x_min, x_max = dark_coords[1].min(), dark_coords[1].max()
    
    return x_min, y_min, x_max, y_max


def create_colorful_background(size):
    """创建彩色背景"""
    colors = [
        (255, 200, 200), (200, 255, 200), (200, 200, 255),
        (255, 255, 200), (255, 200, 255), (200, 255, 255),
        (255, 220, 180), (180, 255, 220), (220, 180, 255),
    ]
    color = random.choice(colors)
    return Image.new("RGBA", size, color + (255,))


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
        
        # 每个case独立随机
        for case_type in ['inside', 'outside', 'overlap']:
            success = False
            for attempt in range(100):
                try:
                    # 重新随机缩放和旋转（每个尝试都用新的）
                    car_scaled = scale_car_to_canvas(original_car, canvas_size, (0.6, 0.8))
                    car_rotated = random_rotate(car_scaled, 0, 360)
                    car_w, car_h = car_rotated.size
                    
                    frame_bg_scaled = scale_frame_to_car(original_frame_bg, (car_w, car_h), ratio=1.2)
                    frame_w, frame_h = frame_bg_scaled.size
                    
                    frame_x = (canvas_w - frame_w) // 2
                    frame_y = (canvas_h - frame_h) // 2
                    
                    # 检测黑色边框内边界（相对于车框，转为画布坐标）
                    inner_rel = detect_frame_bounds(frame_bg_scaled)
                    inner = (
                        frame_x + inner_rel[0],
                        frame_y + inner_rel[1],
                        frame_x + inner_rel[2],
                        frame_y + inner_rel[3]
                    )
                    
                    # 根据类型生成位置
                    if case_type == 'inside':
                        pos = make_inside_pos(car_rotated, inner, canvas_size)
                    elif case_type == 'outside':
                        pos = make_outside_pos(car_rotated, inner, canvas_size)
                    else:
                        pos = make_overlap_pos(car_rotated, inner, canvas_size)
                    
                    if pos is None:
                        continue
                    
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
    """生成inside位置：车辆完全在框内，不碰边框"""
    canvas_w, canvas_h = canvas_size
    car_w, car_h = car_img.size
    fl, ft, fr, fb = frame_inner
    
    margin = 3
    
    min_x = fl + margin
    max_x = fr - car_w - margin
    min_y = ft + margin
    max_y = fb - car_h - margin
    
    if max_x < min_x or max_y < min_y:
        return None
    
    x = safe_randint(min_x, max_x)
    y = safe_randint(min_y, max_y)
    
    return (x, y)


def make_outside_pos(car_img, frame_inner, canvas_size):
    """生成outside位置：车辆超出框的范围但不超过画布"""
    canvas_w, canvas_h = canvas_size
    car_w, car_h = car_img.size
    fl, ft, fr, fb = frame_inner
    
    positions = []
    margin = 5
    
    # 左
    if fl - car_w - margin >= 0:
        x = safe_randint(0, fl - car_w - margin)
        y = safe_randint(0, canvas_h - car_h)
        positions.append((x, y))
    # 右
    if fr + margin + car_w <= canvas_w:
        x = safe_randint(fr + margin, canvas_w - car_w)
        y = safe_randint(0, canvas_h - car_h)
        positions.append((x, y))
    # 上
    if ft - car_h - margin >= 0:
        x = safe_randint(0, canvas_w - car_w)
        y = safe_randint(0, ft - car_h - margin)
        positions.append((x, y))
    # 下
    if fb + margin + car_h <= canvas_h:
        x = safe_randint(0, canvas_w - car_w)
        y = safe_randint(fb + margin, canvas_h - car_h)
        positions.append((x, y))
    
    if positions:
        return random.choice(positions)
    return None


def make_overlap_pos(car_img, frame_inner, canvas_size):
    """生成overlap位置：车辆边与边框重合，不超出"""
    canvas_w, canvas_h = canvas_size
    car_w, car_h = car_img.size
    fl, ft, fr, fb = frame_inner
    
    edge = random.choice(['left', 'right', 'top', 'bottom'])
    
    if edge == 'left':
        x = fl
        min_y = ft
        max_y = fb - car_h
    elif edge == 'right':
        x = fr - car_w
        min_y = ft
        max_y = fb - car_h
    elif edge == 'top':
        y = ft
        min_x = fl
        max_x = fr - car_w
    else:  # bottom
        y = fb - car_h
        min_x = fl
        max_x = fr - car_w
    
    if edge in ['left', 'right']:
        if max_y < min_y:
            return None
        y = safe_randint(min_y, max_y)
    else:
        if max_x < min_x:
            return None
        x = safe_randint(min_x, max_x)
    
    x = max(0, min(x, canvas_w - car_w))
    y = max(0, min(y, canvas_h - car_h))
    
    return (x, y)


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