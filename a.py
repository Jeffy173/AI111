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
    """检测车框图片中黑色边框的边界"""
    img_array = np.array(frame_img)
    black_mask = (img_array[:, :, 0] < 50) & (img_array[:, :, 1] < 50) & (img_array[:, :, 2] < 50)
    black_coords = np.where(black_mask)
    
    if len(black_coords[0]) == 0:
        h, w = img_array.shape[:2]
        return 0, 0, w, h
    
    y_min, y_max = black_coords[0].min(), black_coords[0].max()
    x_min, x_max = black_coords[1].min(), black_coords[1].max()
    
    return x_min, y_min, x_max, y_max


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
                    
    else:
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
    """将车辆等比例缩放到占画布的指定比例范围"""
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
    """将车框缩放到车辆的指定倍数"""
    car_w, car_h = car_size
    car_max_dim = max(car_w, car_h)
    target_frame_max = car_max_dim * ratio
    
    frame_w, frame_h = frame_img.size
    frame_max_dim = max(frame_w, frame_h)
    resize_factor = target_frame_max / frame_max_dim
    
    new_w = int(frame_w * resize_factor)
    new_h = int(frame_h * resize_factor)
    
    return frame_img.resize((new_w, new_h), Image.LANCZOS)


def paste_on_background(background, foreground, position):
    """将前景图片粘贴到背景上"""
    temp = background.copy()
    temp.paste(foreground, position, foreground)
    return temp


def random_rotate(img, min_angle=0, max_angle=360):
    """随机旋转图片"""
    angle = random.uniform(min_angle, max_angle)
    return img.rotate(angle, Image.BICUBIC, expand=True)


def safe_randint(a, b):
    """安全的randint"""
    if a > b:
        a, b = b, a
    if a == b:
        return a
    return random.randint(a, b)


def generate_inside_position(car_img, frame_bounds, canvas_size):
    """生成车辆在框内的随机位置"""
    canvas_w, canvas_h = canvas_size
    car_w, car_h = car_img.size
    frame_left, frame_top, frame_right, frame_bottom = frame_bounds
    
    margin = 5
    max_x = frame_right - car_w - margin
    max_y = frame_bottom - car_h - margin
    
    if max_x >= frame_left + margin and max_y >= frame_top + margin:
        x = safe_randint(frame_left + margin, max_x)
        y = safe_randint(frame_top + margin, max_y)
    else:
        x = frame_left + (frame_right - frame_left - car_w) // 2
        y = frame_top + (frame_bottom - frame_top - car_h) // 2
    
    x = max(0, min(x, canvas_w - car_w))
    y = max(0, min(y, canvas_h - car_h))
    
    return x, y


def generate_outside_position(car_img, frame_bounds, canvas_size):
    """生成车辆在框外的随机位置"""
    canvas_w, canvas_h = canvas_size
    car_w, car_h = car_img.size
    frame_left, frame_top, frame_right, frame_bottom = frame_bounds
    
    positions = []
    margin = 10
    
    if frame_left - car_w - margin >= 0:
        x = safe_randint(0, frame_left - car_w - margin)
        y = safe_randint(0, max(0, canvas_h - car_h))
        positions.append((x, y))
    
    if frame_right + margin + car_w <= canvas_w:
        x = safe_randint(frame_right + margin, canvas_w - car_w)
        y = safe_randint(0, max(0, canvas_h - car_h))
        positions.append((x, y))
    
    if frame_top - car_h - margin >= 0:
        x = safe_randint(max(0, frame_left - car_w), min(canvas_w - car_w, frame_right))
        y = safe_randint(0, frame_top - car_h - margin)
        positions.append((x, y))
    
    if frame_bottom + margin + car_h <= canvas_h:
        x = safe_randint(max(0, frame_left - car_w), min(canvas_w - car_w, frame_right))
        y = safe_randint(frame_bottom + margin, canvas_h - car_h)
        positions.append((x, y))
    
    if positions:
        return random.choice(positions)
    
    return 0, 0


def generate_on_edge_position(car_img, frame_bounds, canvas_size, tolerance=2):
    """生成车辆压边的随机位置"""
    canvas_w, canvas_h = canvas_size
    car_w, car_h = car_img.size
    frame_left, frame_top, frame_right, frame_bottom = frame_bounds
    
    edge = random.choice(['left', 'right', 'top', 'bottom'])
    
    if edge == 'left':
        x = frame_left
        min_y = frame_top
        max_y = frame_bottom - car_h
        if max_y < min_y:
            max_y = min_y
        y = safe_randint(min_y, max_y)
        
    elif edge == 'right':
        x = frame_right - car_w
        min_y = frame_top
        max_y = frame_bottom - car_h
        if max_y < min_y:
            max_y = min_y
        y = safe_randint(min_y, max_y)
        
    elif edge == 'top':
        y = frame_top
        min_x = frame_left
        max_x = frame_right - car_w
        if max_x < min_x:
            max_x = min_x
        x = safe_randint(min_x, max_x)
        
    else:
        y = frame_bottom - car_h
        min_x = frame_left
        max_x = frame_right - car_w
        if max_x < min_x:
            max_x = min_x
        x = safe_randint(min_x, max_x)
    
    x += random.randint(-tolerance, tolerance)
    y += random.randint(-tolerance, tolerance)
    
    x = max(0, min(x, canvas_w - car_w))
    y = max(0, min(y, canvas_h - car_h))
    
    return x, y


def generate_examples(background_path, car_path, output_dir, count_per_type=10, canvas_size=(400, 400)):
    """生成车辆位置示例图片"""
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
    
    print(f"画布尺寸: {canvas_w}x{canvas_h}")
    print(f"背景底图填充整个画布，车框=车辆x1.2，车辆占画布60%%-80%%")
    
    inside_count = 0
    outside_count = 0
    overlap_count = 0

    for i in range(1, count_per_type + 1):
        try:
            # 1. 缩放车辆到占画布60%-80%
            car_scaled = scale_car_to_canvas(original_car, canvas_size, (0.6, 0.8))
            
            # 2. 旋转车辆
            car_rotated = random_rotate(car_scaled, 0, 360)
            car_w, car_h = car_rotated.size
            
            # 3. 缩放车框为车辆的1.2倍
            frame_bg_scaled = scale_frame_to_car(original_frame_bg, (car_w, car_h), ratio=1.2)
            frame_w, frame_h = frame_bg_scaled.size
            
            # 车框居中放置
            frame_x = (canvas_w - frame_w) // 2
            frame_y = (canvas_h - frame_h) // 2
            
            # 4. 检测黑色边框的边界
            frame_inner_bounds_rel = detect_frame_bounds(frame_bg_scaled)
            
            # 转换为画布绝对坐标
            frame_inner_bounds = (
                frame_x + frame_inner_bounds_rel[0],
                frame_y + frame_inner_bounds_rel[1],
                frame_x + frame_inner_bounds_rel[2],
                frame_y + frame_inner_bounds_rel[3]
            )
            
            print(f"\r第{i}/{count_per_type} 车辆:{car_w}x{car_h} 车框:{frame_w}x{frame_h}", end="")
            
            # 5. 生成三种位置
            inside_pos = generate_inside_position(car_rotated, frame_inner_bounds, canvas_size)
            outside_pos = generate_outside_position(car_rotated, frame_inner_bounds, canvas_size)
            on_edge_pos = generate_on_edge_position(car_rotated, frame_inner_bounds, canvas_size)
            
            # === 白色底图版本 ===
            # 1. 先创建白色底图（填满整个画布）
            white_canvas = Image.new("RGBA", canvas_size, (255, 255, 255, 255))
            # 2. 在白色底图上放车框（居中）
            white_canvas.paste(frame_bg_scaled, (frame_x, frame_y), frame_bg_scaled)
            # 3. 放车辆
            inside_img = paste_on_background(white_canvas, car_rotated, inside_pos)
            outside_img = paste_on_background(white_canvas, car_rotated, outside_pos)
            on_edge_img = paste_on_background(white_canvas, car_rotated, on_edge_pos)
            
            inside_img.save(inside_dir / f"car_inside_{i:03d}.png")
            outside_img.save(outside_dir / f"car_outside_{i:03d}.png")
            on_edge_img.save(overlap_dir / f"car_overlap_{i:03d}.png")
            
            inside_count += 1
            outside_count += 1
            overlap_count += 1
            
            # === 彩色底图版本 ===
            colorful_canvas = create_colorful_background(canvas_size)
            colorful_canvas.paste(frame_bg_scaled, (frame_x, frame_y), frame_bg_scaled)
            
            paste_on_background(colorful_canvas, car_rotated, inside_pos).save(inside_dir / f"car_inside_color_{i:03d}.png")
            paste_on_background(colorful_canvas, car_rotated, outside_pos).save(outside_dir / f"car_outside_color_{i:03d}.png")
            paste_on_background(colorful_canvas, car_rotated, on_edge_pos).save(overlap_dir / f"car_overlap_color_{i:03d}.png")
            
            inside_count += 1
            outside_count += 1
            overlap_count += 1
                
        except Exception as e:
            print(f"\n警告: 第{i}组失败: {e}")
            continue

    print(f"\n\n完成! inside:{inside_count} outside:{outside_count} overlap:{overlap_count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成车辆与车框位置关系的示例图片")
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