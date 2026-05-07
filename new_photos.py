import os
import sys
import random
from datetime import datetime
from PIL import Image, ImageEnhance
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

def remove_bg_color(img, bg_color=(255, 255, 255), tolerance=20):
    img = img.convert('RGBA')
    data = img.getdata()
    if isinstance(bg_color, str):
        temp = Image.new('RGB', (1, 1), bg_color)
        bg_color = temp.getpixel((0, 0))
    new_data = []
    for pixel in data:
        if abs(pixel[0] - bg_color[0]) <= tolerance and \
           abs(pixel[1] - bg_color[1]) <= tolerance and \
           abs(pixel[2] - bg_color[2]) <= tolerance:
            new_data.append((pixel[0], pixel[1], pixel[2], 0))
        else:
            new_data.append(pixel)
    img.putdata(new_data)
    return img

def adjust_brightness(img, factor):
    return ImageEnhance.Brightness(img).enhance(factor)

def perspective_3d(img, angle_x, angle_y):
    """
    3D透视变换，模拟纸倾斜的效果
    结果图像会自动扩展以容纳全部内容
    """
    w, h = img.size
    
    # 角度转弧度
    ax = angle_x * np.pi / 180
    ay = angle_y * np.pi / 180
    
    # 透视投影：远处的边缩小
    # 假设摄像机距离 = 2倍图像高度
    camera_dist = h * 2
    
    # 四个角点（3D坐标，z=0平面）
    corners_3d = [
        (-w/2, -h/2, 0),  # 左上
        ( w/2, -h/2, 0),  # 右上
        ( w/2,  h/2, 0),  # 右下
        (-w/2,  h/2, 0),  # 左下
    ]
    
    # 先绕X轴旋转，再绕Y轴旋转
    def rotate_x(p, angle):
        x, y, z = p
        new_y = y * np.cos(angle) - z * np.sin(angle)
        new_z = y * np.sin(angle) + z * np.cos(angle)
        return (x, new_y, new_z)
    
    def rotate_y(p, angle):
        x, y, z = p
        new_x = x * np.cos(angle) + z * np.sin(angle)
        new_z = -x * np.sin(angle) + z * np.cos(angle)
        return (new_x, y, new_z)
    
    # 应用旋转
    rotated = []
    for p in corners_3d:
        p = rotate_x(p, ax)
        p = rotate_y(p, ay)
        rotated.append(p)
    
    # 投影到平面（z=distance 的平面，透视投影）
    projected = []
    for p in rotated:
        x, y, z = p
        # 投影后的坐标
        scale = camera_dist / (camera_dist + z)
        px = x * scale
        py = y * scale
        projected.append((px, py))
    
    # 转换为图像坐标（原点在左上角）
    src_pts = [(0, 0), (w, 0), (w, h), (0, h)]
    dst_pts = []
    
    for px, py in projected:
        # 偏移使中心对齐
        dst_x = px + w/2
        dst_y = py + h/2
        dst_pts.append((dst_x, dst_y))
    
    # 计算边界并偏移到正坐标
    xs = [p[0] for p in dst_pts]
    ys = [p[1] for p in dst_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    dst_pts = [(x - min_x, y - min_y) for x, y in dst_pts]
    out_w = int(max_x - min_x)
    out_h = int(max_y - min_y)
    
    # 计算透视变换系数
    def find_coeffs(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
        A = np.matrix(matrix, dtype=np.float32)
        B = np.array(pb).reshape(8)
        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)
    
    coeffs = find_coeffs(src_pts, dst_pts)
    result = img.transform((out_w, out_h), Image.Transform.PERSPECTIVE, coeffs,
                          resample=Image.Resampling.BICUBIC, fillcolor=(255,255,255))
    return result

def transform_foreground(bg, fg, center, angle):
    """合成前景到背景"""
    bg = bg.convert('RGBA')
    fg = fg.convert('RGBA')
    fg = fg.rotate(angle, expand=True, fillcolor=(0,0,0,0))
    x = int(center[0] - fg.width/2)
    y = int(center[1] - fg.height/2)
    overlay = Image.new('RGBA', bg.size, (0,0,0,0))
    overlay.paste(fg, (x, y), fg)
    return Image.alpha_composite(bg, overlay).convert('RGB')

def find_parking_boundaries(parking_img):
    """检测车位黑边位置"""
    gray = parking_img.convert('L')
    binary = gray.point(lambda x: 0 if x < 128 else 255)
    arr = np.array(binary)
    black = np.where(arr == 0)
    if len(black[0]) == 0:
        return None
    return {
        'top': int(np.min(black[0])),
        'bottom': int(np.max(black[0])),
        'left': int(np.min(black[1])),
        'right': int(np.max(black[1])),
    }

def resize_224(img):
    return img.resize((224, 224), Image.Resampling.LANCZOS)

def generate_test():
    """生成测试图片"""
    print("生成测试图片...")
    
    # 加载图片
    car_raw = Image.open('./car/car.png')
    car = remove_bg_color(car_raw, (255,255,255), 20)
    parking = Image.open('./car/background.png')
    
    # 获取车位边界
    bounds = find_parking_boundaries(parking)
    if bounds:
        print(f"车位边界: 左={bounds['left']}, 右={bounds['right']}, 上={bounds['top']}, 下={bounds['bottom']}")
    
    # 白色背景 (2480x2480)
    white_bg = Image.new('RGB', (2480, 2480), (255,255,255))
    # 车位居中
    px = (white_bg.width - parking.width) // 2
    py = (white_bg.height - parking.height) // 2
    white_bg.paste(parking, (px, py))
    
    # A4背景 (2480x3508)
    a4_bg = Image.new('RGB', (2480, 3508), (255,255,255))
    a4_bg.paste(parking, (0, 0))
    
    # 计算有效的停车区域（车的中心点范围）
    # 内边界
    inner_l = bounds['left'] + 20  # 假设黑边厚度20
    inner_r = bounds['right'] - 20
    inner_t = bounds['top'] + 20
    inner_b = bounds['bottom'] - 20
    # 外边界
    outer_l = bounds['left']
    outer_r = bounds['right']
    outer_t = bounds['top']
    outer_b = bounds['bottom']
    
    # 简单的位置：规范停车放在内边界中心
    car_w, car_h = car.size
    proper_x = (inner_l + inner_r) // 2
    proper_y = (inner_t + inner_b) // 2
    
    os.makedirs('./test_output', exist_ok=True)
    
    # 测试不同角度
    for status in ['proper', 'improper', 'violate']:
        # 白色背景 + 3D透视
        img = transform_foreground(white_bg.copy(), car, (proper_x, proper_y), 0)
        img = perspective_3d(img, 10, 10)  # 10度透视
        img = adjust_brightness(img, random.uniform(0.6, 1.2))
        resize_224(img).save(f'./test_output/white_{status}.png')
        
        # 彩色背景：A4先合成，然后透视，最后贴到彩色背景
        a4_img = transform_foreground(a4_bg.copy(), car, (proper_x, proper_y), 0)
        a4_persp = perspective_3d(a4_img, 10, 10)
        # 随机彩色背景
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        color_bg = Image.new('RGB', (2480, 2480), color)
        # 居中粘贴透视后的A4
        x = (color_bg.width - a4_persp.width) // 2
        y = (color_bg.height - a4_persp.height) // 2
        color_bg.paste(a4_persp, (x, y))
        color_bg = adjust_brightness(color_bg, random.uniform(0.6, 1.2))
        resize_224(color_bg).save(f'./test_output/color_{status}.png')
        
        # 保存A4原图供对比
        resize_224(a4_img).save(f'./test_output/a4_original_{status}.png')
    
    print("测试图片已保存到 ./test_output/")
    print("请检查：")
    print("  - white_*.png: 白色背景+透视，应该看到倾斜效果")
    print("  - color_*.png: 彩色背景，A4完整显示在中间，周围是彩色")
    print("  - a4_original_*.png: 透视前的A4原图")

if __name__ == "__main__":
    generate_test()