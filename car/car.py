import random
import os
from PIL import Image

# ========== 最终命令配置 ==========
PARKING_IMG = "parking.png"
CAR_IMG = "car.png"
OUTPUT_DIR = "output_violate_safe"
A4_SIZE = (2480, 3508)

PARKING_SCALE = 0.50
CAR_SCALE = 0.60

# Proper 的安全距离（车离黑框）
PROPER_SAFE_BORDER = 60

# Violate 的安全距离（车离A4纸边框）
VIOLATE_SAFE_BORDER = 100

CATEGORIES = {
    "Proper": 3000,
    "Improper": 6000,
    "Violate": 3000
}
# =================================

def resize_parking(parking_img, scale):
    new_w = int(A4_SIZE[0] * scale)
    new_h = int(parking_img.height * (new_w / parking_img.width))
    return parking_img.resize((new_w, new_h), Image.LANCZOS)

def resize_car(car_img, parking_img, car_scale):
    parking_w = parking_img.width
    car_w = int(parking_w * car_scale)
    car_h = int(car_img.height * (car_w / car_img.width))
    return car_img.resize((car_w, car_h), Image.LANCZOS)

def get_car_pixels(car_img):
    """获取车的所有不透明像素"""
    if car_img.mode != 'RGBA':
        car_img = car_img.convert('RGBA')
    pixels = []
    w, h = car_img.size
    for y in range(h):
        for x in range(w):
            r, g, b, a = car_img.getpixel((x, y))
            if a > 10:
                pixels.append((x, y))
    return pixels

def is_completely_inside_safe_zone(car_pixels, car_x, car_y, safe_left, safe_right, safe_top, safe_bottom):
    """检查车是否完全在安全区域内"""
    for dx, dy in car_pixels:
        abs_x = car_x + dx
        abs_y = car_y + dy
        if not (safe_left <= abs_x <= safe_right and safe_top <= abs_y <= safe_bottom):
            return False
    return True

def is_touching_border(car_pixels, car_x, car_y, lot_left, lot_right, lot_top, lot_bottom):
    """检查车是否碰到黑框（从内或从外）"""
    for dx, dy in car_pixels:
        abs_x = car_x + dx
        abs_y = car_y + dy
        # 检查是否碰到车位边框（允许2像素误差）
        if abs(abs_x - lot_left) <= 2 or abs(abs_x - lot_right) <= 2 or \
           abs(abs_y - lot_top) <= 2 or abs(abs_y - lot_bottom) <= 2:
            return True
    return False

def is_completely_outside(car_pixels, car_x, car_y, lot_left, lot_right, lot_top, lot_bottom):
    pass