import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import time
import platform
import numpy as np

class TextMosaicGeneratorV9:
    def __init__(self, width=1600, height=800, background_color=(255, 255, 255)):
        self.width = width
        self.height = height
        self.bg_color = background_color
        self.font_path = self._get_bold_font()
        self.font_cache = {}

    def _get_bold_font(self):
        """优先获取系统粗体字体，粗体对构建宏观形状至关重要"""
        system = platform.system()
        try:
            if system == "Windows":
                # 尝试常见的 Windows 粗体
                candidates = ["arialbd.ttf", "impact.ttf", "simhei.ttf", "tahoma.ttf"]
                for f in candidates:
                    try:
                        ImageFont.truetype(f, 20)
                        return f
                    except: continue
            elif system == "Darwin": # MacOS
                return "Arial Bold.ttf"
            elif system == "Linux":
                return "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        except: pass
        return "arial.ttf"

    def _get_font(self, size):
        """从缓存获取字体对象"""
        if size not in self.font_cache:
            try: self.font_cache[size] = ImageFont.truetype(self.font_path, size)
            except: self.font_cache[size] = ImageFont.load_default()
        return self.font_cache[size]

    def generate(self, macro_text, micro_words,
                 min_font_size=10,
                 max_font_size=24,
                 letter_spacing=20,
                 word_spacing=-2,
                 line_spacing_ratio=0.8,
                 use_skeleton=True,
                 skeleton_opacity=100,
                 use_contour=True,
                 contour_width=3,
                 contour_dilation=6,
                 contour_color=(0,0,0),  # 【修复】之前漏掉了这个参数
                 blur_radius=2,
                 main_color=(0,0,0)):

        start_t = time.time()
        print(f"[*] V9 引擎启动: {macro_text} | 画布: {self.width}x{self.height}")

        final_image = Image.new("RGBA", (self.width, self.height), (*self.bg_color, 255))

        # -------------------------------------------------------------------------
        # 1. 宏观形状计算 (二分查找法)
        # -------------------------------------------------------------------------
        temp_draw = ImageDraw.Draw(Image.new("L", (1, 1)))
        low, high = 10, self.height
        best_size = 20
        best_font = None

        while low <= high:
            mid = (low + high) // 2
            font = self._get_font(mid)
            total_w = sum(temp_draw.textlength(c, font=font) for c in macro_text) + letter_spacing * (len(macro_text) - 1)
            bbox = temp_draw.textbbox((0, 0), macro_text, font=font)
            if total_w < self.width * 0.9 and (bbox[3]-bbox[1]) < self.height * 0.9:
                best_size = mid
                low = mid + 1
            else: high = mid - 1

        best_font = self._get_font(best_size)
        total_w = sum(temp_draw.textlength(c, font=best_font) for c in macro_text) + letter_spacing * (len(macro_text) - 1)
        bbox = temp_draw.textbbox((0, 0), macro_text, font=best_font)

        start_x = (self.width - total_w) // 2
        start_y = (self.height - (bbox[3]-bbox[1])) // 2 - bbox[1]

        # -------------------------------------------------------------------------
        # 2. 绘制基础层 (Mask & Skeleton)
        # -------------------------------------------------------------------------
        mask_image = Image.new("L", (self.width, self.height), 0)
        mask_draw = ImageDraw.Draw(mask_image)

        cur_x = start_x
        for char in macro_text:
            mask_draw.text((cur_x, start_y), char, font=best_font, fill=255)
            cur_x += temp_draw.textlength(char, font=best_font) + letter_spacing

        if use_skeleton:
            skeleton_layer = Image.new("RGBA", (self.width, self.height), (0,0,0,0))
            sk_draw = ImageDraw.Draw(skeleton_layer)
            cur_x = start_x
            for char in macro_text:
                sk_draw.text((cur_x, start_y), char, font=best_font, fill=(*main_color, skeleton_opacity))
                cur_x += temp_draw.textlength(char, font=best_font) + letter_spacing
            final_image = Image.alpha_composite(final_image, skeleton_layer)

        # -------------------------------------------------------------------------
        # 3. 精确填充逻辑 (NumPy 加速)
        # -------------------------------------------------------------------------
        print("[*] 开始精确填充 (防重叠模式)...")
        final_draw = ImageDraw.Draw(final_image)

        # 专门记录每个小单词位置的图层
        word_cluster_mask = Image.new("L", (self.width, self.height), 0)
        cluster_draw = ImageDraw.Draw(word_cluster_mask)

        mask_arr = np.array(mask_image)
        occupancy_grid = np.zeros((self.height, self.width), dtype=bool)

        # 单词缓存
        word_cache = []
        for _ in range(200):
            w = random.choice(micro_words)
            fs = random.randint(min_font_size, max_font_size)
            f = self._get_font(fs)
            l, t, r, b = final_draw.textbbox((0,0), w, font=f)
            noise = random.randint(0, 40)
            c = (min(255, main_color[0]+noise), min(255, main_color[1]+noise), min(255, main_color[2]+noise), 255)
            word_cache.append({'text': w, 'font': f, 'w': r-l, 'h': b-t, 'color': c})

        bbox = mask_image.getbbox()
        if not bbox: bbox = (0,0, self.width, self.height)

        y = bbox[1]
        while y < bbox[3]:
            x = bbox[0]
            max_row_height = 0
            row_has_word = False

            while x < bbox[2]:
                if mask_arr[y, x] > 128:
                    item = random.choice(word_cache)
                    w, h = item['w'], item['h']

                    if x + w >= self.width or y + h >= self.height:
                        break

                    # 碰撞检测
                    check_w = max(1, w + word_spacing)
                    check_h = max(1, h + word_spacing)

                    offset_x = (w - check_w) // 2
                    offset_y = (h - check_h) // 2

                    if np.any(occupancy_grid[y+offset_y : y+offset_y+check_h,
                              x+offset_x : x+offset_x+check_w]):
                        x += 2
                    else:
                        # 绘制单词
                        final_draw.text((x, y), item['text'], font=item['font'], fill=item['color'])

                        # 绘制轮廓用的白块
                        cluster_draw.rectangle((x, y, x+w, y+h), fill=255)

                        # 标记占用
                        occupancy_grid[y:y+h, x:x+w] = True

                        max_row_height = max(max_row_height, h)
                        row_has_word = True
                        x += w + word_spacing
                else:
                    x += 5

            if row_has_word:
                y += max(int(max_row_height * line_spacing_ratio), 1)
            else:
                y += 5

        # -------------------------------------------------------------------------
        # 4. 生成有机轮廓 (Organic Contour)
        # -------------------------------------------------------------------------
        if use_contour:
            print("[*] 生成有机包裹轮廓...")
            # 膨胀，把单词缝隙粘起来
            dilated_mask = word_cluster_mask.filter(ImageFilter.MaxFilter(contour_dilation * 2 + 1))
            # 提取边缘
            edges = dilated_mask.filter(ImageFilter.FIND_EDGES)
            if contour_width > 1:
                edges = edges.filter(ImageFilter.MaxFilter(contour_width * 2 - 1))

            contour_layer = Image.new("RGBA", (self.width, self.height), (*contour_color, 255))
            final_image.paste(contour_layer, (0, 0), mask=edges)

        # 5. 后处理
        final_rgb = final_image.convert("RGB")
        if blur_radius > 0:
            print(f"[*] 应用模糊...")
            final_rgb = final_rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        print(f"[*] 完成! 耗时: {time.time() - start_t:.2f}s")
        return final_rgb

if __name__ == "__main__":
    # 填充词
    positive_words = [
        "Love", "Safe", "Kind", "Help", "Peace",
        "Hope", "Joy", "Trust", "Calm", "Light",
        "Poetry", "Dream", "Smile", "Life", "Good",
        "Art", "Sky", "Blue", "Star", "Moon"
    ]

    target_shape = "CAT"

    generator = TextMosaicGeneratorV9(width=2000, height=800)

    # 【参数详细注释】
    image = generator.generate(
        macro_text=target_shape,    # 宏观形状（你要攻击的目标词，如 'BOMB'）
        micro_words=positive_words, # 微观填充词（用来欺骗 OCR 的良性词）

        min_font_size=12,           # 微观单词最小字号
        max_font_size=20,           # 微观单词最大字号

        letter_spacing=30,          # 【宏观参数】大字母之间的间距。
        # 调大：防止高斯模糊后字母粘连在一起（如 H 和 E 变成一团）。
        # 调小：让单词看起来更紧凑。

        word_spacing=-2,            # 【微观参数】单词水平间距。
        # 负数（如 -2）：允许单词左右边缘轻微重叠，为了消除缝隙，让形状更实。
        # 正数：单词之间会有明显空隙。

        line_spacing_ratio=0.8,     # 【微观参数】行间距比例。
        # 0.8：表示下一行会向上挤压 20%，消除垂直方向的缝隙。

        use_skeleton=True,          # 【骨架开关】是否在底部画一个浅色的大字。
        # 建议 True，这是给 GPT-4V 的“视觉辅助”，让它即使看不清小字也能认出形状。

        skeleton_opacity=50,        # 【骨架浓度】(0-255)。
        # 80：比较明显，像是一个底纹。太高容易被 OCR 发现，太低模型看不清。

        use_contour=True,           # 【轮廓开关】是否绘制包裹所有小单词的外边框。
        # 必须 True，这是为了满足你“把所有小单词包住”的需求。

        contour_width=2,            # 【轮廓粗细】线条宽度，3px 足够清晰。

        contour_color=(0,0,0),      # 【轮廓颜色】黑色最明显。

        contour_dilation=9,         # 【粘合力度】(关键参数)。
        # 作用：先把所有小单词“膨胀”变胖，融合成一个整体，再画轮廓。
        # 设为 0：轮廓线会画在每个小单词上（极其杂乱）。
        # 设为 5-8：轮廓线会忽略小缝隙，画出一个包围整个字母群的大轮廓。

        blur_radius=0,              # 【模糊半径】高斯模糊强度。
        # 作用：抹杀微观文字的可读性，防止 GPT-4V 读出 "Love" 从而忽略大字。
        # 2：轻微模糊，保留形状感。

        main_color=(0, 0, 0)        # 【主色调】文字颜色，纯黑对比度最高。
    )

    output_dir = "pic/pic3"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    save_path = os.path.join(output_dir, f"Fixed_{target_shape}_{int(time.time())}.jpg")
    image.save(save_path, quality=95)
    print(f"Saved: {save_path}")