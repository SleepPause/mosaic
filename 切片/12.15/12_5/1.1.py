import random
from PIL import Image, ImageDraw, ImageFont
import os
import time

class TextMosaicGenerator:
    def __init__(self, width=1200, height=800, background_color=(255, 255, 255), font_path="arial.ttf"):
        self.width = width
        self.height = height
        self.bg_color = background_color
        self.font_path = font_path

    def generate(self, macro_text, micro_words, density=5, min_font_size=10, max_font_size=15, word_spacing=0, overlap_ratio=0.25, letter_spacing=0):
        """
        生成文字马赛克图像
        :param macro_text: 宏观形状文字
        :param micro_words: 填充单词列表
        :param density: 扫描精度(像素)
        :param min_font_size: 最小字号
        :param max_font_size: 最大字号
        :param word_spacing: 微观单词之间的额外水平间距 (像素)
        :param overlap_ratio: 微观单词允许重叠的比例
        :param letter_spacing: 宏观字母之间的间距 (像素)，控制如 'H' 和 'E' 之间的距离
        """

        # 1. 初始化画布
        final_image = Image.new("RGB", (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(final_image)

        # 2. 生成宏观形状蒙版
        mask_image = Image.new("L", (self.width, self.height), 255)
        mask_draw = ImageDraw.Draw(mask_image)

        # 【修改点1】初始字号设小一点，方便从头开始放大
        macro_font_size = 20
        try:
            macro_font = ImageFont.truetype(self.font_path, macro_font_size)
        except IOError:
            macro_font = ImageFont.load_default()

        # 【修改点2】最大化填充逻辑
        step = 5 # 每次尝试增加的字号大小
        while True:
            # 计算包含间距的总宽度
            total_width = 0
            for char in macro_text:
                total_width += mask_draw.textlength(char, font=macro_font)

            if len(macro_text) > 1:
                total_width += letter_spacing * (len(macro_text) - 1)

            bbox = mask_draw.textbbox((0, 0), macro_text, font=macro_font)
            text_height = bbox[3] - bbox[1]

            # 【修改点3】将阈值从 0.8 提升到 0.96，让文字尽量撑满画布
            if total_width < self.width * 0.96 and text_height < self.height * 0.96:
                macro_font_size += step
                try:
                    macro_font = ImageFont.truetype(self.font_path, macro_font_size)
                except:
                    break
            else:
                # 【修改点4】一旦超出，回退一步，确保不会被截断
                macro_font_size -= step
                try:
                    macro_font = ImageFont.truetype(self.font_path, macro_font_size)
                except:
                    pass

                # 重新计算最终的宽高，用于后续的居中定位
                total_width = 0
                for char in macro_text:
                    total_width += mask_draw.textlength(char, font=macro_font)
                if len(macro_text) > 1:
                    total_width += letter_spacing * (len(macro_text) - 1)
                bbox = mask_draw.textbbox((0, 0), macro_text, font=macro_font)
                text_height = bbox[3] - bbox[1]
                break

        # 计算起始位置
        text_x = (self.width - total_width) // 2
        text_y = (self.height - text_height) // 2

        # 逐字绘制宏观文字以应用间距
        current_x = text_x
        for char in macro_text:
            mask_draw.text((current_x, text_y), char, font=macro_font, fill=0)
            current_x += mask_draw.textlength(char, font=macro_font) + letter_spacing

        # 3. 填充微观文字 (使用宽松的碰撞检测)
        occupancy_mask = Image.new("L", (self.width, self.height), 0)
        occupancy_draw = ImageDraw.Draw(occupancy_mask)

        y = 0
        while y < self.height:
            x = 0
            max_row_height = 0
            row_has_word = False

            while x < self.width:
                pixel_value = mask_image.getpixel((x, y))

                # 在宏观形状内
                if pixel_value < 128:
                    word = random.choice(micro_words)

                    # 随机字号
                    current_font_size = random.randint(min_font_size, max_font_size)
                    try:
                        micro_font = ImageFont.truetype(self.font_path, current_font_size)
                    except:
                        micro_font = ImageFont.load_default()

                    gray_val = random.randint(0, 100)
                    text_color = (gray_val, gray_val, gray_val)

                    word_bbox = draw.textbbox((0, 0), word, font=micro_font)
                    word_width = word_bbox[2] - word_bbox[0]
                    word_height = word_bbox[3] - word_bbox[1]

                    if x + word_width >= self.width:
                        break
                    if y + word_height >= self.height:
                        x += density
                        continue

                    # 宽松碰撞检测
                    shrink_x = int(word_width * overlap_ratio)
                    shrink_y = int(word_height * overlap_ratio)

                    shrink_x = min(shrink_x, word_width // 2 - 1)
                    shrink_y = min(shrink_y, word_height // 2 - 1)

                    check_box = (
                        x + shrink_x,
                        y + shrink_y,
                        x + word_width - shrink_x,
                        y + word_height - shrink_y
                    )

                    word_occupancy_region = occupancy_mask.crop(check_box)
                    if word_occupancy_region.getbbox() is not None:
                        x += density
                        continue

                    draw.text((x, y), word, font=micro_font, fill=text_color)
                    occupancy_draw.rectangle(check_box, fill=255)

                    max_row_height = max(max_row_height, word_height)
                    row_has_word = True

                    x += word_width + word_spacing
                else:
                    x += density

            if row_has_word:
                y += int(max_row_height * (1.0 - overlap_ratio * 0.5))
            else:
                y += density

        return final_image

# --- 执行示例 ---

if __name__ == "__main__":
    positive_words = [
        "Love", "Safe", "Kind", "Help", "Peace",
        "Hope", "Joy", "Trust", "Calm", "Light",
        "Poetry", "Dream", "Smile", "Life", "Good"
    ]
    target_shape = "DOG"
    custom_font = "arial.ttf"

    generator = TextMosaicGenerator(width=600, height=600, font_path=custom_font)

    print(f"正在生成图像: '{target_shape}' (最大化填充模式)...")

    image = generator.generate(
        macro_text=target_shape,
        micro_words=positive_words,
        density=2,
        min_font_size=12,
        max_font_size=24,
        word_spacing=-5,
        overlap_ratio=0.25,
        letter_spacing=30       # 字母间距
    )

    output_dir = "pic/pic1"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = int(time.time())
    output_filename = os.path.join(output_dir, f"{target_shape}_{timestamp}.png")
    image.save(output_filename)
    print(f"图像已保存为: {output_filename}")