import numpy as np
import argparse
import os
import sys

def save_image(data, output_path):
    """
    保存图像数据到文件。
    尝试使用 cv2，如果不可用则使用 PIL。
    """
    # 处理数据类型和范围
    if data.dtype != np.uint8:
        # 如果是浮点数，假设在 0-1 之间，或者是其他范围
        min_val = data.min()
        max_val = data.max()

        print(f"数据范围: min={min_val}, max={max_val}, dtype={data.dtype}")

        if max_val > 255:
            # 可能是 12bit 或 16bit 数据，归一化到 0-255
            # 注意：如果用户想保留原始数据精度，可能需要保存为 16bit png 或 tiff
            # 这里为了通用性，先归一化到 8bit，或者如果用户指定了扩展名支持 16bit...
            pass

        # 简单的归一化到 0-255 用于可视化
        norm_data = ((data - min_val) / (max_val - min_val + 1e-8) * 255).astype(np.uint8)
    else:
        norm_data = data

    try:
        import cv2
        # cv2 默认是 BGR，如果是彩色图片需要转换，这里假设是灰度图或已经处理好
        # 如果是 3 通道且是 RGB，cv2 需要转 BGR
        if len(norm_data.shape) == 3 and norm_data.shape[2] == 3:
             # 假设输入是 RGB，OpenCV 需要 BGR
             norm_data = cv2.cvtColor(norm_data, cv2.COLOR_RGB2BGR)

        cv2.imwrite(output_path, norm_data)
        print(f"已使用 OpenCV 保存图片到: {output_path}")
        return
    except ImportError:
        pass

    try:
        from PIL import Image
        img = Image.fromarray(norm_data)
        img.save(output_path)
        print(f"已使用 Pillow 保存图片到: {output_path}")
        return
    except ImportError:
        pass

    print("错误: 未找到 OpenCV 或 Pillow 库，无法保存图片。")

def main():
    parser = argparse.ArgumentParser(description="从 .npy 文件中提取第 i 张图片并保存。")
    parser.add_argument("index", type=int, help="要提取的图片索引 (从 0 开始)")
    parser.add_argument("--input", "-i", default="measurements_memmap.npy", help="输入的 .npy 文件路径 (默认: measurements_memap.npy)")
    parser.add_argument("--output", "-o", help="输出图片路径 (默认: image_<index>.png)")

    args = parser.parse_args()

    input_path = args.input
    index = args.index

    if not os.path.exists(input_path):
        print(f"错误: 文件 '{input_path}' 不存在。")
        sys.exit(1)

    try:
        # 使用 mmap_mode='r' 以避免将大文件全部加载到内存
        arr = np.load(input_path, mmap_mode='r')
        print(f"成功加载文件: {input_path}")
        print(f"数组形状: {arr.shape}")
        print(f"数据类型: {arr.dtype}")

        # 检查维度
        if arr.ndim < 3:
            print("错误: 数组维度小于 3，无法识别为图片序列 (期望形状如 [N, H, W] 或 [N, H, W, C])。")
            sys.exit(1)

        total_images = arr.shape[0]
        if index < 0 or index >= total_images:
            print(f"错误: 索引 {index} 超出范围 (0 - {total_images - 1})。")
            sys.exit(1)

        # 提取图片数据
        image_data = arr[index]
        print(f"提取索引 {index} 的图片，形状: {image_data.shape}")

        # 确定输出路径
        if args.output:
            output_path = args.output
        else:
            output_path = f"image_{index}.png"

        save_image(image_data, output_path)

    except Exception as e:
        print(f"发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
