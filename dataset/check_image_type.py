import pandas as pd
import json
import numpy as np
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms

df = pd.read_parquet("/home/s30034190/geo3k/train.parquet")
image_data = df['images'][0]

print("原始类型：", type(image_data))
print("内容预览：", repr(str(image_data)[:100]))

# 读取第一个图像字段
image_entry = df['images'][0]

print("字段类型：", type(image_entry))
print("内容预览（前100字符）：", repr(str(image_entry)[:100]))

# ✅ 正确处理方式
if isinstance(image_entry, np.ndarray) and isinstance(image_entry[0], dict) and 'bytes' in image_entry[0]:
    image_bytes = image_entry[0]['bytes']  # 取出二进制数据
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    print("✅ 图像加载成功，尺寸：", image.size)
    image.show()  # 或保存 image.save("test.jpg")
# Case 1: 如果是字符串但不是 base64，尝试 json.loads（可能是保存的 list）
elif isinstance(image_data, str):
    try:
        parsed = json.loads(image_data)
        if isinstance(parsed, list):
            print(f"✅ 成功解析为 list（长度: {len(parsed)}）")
            # 转换为图像
            array = np.array(parsed, dtype=np.uint8)
            if array.ndim == 1:
                print("📏 数据为 1D，尝试 reshape 为灰度图")
                side = int(np.sqrt(array.shape[0]))
                array = array[:side*side].reshape((side, side))
            elif array.ndim == 2:
                print("🖼️ 数据为灰度图")
            elif array.ndim == 3:
                print("🖼️ 数据为彩色图")
            img = Image.fromarray(array)
            img.show()
        else:
            print("⚠️ JSON 解析成功但不是 list 类型")
    except Exception as e:
        print("❌ json.loads 失败，错误信息：", e)

# Case 2: 如果是 list 直接当像素向量
elif isinstance(image_data, list):
    print(f"✅ 字段为 list，长度: {len(image_data)}")
    array = np.array(image_data, dtype=np.uint8)
    # 尝试 reshape（假设是灰度图或 RGB 图）
    if array.ndim == 1:
        side = int(np.sqrt(array.shape[0]))
        array = array[:side*side].reshape((side, side))
    elif array.ndim == 2:
        pass
    elif array.ndim == 3:
        pass
    img = Image.fromarray(array)
    img.show()

# Case 3: 如果是 bytes，说明是压缩后的图像
elif isinstance(image_data, bytes):
    print("字段为 bytes，尝试作为图片解码")
    from io import BytesIO
    img = Image.open(BytesIO(image_data))
    img.show()

else:
    print("❓未知类型，不能解析为图像")
