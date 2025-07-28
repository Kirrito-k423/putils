import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm

INPUT_PARQUET = "/home/s30034190/geo3k/train.parquet"
OUTPUT_PARQUET = "train_resized_16k.parquet"

df = pd.read_parquet(INPUT_PARQUET)

new_images = []
for row in tqdm(df.itertuples(), total=len(df)):
    try:
        image_bytes = row.images[0]['bytes']
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize((1792, 1792))
        
        # 转换成PNG bytes
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
        
        # 放回字典
        new_images.append([{'bytes': png_bytes}])
    except Exception as e:
        print(f"❌ 处理失败 idx={row.Index}，跳过：{e}")
        new_images.append(row.images)  # 保留原图或 None

# 替换images列
df['images'] = new_images

# 保存
df.to_parquet(OUTPUT_PARQUET, index=False)
print(f"✅ 已保存 resized parquet 到 {OUTPUT_PARQUET}")
