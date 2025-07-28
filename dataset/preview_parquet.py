import pandas as pd

# 路径设置
file_path = "/home/s30034190/geo3k/train.parquet"

# 读取 parquet 文件
try:
    df = pd.read_parquet(file_path)
except Exception as e:
    print(f"❌ 读取 parquet 文件失败: {e}")
    exit(1)

# 展示所有字段（列名）
print("✅ 文件成功读取，包含的字段如下：")
print(df.columns.tolist())

# 显示前几行数据（默认5行）
print("\n📝 数据预览（前5行）：")
print(df.head())

# 如果你想只看某些字段，可以像这样筛选
# 修改下面这个列表以查看特定列（如果你想）
desired_fields = ['text', 'label']  # 举例：你想看文本和标签列

if all(field in df.columns for field in desired_fields):
    print(f"\n🔍 仅展示字段：{desired_fields}")
    print(df[desired_fields].head())
else:
    print(f"\n⚠️ 你指定的字段中有不存在的列：{desired_fields}")
