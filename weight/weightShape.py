import os
from safetensors import safe_open

# 替换为你自己的模型文件路径
model_path = "/home/t00906153/weights/glm-4.1v-9b-0624"  # 放着 model-00001-of-00004.safetensors 的目录

# 获取所有 .safetensors 文件
tensor_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]

# 遍历每个文件
with open("model_shapes.txt", "w") as fw:
    for file_name in tensor_files:
        file_path = os.path.join(model_path, file_name)
        print(f"\nLoading tensors from: {file_name}")

        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                print(f"Tensor: {key} | Shape: {tuple(tensor.shape)} | Dtype: {tensor.dtype}")
                line = f"{key} | {tuple(tensor.shape)} | {tensor.dtype}\n"
                fw.write(line)



        