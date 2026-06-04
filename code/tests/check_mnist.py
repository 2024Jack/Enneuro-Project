"""检查 MNIST 数据格式"""
import pickle
import numpy as np

with open('D:/Undergraduate/WZH/EnNeuro/tmp/Enneuro-Project/code/tests/testdata/MNIST_data/mnist.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"数据类型: {type(data)}")
if isinstance(data, dict):
    print(f"键: {list(data.keys())}")
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}, dtype: {value.dtype}")
        else:
            print(f"  {key}: {type(value)}")
elif isinstance(data, tuple):
    print(f"元组长度: {len(data)}")
    for i, item in enumerate(data):
        if isinstance(item, np.ndarray):
            print(f"  [{i}]: {item.shape}, dtype: {item.dtype}")
        else:
            print(f"  [{i}]: {type(item)}")