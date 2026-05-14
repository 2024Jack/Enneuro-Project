import cupy as cp
import time

# 创建一个 float16 类型的数组，看它是否能正常运行
try:
    a = cp.array([1.0, 2.0, 3.0], dtype=cp.float16)
    b = cp.array([4.0, 5.0, 6.0], dtype=cp.float16)
    # 执行一个简单的矩阵乘法，它会在现代GPU上自动调用Tensor Cores进行加速
    result = cp.dot(a, b)
    print("CuPy 成功执行了 float16 运算！")
    print("数组数据类型:", result.dtype)
except Exception as e:
    print(f"CuPy 执行 float16 运算时出现问题: {e}")


repeat = 1e3
for dtype in [cp.float16, cp.float32, cp.float64]:
    a = cp.random.rand(1000, 1000).astype(dtype)
    b = cp.random.rand(1000, 1000).astype(dtype)

    # 预热 GPU，确保后续测量更准确
    for _ in range(10):
        cp.dot(a, b)

    start_time = time.time()
    for _ in range(int(repeat)):
        cp.dot(a, b)
    end_time = time.time()

    avg_time = (end_time - start_time) / repeat
    print(f"Average time for dtype {dtype}: {avg_time:.6f} seconds. Total time: {end_time - start_time:.4f} seconds.")