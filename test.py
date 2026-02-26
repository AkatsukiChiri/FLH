import torch
import numpy as np
import matplotlib.pyplot as plt

def swap_rows_cols(A, i, j):
    A = A.clone()
    A[[i, j], :] = A[[j, i], :]
    A[:, [i, j]] = A[:, [j, i]]
    return A

def hadamard(n):
    if n == 1:
        return torch.tensor([[1.]])
    else:
        H = hadamard(n // 2)
        top = torch.cat([H, H], dim=1)
        bottom = torch.cat([H, -H], dim=1)
        return torch.cat([top, bottom], dim=0)

def analyze_matrix_distribution(matrix, bins=200, smooth_window=5, path=None):
    arr = np.array(matrix).flatten()
    sorted_arr = np.sort(arr)

    # 直方图统计
    hist, bin_edges = np.histogram(sorted_arr, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 包络线：滑动平均
    kernel = np.ones(smooth_window) / smooth_window
    hist_smooth = np.convolve(hist, kernel, mode="same")

    # 绘图
    plt.figure()
    plt.hist(sorted_arr, bins=bins)
    plt.plot(bin_centers, hist_smooth)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Distribution with Envelope")
    plt.show()
    
    if path is not None:
        plt.savefig(path)
        plt.close()

    return arr.mean(), arr.std()
    
n = 1024
A = torch.randn(n, n)
A[2, :] += 100
print(A)

H = hadamard(n)
H = swap_rows_cols(H, 0, 2)
print(H)
print(H @ A)
print(analyze_matrix_distribution(A, path="A.png"))
print(analyze_matrix_distribution(H @ A, path="H_A.png"))