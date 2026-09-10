# 32×24 传输矩阵恢复算法

32×24 Profile 的输入自由度是 `24×32=768`。采集文件仍然只有一份：

```text
probe.npy                  (M, 768) complex64
measurements_memmap.npy    (M, N_out) uint16 intensity
```

恢复时会按输出像素分块，把相机强度转换为振幅
`y = sqrt(max(I-dark, 0))`，并求解 `y = |X @ H.T|`。算法选择不改变
Probe/Pattern 采集结果。

## 可选项

| 选项 | 类型 | 作用 |
| --- | --- | --- |
| `GGS21` | 广义 Gerchberg–Saxton | 现有默认路径，保留旧版伪逆/Cholesky 实现 |
| `GS` | Gerchberg–Saxton | 全程使用振幅约束 |
| `RAF21` | Relaxed/Robust Amplitude Flow | 2→1 约束 continuation + 鲁棒残差权重 |
| `RAF` | Robust Amplitude Flow | 鲁棒振幅流 |
| `AF` | Amplitude Flow | 基础振幅流 |
| `TAF` | Truncated Amplitude Flow | 对异常残差做截断 |
| `WF` | Wirtinger Flow | 强度损失梯度基线 |
| `prVBEM` | Variational Bayesian EM | 逐坐标均值场更新并估计有效噪声 |
| `prVAMP` / `prVAM` | Vector AMP | von-Mises 幅度通道 + Gram 特征基中的 LMMSE 更新 |

`prVAM` 是 `prVAMP` 的兼容写法。所有方法都在
`tm_recovery_algorithms.py` 中实现，不依赖相机或 DMD SDK；计算只需要
PyTorch。32×24 的 768 模式规模会预计算一个 `768×768` Gram 矩阵，适合
在 GPU 上按相机输出块运行。prVBEM 在小矩阵上保留逐坐标参考更新，
在 768 模式上自动使用并行变分松弛，避免每个输出块产生数十万次小循环。

## 在界面中使用

### 统一 PySide6 界面

```powershell
python run_gui.py
```

选择 `32 × 24 标准测量` 或 `32 × 24 Cholesky 重建`，在“32×24 恢复算法”
下拉框选择方法，然后执行“恢复传输矩阵”。

### 旧版 Tk 入口

```powershell
python run_calibration.py --profile v4_32x24
```

在 Transmission Matrix 区域选择算法后恢复。也可以在脚本中设置：

```python
controller.recovery_algorithm = "RAF21"  # 或 prVAMP / prVBEM 等
```

默认的兼容文件仍会写到 `reconstructed_filename`、`tm_memmap_filename`
和 `error_curve_filename`。另外会保留带算法后缀的比较文件，例如：
`reconstructed_field_..._raf21.npy`、`recovery_error_..._raf21.npy`。

## 参数建议

32×24 默认使用 200 次迭代、每块 128 个输出像素和 8 次谱初始化迭代。
`RAF21` 默认在约 2/3 迭代处从平方振幅约束切换到振幅约束；`prVAMP` 和
`prVBEM` 的阻尼/更新步长可以通过控制器的
`recovery_damping`、`recovery_step` 调整，`recovery_ratio` 用于 RAF21 的
continuation 切换点。

比较算法时应使用相同的 Probe、暗场设置、强度归一化和输出块大小；不应
只比较未经尺度对齐的 TM 元素，而应同时查看误差曲线、相位保真度和实际
聚焦 PBR。
