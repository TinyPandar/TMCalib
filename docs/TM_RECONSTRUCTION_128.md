# 128×128 TM 重建

## 从界面运行

运行 `calibrate_128x128.py`，在 Transmission Matrix 区域点击
Recover Transmission Matrix (Local)。不需要配置环境变量。

默认参数：

- 输入自由度：128×128（16,384）
- DMD 映射：每个输入点对应一个 4×4 超像素，中央 512×512 有效
- DMD 留空：左右各 256 像素，上下各 128 像素
- 相机输出：128×128（16,384）
- 探测帧：65,536
- 算法：GGS2-1，200 次迭代
- 线性求解：正则化 Cholesky，ridge=1e-4
- 输出分块：每块 512 个相机像素
- 设备：自动选择可用且更快的 CUDA GPU

新的完整 probe 和 Pattern 位于
pregenerated_patterns_128_px4_active512_full：

- probe.npy：形状 (65536, 128, 128)，complex64，约 8 GiB
- patterns_pregenerated.npy：形状 (65536, 768, 1024)，uint8，约 48 GiB
- 每张 Pattern 只有中央 512×512 区域可能非零

如需重新生成：

    python -m tools.generate_probe_samples_128 --full

第一次重建会生成 probe_cholesky_128_px4_active512.npy（约 2 GiB）。只要
probe.npy 和 ridge 没有变化，后续运行会直接复用该缓存。

正式输出：

- reconstructed_field_128_px4_active512.npy：标准 NPY，形状
  (16384, 16384)，complex64，约 2 GiB
- ggs21_error_curve_128_px4_active512.npy：平均相对振幅误差曲线
- tm_reconstruction_128_px4_active512.json：重建参数、耗时和最终误差

重建过程中使用 reconstructed_field_128_px4_active512.npy.partial 和对应的
.progress.json 保存进度。如果程序中断，重新点击本地重建按钮会从已完成
的输出块继续。正式 TM 只有在全部输出块完成并归一化后才会替换。

## 命令行运行

请使用项目现有的 py38 环境：

    python tm_reconstruction_128.py

## 先做部分像素测试

直接运行下面的独立测试脚本，不会覆盖正式 TM：

    python -m tools.run_tm_reconstruction_128_test

默认恢复相机第 64 行中央的 64 个连续像素（x=32 到 95），迭代 40 次。
输出为 reconstructed_field_128_px4_active512_test_subset.npy，形状是
(64, 16384)。它不能代替完整 TM 做任意位置聚焦，但可以验证这 64 个位置
的实际聚焦效果。

生成测试 TM 后，回到 `calibrate_128x128.py`：

1. 初始化 DMD。
2. 保持默认坐标 X=64、Y=64（或选择 y=64、x=32 到 95）。
3. 点击 Test Focus (Partial TM)。

界面会显示聚焦后的相机图像、实际峰值位置、目标点强度和目标点 PBR。选择
测试范围外的坐标时不会错误读取 TM，而是提示该像素尚未恢复。

## Pixel-wise 聚焦报告

完整 TM 生成后，点击 Full 128x128 Pixel-wise + PBR Report。默认
Stride=1、Max=0，会依次测试相机 ROI 的全部 16384 个位置。程序会记录每个
点的坐标、目标点强度、全图峰值与峰值坐标、峰值到目标的距离、背景强度、
目标点 PBR 和失败信息。

测试过程中 Camera View 会同步显示最近完成采集的目标点画面，并更新该点的
目标 PBR、峰值强度和背景强度。如果绘图稍慢，只保留最新帧，不会拖住采集。

结果保存在 pixelwise_focus_results_128_px4_active512 目录：

- points.csv：所有测试点的逐点记录
- maps.npz：128×128 的 PBR、目标强度和峰值距离数组；未测点为 NaN
- distribution.png：目标点强度与目标 PBR 分布
- pbr_heatmap.png：PBR 空间热力图；失败或未测点显示为灰色
- summary.json：成功率和均值、中位数、P95、最大值等摘要

只有 Stride=1、Max=0 才是全部 16384 个相机像素。Stride 大于 1 或 Max
大于 0 时属于抽样/限量检查。仅 50 ms 的投影稳定等待累计就约 13.7 分钟，
加上全息图加载、相机采集和文件操作后，实际全量测试会更久。

## 傅里叶一级恢复验证

可对中央 512×512 二值 Pattern 做 FFT，圆形滤取一级后恢复对应 Probe：

    python -m tools.analyze_fourier_probe_recovery_128

默认使用半径 96 像素的圆形孔径。在 NumPy FFT 坐标中，直接恢复原 Probe
的一级相对频谱中心偏移为 (Δx=-128, Δy=-32)；对称一级恢复共轭 Probe。
物理光路中的坐标方向可能因反射和相机方向而翻转。

可以用 --count 和 --iterations 改测试规模，例如只恢复 8 个像素：

    python -m tools.run_tm_reconstruction_128_test --count 8 --iterations 40

当前机器的实测基准表明，全量 200 次迭代预计约需 1–2 小时。实际时间取决于
GPU 是否被其他程序占用。进度会细化到当前输出块和迭代次数。

如需只验证一个相机像素而不覆盖正式结果：

    python tm_reconstruction_128.py --output-start 8256 --output-count 1 --iterations 40 --output reconstructed_field_128_px4_active512_check.npy --error-curve ggs21_error_curve_128_px4_active512_check.npy --metadata tm_reconstruction_128_px4_active512_check.json

adjoint 求解器仅用于快速排查文件和 GPU 链路；正式重建应保持默认的
cholesky，因为随机探测矩阵在四倍过采样下仍不能当作严格酉矩阵。
