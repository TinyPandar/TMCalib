# 128×96 振幅级数标定实验

## 实验问题

测量 4 倍 `fourfold_128x96` 配置中，DMD `holo_SP` 编码能够稳定区分多少个复光场振幅级，并找出：

- 与零振幅显著不同的最低指令振幅；
- 相邻振幅级是否可分辨；
- 指令振幅到实测振幅的非线性；
- 线性化调制所需的反向 LUT；
- 相机饱和、激光漂移和非单调区间。

该实验使用全场均匀振幅、固定零相位。对于线性光学系统，输入场整体乘以 (A) 后，相机输出场也整体乘以 (A)，因此暗场校正后的输出强度应满足：

\[
I(A)-I_{\mathrm{dark}}\propto A^2.
\]

由此定义实测归一化振幅：

\[
\hat A=
\sqrt{\max\left(
\frac{I(A)-I_{\mathrm{dark}}}
{I_{\mathrm{white}}-I_{\mathrm{dark}}},0
\right)}.
\]

## 默认协议

| 参数 | 默认值 |
| --- | ---: |
| 逻辑输入 | 128×96 |
| DMD | 1024×768 |
| 相机 ROI | 128×128 |
| 指令振幅 | 0.000–1.000 |
| 级数 | 41（步长 0.025） |
| 重复次数 | 10 |
| 相位 | 0 rad |
| 相邻可分辨阈值 | 合并标准误差的 3 倍 |
| 默认曝光 | 60 μs |

每次重复内部随机打乱41个振幅级，并按以下顺序加入参考帧：

```text
黑场 → 满幅参考 → 随机顺序的41级振幅 → 满幅参考 → 黑场
```

这样可以使用每次重复自己的黑场和满幅参考做归一化，降低激光功率慢漂和相机偏置的影响。零振幅与满振幅也保留在正式级数中，用于检查参考帧是否引入额外偏差。

## 运行方法

必须从仓库根目录运行。

### 1. 只生成协议和DMD图样，不打开硬件

```powershell
python -m tools.amplitude_level_calibration_128x96 --prepare-only
```

默认输出到：

```text
experiments/amplitude_levels_128x96/<timestamp>/
```

### 2. 采集并自动分析

确认激光功率、衰减片、光圈、DMD触发线和相机曝光安全后运行：

```powershell
python -m tools.amplitude_level_calibration_128x96 --acquire
```

指定输出目录或曝光时间：

```powershell
python -m tools.amplitude_level_calibration_128x96 `
  --acquire `
  --output-dir experiments/amplitude_levels_128x96/run_01 `
  --exposure-us 60
```

### 3. 重新分析已有采集

重新分析时必须保持和采集时相同的 `--level-count`、`--repeats` 与 `--seed`：

```powershell
python -m tools.amplitude_level_calibration_128x96 `
  --analyze experiments/amplitude_levels_128x96/run_01/raw_frames.npy `
  --output-dir experiments/amplitude_levels_128x96/run_01
```

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `protocol.json` | 实验配置、维度、种子与归一化定义 |
| `sequence.csv` | 实际投影顺序及每帧标签 |
| `unique_level_patterns.npy` | 41张唯一振幅DMD全息图样 |
| `raw_frames.npy` | 相机原始128×128帧，仅采集模式生成 |
| `per_capture.csv` | 每帧暗场/参考校正结果 |
| `amplitude_response.csv` | 每个振幅级的均值、标准差与可分辨性 |
| `amplitude_response.png` | 振幅响应和强度平方律曲线 |
| `amplitude_lut.json` | 256级目标振幅到DMD指令振幅的反向LUT |
| `experiment_summary.json` | 最低可检测振幅、RMSE、幂指数和单调性 |

## 结果判读

1. `intensity_power_law_exponent` 理想值为2；明显偏离时检查相机伽马、饱和、背景光和零级漏光。
2. `amplitude_rmse` 衡量线性振幅误差，而不是强度误差。
3. `lowest_amplitude_distinguishable_from_zero` 给出当前噪声水平下能与零振幅区分的最低级。
4. `adjacent_distinguishable_transition_count` 越接近40，说明41级调制越可用。
5. `monotonic_violation_count` 非零时，不应直接使用原始指令振幅；先检查漂移和饱和，再使用导出的单调化LUT。

## 实验控制要求

- 所有振幅级必须使用相同的零相位、曝光、增益、光圈和激光功率。
- 相机保持 Gamma=1、Gain=0，并记录实际曝光时间。
- 满幅参考像素饱和比例应为0；出现饱和时先降低曝光或光功率后整组重测。
- 不要按振幅从小到大顺序采集，否则激光漂移会被误认为响应非线性。
- 比较不同光圈或曝光时，每个设置分别运行一套完整实验，不要混合归一化参考。
- 原始帧和报告保留在同一时间戳目录，不覆盖此前实验。

