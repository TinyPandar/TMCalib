# TMCalib

用于 JUOPT DLP/DMD 与 FLIR/Teledyne 偏振相机的传输矩阵（Transmission Matrix, TM）测量、GGS2-1 重建和逐点聚焦验证程序。

本仓库由原实验目录中的 `combined_app_v4.py`、`combined_app_v4_128.py` 和 `llh_v2` 整理而成。实验数据、厂商 SDK 和历史代码快照没有复制进仓库。

## 支持的配置

| Profile | DMD 逻辑输入 | DMD 有效区 | 相机输出 | 说明 |
| --- | --- | --- | --- | --- |
| `v4_32x24` | 32×24 | 1024×768 | 128×128 | 原 v4 粗网格程序 |
| `v4_32x24_cholesky` | 32×24 | 1024×768 | 128×128 | v4 的 Cholesky 重建入口 |
| `fourfold_128x96` | 128×96 | 1024×768 | 128×128 | 4 倍逻辑场以 8×8 宏像素、4×4 `holo_SP` 编码铺满 DMD |
| `fivefold_160x120` | 160×120 | 1024×768 | 128×128 | 5 倍逻辑场先扩展为 256×192 超像素网格，再以 4×4 编码铺满 DMD |
| `dense_128x128` | 128×128 | 中央 512×512 | 128×128 | 稠密输入程序 |
| `dense_128x128_roi26` | 128×128 | 中央 512×512 | 26×26 | I0/I90 可选偏振通道 |

`llh_v2` 原来的 I0、I90 两个大文件已收敛为一个 `calibrate_128x128_26x26.py`，通道由启动参数选择。

## 目录结构

```text
tmcalib-repo/
├── run_calibration.py                 # 统一启动器
├── calibrate_v4_32x24.py              # 32×24 主程序
├── calibrate_128x96.py                # 128×96→128×128、全 DMD 主程序
├── calibrate_160x120.py               # 160×120→128×128、全 DMD 主程序
├── calibrate_128x128.py               # 128×128→128×128 主程序
├── calibrate_128x128_26x26.py         # 128×128→26×26，I0/I90 共用
├── calibration_profiles.py            # Profile 与偏振通道定义
├── dmd_pattern_128x96.py              # 128×96→256×192 SP→全 DMD 映射
├── dmd_pattern_160x120.py             # 160×120→256×192 SP→全 DMD 映射
├── dmd_pattern_128.py                 # 中央 512×512 DMD 映射
├── tm_reconstruction_128.py           # 通用分块 GGS2-1 重建器
├── low_precision_pinv.py              # 低精度逆矩阵存储/计算
├── holograms/                          # 仓库内全息编码实现
├── tools/                              # 数据生成、诊断和分析工具
├── tests/                              # 合成数据回归测试
├── notebooks/                          # 分析 Notebook
└── docs/                               # 架构、硬件和数据说明
```

## 环境

当前厂商 Python 扩展名为 `JUOPT_V4_PYTHON_DLL.cp38-win_amd64.pyd`，因此测量环境固定为 Windows x64 + CPython 3.8。

```powershell
conda env create -f environment.yml
conda activate tmcalib
```

或者先安装匹配显卡驱动的 PyTorch/CUDA，再安装普通依赖：

```powershell
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`PySpin` 和 JUOPT SDK 必须通过硬件厂商安装，不能由 `pip` 安装。详见 [硬件配置](docs/HARDWARE_SETUP.md)。

## 启动

必须从仓库根目录启动，以保持 SDK、Pattern 和输出文件的相对路径一致。

```powershell
# 原 v4：32×24 输入
python run_calibration.py --profile v4_32x24

# 4 倍：128×96 输入按 8×8 宏像素铺满 DMD，128×128 相机输出
python run_calibration.py --profile fourfold_128x96

# 5 倍：160×120 输入扩展后铺满 DMD，128×128 相机输出
python run_calibration.py --profile fivefold_160x120

# 128×128 输入、128×128 相机输出
python run_calibration.py --profile dense_128x128

# 128×128 输入、26×26 相机输出、I0 通道
python run_calibration.py --profile dense_128x128_roi26 --channel I0

# 同一程序切换到 I90
python run_calibration.py --profile dense_128x128_roi26 --channel I90
```

程序会直接控制实验硬件。启动 GUI 前应确认 DMD、相机、触发线、曝光和光路功率处于安全状态。

## 4 倍数据生成

128×96 输入共有 12,288 个自由度，默认 8N 数据集包含 98,304 个随机 16 级相位 Probe。每个逻辑输入沿两个方向各重复到 2 个光学超像素，再以 4×4 `holo_SP` 编码，因此每个输入对应一个对齐的 8×8 DMD 宏像素：

```powershell
python -m tools.generate_probe_samples_128x96_8n --output-dir pregenerated_patterns_128x96_fill_8N_full

# 中断后从 generation_progress.json 继续
python -m tools.generate_probe_samples_128x96_8n --output-dir pregenerated_patterns_128x96_fill_8N_full --resume
```

完整 Probe + Pattern 恰好为 81 GiB。生成器会预检磁盘空间并分批写入 `.npy` memmap；完整 1024×768 DMD 区域均参与编码，没有外围 zero padding。

## 5 倍数据生成

160×120 输入共有 19,200 个自由度，默认 8N 数据集包含 153,600 个随机 16 级相位 Probe。生成器采用中心对齐最近邻，把每个 Probe 扩展到完整 256×192 超像素网格，再做 4×4 `holo_SP` 编码：

```powershell
python -m tools.generate_probe_samples_160x120_8n --output-dir pregenerated_patterns_160x120_fill_8N_full

# 中断后从 generation_progress.json 继续
python -m tools.generate_probe_samples_160x120_8n --output-dir pregenerated_patterns_160x120_fill_8N_full --resume
```

完整 Probe + Pattern 约 134.5 GiB；生成器会预检磁盘空间并分批写入 `.npy` memmap。横向和纵向的每个源像素分别占 1 或 2 个光学超像素，因此 DMD 端对应 4 或 8 个微镜像素；完整 1024×768 区域均参与编码，没有外围 zero padding。

## 数据与 SDK

克隆仓库后，需单独恢复 Pattern 数据和厂商 SDK。默认路径仍兼容原程序，例如：

```text
JUOPT_DLP V4.0.002 20250522 release/4.DLL/DLL/JUOPT_DLL_V4.dll
pregenerated_patterns_8N/
pregenerated_patterns_128x96_fill_8N_full/
pregenerated_patterns_160x120_fill_8N_full/
pregenerated_patterns_128_px4_active512_8N_full/
```

这些目录已由 `.gitignore` 排除。不要把数 GiB 到上百 GiB 的 Pattern、测量矩阵或 TM 提交到 Git。详见 [数据管理](docs/DATA_MANAGEMENT.md)。

## 工具与测试

分析工具以模块方式从仓库根目录运行：

```powershell
python -m tools.generate_probe_samples_128x96_8n --help
python -m tools.generate_probe_samples_160x120_8n --help
python -m tools.generate_probe_samples_128 --help
python -m tools.analyze_full_measurements_128 --help
```

静态编译与合成数据测试：

```powershell
python -m compileall -q .
python -m unittest discover -s tests -v
```

低精度伪逆测试需要 CUDA；没有 CUDA 时相关测试会自动跳过。

## 代码状态

- 相机、JUOPT DMD 和 GUI 仍来自经过实验使用的单文件程序，以降低第一次仓库化对硬件行为的影响。
- 128×96 Profile 使用对齐的 8×8 宏像素，复用 128×128 相机采集、低显存重建和整数倍 GPU 聚焦编码流程。
- 160×120 Profile 复用已验证的 128×128 相机采集顺序，仅替换输入维度、数据契约和 DMD 编码策略。
- 26×26 的 I0/I90 已参数化，并统一使用最新版可配置重建器。
- 下一步重构应逐步抽取公共相机、DMD 和采集类，并用模拟硬件测试保护行为；不要一次性重写硬件控制链。
- 项目当前未附带开源许可证；公开发布前请由代码所有者选择许可证。
