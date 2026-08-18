# TMCalib

用于 JUOPT DLP/DMD 与 FLIR/Teledyne 偏振相机的传输矩阵（Transmission Matrix, TM）测量、GGS2-1 重建和逐点聚焦验证程序。

本仓库由原实验目录中的 `combined_app_v4.py`、`combined_app_v4_128.py` 和 `llh_v2` 整理而成。实验数据、厂商 SDK 和历史代码快照没有复制进仓库。

## 支持的配置

| Profile | DMD 逻辑输入 | DMD 有效区 | 相机输出 | 说明 |
| --- | --- | --- | --- | --- |
| `v4_32x24` | 32×24 | 1024×768 | 128×128 | 原 v4 粗网格程序 |
| `v4_32x24_cholesky` | 32×24 | 1024×768 | 128×128 | v4 的 Cholesky 重建入口 |
| `dense_128x128` | 128×128 | 中央 512×512 | 128×128 | 稠密输入程序 |
| `dense_128x128_roi26` | 128×128 | 中央 512×512 | 26×26 | I0/I90 可选偏振通道 |

`llh_v2` 原来的 I0、I90 两个大文件已收敛为一个 `calibrate_128x128_26x26.py`，通道由启动参数选择。

## 目录结构

```text
tmcalib-repo/
├── run_calibration.py                 # 统一启动器
├── calibrate_v4_32x24.py              # 32×24 主程序
├── calibrate_128x128.py               # 128×128→128×128 主程序
├── calibrate_128x128_26x26.py         # 128×128→26×26，I0/I90 共用
├── calibration_profiles.py            # Profile 与偏振通道定义
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

# 128×128 输入、128×128 相机输出
python run_calibration.py --profile dense_128x128

# 128×128 输入、26×26 相机输出、I0 通道
python run_calibration.py --profile dense_128x128_roi26 --channel I0

# 同一程序切换到 I90
python run_calibration.py --profile dense_128x128_roi26 --channel I90
```

程序会直接控制实验硬件。启动 GUI 前应确认 DMD、相机、触发线、曝光和光路功率处于安全状态。

## 数据与 SDK

克隆仓库后，需单独恢复 Pattern 数据和厂商 SDK。默认路径仍兼容原程序，例如：

```text
JUOPT_DLP V4.0.002 20250522 release/4.DLL/DLL/JUOPT_DLL_V4.dll
pregenerated_patterns_8N/
pregenerated_patterns_128_px4_active512_8N_full/
```

这些目录已由 `.gitignore` 排除。不要把数 GiB 到上百 GiB 的 Pattern、测量矩阵或 TM 提交到 Git。详见 [数据管理](docs/DATA_MANAGEMENT.md)。

## 工具与测试

分析工具以模块方式从仓库根目录运行：

```powershell
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
- 26×26 的 I0/I90 已参数化，并统一使用最新版可配置重建器。
- 下一步重构应逐步抽取公共相机、DMD 和采集类，并用模拟硬件测试保护行为；不要一次性重写硬件控制链。
- 项目当前未附带开源许可证；公开发布前请由代码所有者选择许可证。
