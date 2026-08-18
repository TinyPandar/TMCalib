# 架构与代码谱系

## 配置维度

原目录中的 v4、128 和 llh_v2 并不是三个完全独立的系统，而是以下配置维度的组合：

- 输入网格：32×24 或 128×128；
- 相机输出：128×128 或 26×26；
- 偏振通道：I0 或 I90；
- Pattern 集：64 帧光学测试、4N 或 8N；
- 重建器：普通伪逆、Cholesky 或 planar-complex32 低精度逆。

稳定标识定义在 `calibration_profiles.py`。实验输出名应包含输入 Profile、Pattern 集和偏振通道，避免不同实验互相覆盖。

## 已完成的去重

- `llh_v2` 的 I0/I90 GUI 合并为 `calibrate_128x128_26x26.py --channel ...`。
- 26×26 分支不再维护删减版重建器，而是调用 `tm_reconstruction_128.py` 并传入 `output_shape=(26, 26)`。
- `dmd_pattern_128.py` 只保留一份。
- 原来位于仓库外部的 `holograms` 源码已纳入仓库。

## 暂时保留的重复

三个 GUI 文件仍包含重复的 `CameraHandler`、DMD SDK 绑定和部分聚焦代码。这是有意的过渡状态：这些代码直接控制硬件，在没有模拟硬件测试前贸然抽象会扩大实验风险。

推荐后续顺序：

1. 为 PySpin 和 JUOPT DLL 建立模拟接口及触发顺序测试。
2. 原样抽取公共 `CameraHandler` 和低层 DMD API，不改变控制顺序。
3. 把 Pattern 编码定义为 32×24 与 128×128 两个策略。
4. 抽取公共采集循环，Profile 只提供 shape、文件名和能力开关。
5. 最后合并 GUI 控件和一键流程。

## 重建器

`tm_reconstruction_128.py` 的名称保留是为了兼容现有导入，但其主体按 `ReconstructionConfig.input_shape` 和 `output_shape` 工作。正式重命名前应先保留一个兼容导入层，防止旧实验脚本失效。
