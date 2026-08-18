# 数据管理

## Git 边界

Git 只保存源码、测试、Notebook 和文档。以下内容由 `.gitignore` 排除：

- Probe、Pattern、测量 memmap、Cholesky/伪逆缓存和重建 TM；
- 相机图像、质量图、报告 CSV/JSON 和中断文件；
- JUOPT SDK、DLL、Pyd 和其他厂商二进制文件。

这些规则不会删除磁盘上的实验数据。

## 典型数据规模

以 128×128 输入、8N Probe 为例：

| 相机输出 | 测量矩阵（uint16） | TM（complex64） |
| --- | ---: | ---: |
| 128×128 | 4 GiB | 2 GiB |
| 26×26 | 169 MiB | 84.5 MiB |

两种输出共用的完整 8N Pattern 与 Probe 约 112 GiB，因此不适合普通 Git 或 Git LFS。

## 实验归档建议

每次正式运行在独立实验存储中保存：

- 原始测量；
- 对应 Probe/Pattern metadata 与校验和；
- 重建配置、误差曲线和结果；
- 质量报告；
- 当前 Git commit：`git rev-parse HEAD`；
- 操作者、设备、曝光、帧率、ROI、偏振通道和时间。

移动大文件前必须结束测量/重建进程。`.partial` 和 `.progress.json` 是断点续算状态，应与对应输出一起搬迁。
