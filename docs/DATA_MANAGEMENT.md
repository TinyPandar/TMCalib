# 数据管理

## Git 边界

Git 只保存源码、测试、Notebook 和文档。以下内容由 `.gitignore` 排除：

- Probe、Pattern、测量 memmap、Cholesky/伪逆缓存和重建 TM；
- 相机图像、质量图、报告 CSV/JSON 和中断文件；
- JUOPT SDK、DLL、Pyd 和其他厂商二进制文件。

这些规则不会删除磁盘上的实验数据。

## 典型数据规模

| 输入与 Pattern 集 | 相机输出 | Probe + Pattern | 测量矩阵（uint16） | TM（complex64） |
| --- | --- | ---: | ---: | ---: |
| 128×128，8N | 128×128 | 约 112 GiB | 4 GiB | 2 GiB |
| 128×128，8N | 26×26 | 共用上行数据 | 169 MiB | 84.5 MiB |
| 160×120，8N | 128×128 | 134.5 GiB | 4.69 GiB | 2.34 GiB |

160×120 Profile 包含 153,600 个 Probe。其 Pattern 逐帧覆盖完整 1024×768 DMD，单独 Pattern 数组约 112.5 GiB；complex64 Probe 约 22.0 GiB。生成、测量和重建放在同一磁盘时，还需为测量、TM、缓存和临时文件预留额外空间。

完整 Pattern 与 Probe 不适合普通 Git 或 Git LFS。

## 实验归档建议

每次正式运行在独立实验存储中保存：

- 原始测量；
- 对应 Probe/Pattern metadata 与校验和；
- 重建配置、误差曲线和结果；
- 质量报告；
- 当前 Git commit：`git rev-parse HEAD`；
- 操作者、设备、曝光、帧率、ROI、偏振通道和时间。

移动大文件前必须结束测量/重建进程。`.partial` 和 `.progress.json` 是断点续算状态，应与对应输出一起搬迁。
