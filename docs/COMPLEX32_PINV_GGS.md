# Complex32 伪逆 GGS 路径

当前 8N 重建默认优先使用 `complex32_pinv`。探测矩阵和正则化伪逆分别以
FP16 实部/虚部平面常驻 GPU，GGS 的场和最终 TM 仍为 complex64。

伪逆由已有的 complex64 Cholesky 缓存离线生成：

```powershell
python -m tools.build_low_precision_pinv_128 --device cuda:0 --measurement-chunk 512
```

8N 生成文件：

- `probe_pinv_128_px4_active512_8N_fp16_real.npy`
- `probe_pinv_128_px4_active512_8N_fp16_imag.npy`
- `probe_pinv_128_px4_active512_8N_fp16.json`

运行 `calibrate_128x128.py` 时，如果这三个文件都存在，8N 配置使用
`complex32_pinv` 和输出批大小 512；文件缺失时自动回退到 Cholesky 和批大小
256。命令行也可以显式传入 `--solver complex32_pinv` 及三个 `--pinv-*` 路径。

## 2026-08-11 RTX 4090 D 实测

- 8N 伪逆尺寸：`16384 x 131072`，存储约 8 GiB。
- 构建耗时：41.1 秒。
- 相同 256 个真实相机像素、20 次迭代：与 Cholesky 的 phase-invariant
  fidelity 平均 0.999732，P05 0.999406，最低 0.998804。
- 误差曲线相对 L2 差：0.000113；最终相对振幅误差差：`8.64e-7`。
- 纯 GGS 时间，批大小 256：complex32 2.36 秒，Cholesky 4.77 秒。
- complex32 批大小 512：3.49 秒；Cholesky 批大小 512 会显存不足，因此处理
  同样 512 个像素需两个 256 块，约 9.55 秒。
- complex32 批大小 1024 在 24 GiB 显卡上仍会显存不足，默认不要超过 512。

计时不包含完整重建后实际光学聚焦验证；正式 TM 完成后仍应运行 pixel-wise
PBR 抽样或全量报告。
