# 硬件环境配置

## Python 与 PySpin

当前 JUOPT Python 扩展绑定 CPython 3.8，因此使用 64 位 Python 3.8。Spinnaker/PySpin 也必须与 Python 和系统位数匹配。

```powershell
python -c "import struct; print(struct.calcsize('P') * 8)"
python -c "import PySpin; print('PySpin OK')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

第一条应输出 `64`。正式大矩阵重建通常需要可用的 CUDA GPU。

## JUOPT SDK

程序默认从仓库根目录下列位置加载 DLL：

```text
JUOPT_DLP V4.0.002 20250522 release/4.DLL/DLL/JUOPT_DLL_V4.dll
```

SDK 和 DLL 不进入 Git。请从厂商安装包恢复，不要从非可信来源复制二进制文件。

## 启动前检查

1. 确认 DMD、相机、电源和触发线连接正确。
2. 确认曝光与光功率不会造成相机饱和或器件损伤。
3. 关闭可能占用相机或 GPU 的其他程序。
4. 检查选定 Profile、Pattern 数量和偏振通道。
5. 优先运行 64 Pattern 测试，再开始完整标定。
6. 完整测量期间不要移动、重命名 memmap 或进度文件。

## 可选远程重建

只有 SSH 远程重建功能需要 Paramiko：

```powershell
pip install -r requirements-optional.txt
```

服务器密码、私钥和个人路径不得提交到仓库。
