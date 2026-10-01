# Heretic 生成头恢复 — 2026-10-01

## 根因与修复

本地旧版 MLX 导出仅保留图像条件编码所需参数，省略了文字生成专用的 `lm_head.weight` 和 `language_model.norm.weight`。原 Heretic 检查点包含这两个张量；不能用随机初始化或直接复用词嵌入替代其独立生成头。

已修复的本地模型：`/Users/vanch/models/mlx/Qwen-Image-2.1-Heretic-mlx-bf16`。

来源为 `pottokao/Qwen-Image-2.1-Text-Encoder-Heretic`，固定版本 `047e54342fc4bfcd2addd54049db9c90bb74731e`。HTTP 范围请求只获取两个 BF16 张量，未下载完整的 4.91 GB / 2.74 GB 源分片。

| 参数 | 形状 | 原始数据大小 | SHA256 |
| --- | --- | --- | --- |
| `lm_head.weight` | 151936 × 4096 | 1,244,659,712 bytes | `a9170f6064e905bb6f9bcbaca6af2c91f9bd44e599a1433188a5cd8926573fd0` |
| `language_model.norm.weight` | 4096 | 8,192 bytes | `fe204de7e848b92e8a758ab5e27632e9b698f29571ca56057d13ba73d0b2012a` |

新权重位于 `text_encoder/generation_head.safetensors`，原索引备份为 `model.safetensors.index.pre-head-20261001T084435856431Z.json`。同目录的 `generation_head.provenance.json` 保存了来源与哈希。新增磁盘约 1.24 GB（1.16 GiB）。其余八个编码器分片及配置文件在操作前后进行了完整 SHA256 比较，全部一致；Transformer、VAE 和 LoRA 未修改。

## 验证

| 检查 | 状态 | 证据 |
| --- | --- | --- |
| 恢复工具回归 | pass | 5 项：原权重/索引保留、精确字节复制和可重复运行、错误形状、截断传输、量化混用与忽略范围请求的拒绝 |
| Qwen 编辑辅助/共享加载/本地 LoRA 相关回归 | pass | 共 126 passed，4.82 秒 |
| 本地 BF16 完整模型加载 | pass | `has_generation_head=True`，生成头和最终归一化层形状正确 |
| 普通图像提示编码 | pass | 同一模型在启用/禁用生成头与最终归一化层时，条件编码逐位一致 |
| 自动定位蒙版 | pass | 从真实参考图返回茶壶框 `[152,315,859,812]`（0–1000 坐标），解析并导出 1024×1024 蒙版 |
| 内置编辑提示重写 | pass | 返回可解析 JSON，扩写“将红茶壶改为深蓝”，描述形状、材质及需保留的场景 |
| 内置编辑结果检查 | pass | 红茶壶原图与既有蓝茶壶编辑图共同输入，返回 `instruction_applied=true`、`outside_unchanged=true` |
| 新恢复工具 lint / 格式 / 类型 | pass | Ruff 与 ty 检查通过 |

三项真实辅助调用分别约 3.12 / 4.98 / 2.22 秒。完整模型加载与全部验证共 18.66 秒，MLX 峰值 32.05 GiB。这是功能验证，非冷机生成速度基准。未新增扩散生成任务；结果检查使用本机此前的真实生成与编辑图。

蒙版来自边界框并作柔边处理，不代表逐像素分割。本次通过说明这组辅助调用与解析链路可用，不保证所有图像的定位或判断准确率。普通生成仍延迟读取生成头，只有调用文字生成辅助功能时才将其读入内存。WebUI 的既有 oMLX PE 路径没有替换为本模型；前端的新生成进程会读取修复后的本地模型。

原始证据保存在 `outputs/heretic-lm-head-2026-10-01/`：`restore-result.json`、前后文件清单、`runtime.log`、`runtime-result.json`、`replies.json`、`auto-mask.png`、回归日志。模型权重和这些大文件保留在本机。

## 复现

```sh
uv run python scripts/restore_qwen21_heretic_head.py \
  --model-path /path/to/Qwen-Image-2.1-Heretic-mlx-bf16
```

工具拒绝覆盖已有未登记的生成头文件，拒绝向量化导出混入 BF16 张量，传输或形状校验失败时不修改原索引。复用已有完整生成头时不重复下载。
