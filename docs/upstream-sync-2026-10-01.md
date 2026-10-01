# 本地上游同步记录 — 2026-10-01

## 结果

| 项目 | 核实的上游 main | 本地处理 |
| --- | --- | --- |
| mflux-community/mflux | `83f4d1dee103674da5f2385251e7794cd7285ba5` | 合入缺少的 27 个上游提交；合并提交 `6e01c0f` |
| Amo643/mflux-paint | `2d5db24942d50b0e8561ee5c4b7e11049b67302a` | 已包含上游全部提交，没有需要合并的新提交 |

两条上游 main 在验证结束时再次通过远端查询核实。本次仅创建本地提交。

## 合入的功能与兼容处理

- Qwen 2.1 共用 Transformer、VAE、语言解码器和权重加载逻辑。旧 `qwen21.reference` 导入保留为兼容入口。
- 文本前缀 KV 缓存与融合 QK normalization / RoPE Metal 内核；加入可选步复用缓存。近似跳步保持默认关闭。
- 编辑 CLI 新增显式蒙版、自动定位蒙版、编辑强度、内置提示重写与结果验证参数。Viggle/Turbo8 仍要求完整的蒸馏采样流程，禁止用较低 strength 截短流程。
- 权重缺失检查、共享 LoRA 格式映射与 RGBA 转 RGB 前的透明度合成修复。
- 其余上游更新随原提交历史合入，包括 Ming-Image 支持、Z-Image CLI/LoRA 更新、独立 Web UI 包命名空间及依赖/CI 修复；没有下载这些模型的权重。
- 保留本地 Heretic 编码器、Viggle v0.3 r128/r256 和 Turbo8、多个效果 LoRA 独立强度、默认独立编辑噪声、独立 1024² 参考编码预算、FP32 VAE 和旧元数据重放策略。
- 修复共享语言解码器对本地 `rope_parameters` 配置的兼容；添加与旧 `rope_scaling` 数值一致的回归检查。
- 补齐模型保存路由和 CI 模型清单中的 Heretic bf16/8-bit/4-bit 登记。
- 前端保留多参考图、自动文生图/编辑路由、LoRA 触发词、图片命名和本地 oMLX PE；启动说明更新为当前 LoRA 版本。

## 验证证据

| 检查 | 状态 | 结果 |
| --- | --- | --- |
| mflux 默认测试选择 | pass | 2271 passed，13 skipped，66 deselected |
| Heretic/共享组件/本地预设专项回归 | pass | 116 passed |
| mflux-paint 回归 | pass | 54 passed |
| 已跟踪 Python 代码 lint / 格式 / 类型 | pass | Ruff 和 ty 均通过；1072 个文件格式检查通过 |
| CI 模型清单生成 | pass | 40 个条目，包含本地三种精度的导出模型 |
| wheel / sdist 构建 | pass | `mflux-0.20.0` 两种包均生成 |
| 实际 WebUI 文生图与单参考图编辑 | pass | 见下表及原始请求、结果、图片 |
| Git 合并冲突与差异检查 | pass | 无未解决冲突；`git diff --cached --check` 通过 |
| 整目录 lint / 类型检查 | fail，既有临时文件 | 根目录未跟踪的评测/上传脚本有原有诊断；没有修改或提交这些脚本 |
| 可选 Diffusers 数值对照及其他模型真实生成 | pending | 本次没有追加运行，不属于默认测试通过结论 |

构建包包含仓库原有的 SeedVR2 固定位置嵌入资源；没有把本地生成图片、模型导出目录或临时评测脚本打入包中。

实测使用已有本地 Heretic MLX bf16、FP32 VAE、Viggle v0.3 r128、scale 1、guidance 1、6 步，输出均为 1024×1024。

| 操作 | 种子 | WebUI 总耗时 | 目视结果 |
| --- | --- | --- | --- |
| 文生图：木桌上的红色陶瓷茶壶，左侧柔和日光 | 42 | 59.34 秒 | 红色茶壶、桌面和灰墙按提示生成 |
| 编辑：将茶壶改为深蓝，保留形状、桌面和光线 | 43 | 423.81 秒 | 茶壶改为蓝色，主体与场景保持一致 |

这些是功能验证耗时。编辑任务与回归测试和调用图索引重叠，不满足冷机、独占负载的测速条件；不能据此推导加速幅度。

原始证据位于 `outputs/upstream-sync-2026-10-01/`：`t2i-request.json`、`edit-request.json`、两份 progress 结果、`t2i-r128-1.png`、`edit-r128-1.png`、测试/检查日志及构建包。该目录按现有规则保留在本机。

## 调用关系检查

GitNexus 1.6.12 的最终提交前检查记录了 179 个变更文件、976 个变更符号、210 个受影响执行流程；返回结果没有 `partial` 或 `truncated`。整次上游合并的影响评级为 critical，反映合并范围大。

专项影响检查：`LanguageModel` 13 个依赖项，LOW；`QwenImage21Edit` 5 个依赖项，LOW；通用 `LoRALoader` 81 个依赖项，MEDIUM。`OVERLAY` 的调用图评级为 UNKNOWN，已通过文本检查确认它由 `build_registry`、`main` 和 CI 提取工作流使用，并实际运行清单生成。动态调用解析存在索引边界，调用图不替代测试证据。

## 现有模型限制

本地 Heretic MLX 导出没有语言生成用的 `lm_head`。文生图、参考编辑和 WebUI 的 oMLX PE 正常保留；新增的核心内置自动蒙版和结果验证需要包含该生成头的权重。内置 `--enhance-prompt` 在缺少生成头时记录警告并保留原提示词。因此本次没有声称这些内置辅助能力已在当前 Heretic 权重上完成实测。

## 本地提交与恢复点

同步前先按功能保存本地改动：mflux 的 `30fdefe`、`c2525ef`、`0803eec`、`6e0f239`；前端的 `ea863a4`、`a2370f9`。随后完成核心合并 `6e01c0f` 和前端启动说明 `e337992`。

两个仓库均有本地恢复分支 `backup/pre-upstream-sync-20261001-152732`，原始工作区备份位于各自 `.git/local-backups/upstream-sync-20261001-152732/`。用户原有的未跟踪日志、脚本和历史计划保留。
