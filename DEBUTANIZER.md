# 脱丁烷塔 MOMENT 软测量基准线

状态：本次仅编写代码与检查文本，**尚未执行本适配的脚本、测试、训练或推理**。
下列命令供后续在 Linux 主机执行；没有现成的训练结果或精度结论。

## 任务与协议

CSV 表头为 `u1,u2,u3,u4,u5,u6,u7,y`，当前文件有 2394 个样本。
行顺序视为采样时间顺序，不做随机时间划分。目标是用截至 t 时刻的过程观测预测
同一时刻的 `y(t)`，属于软测量回归，不是未来多步时间序列预测。
本适配不调用官方 forecasting 头，而是为预训练编码器增加标量回归头。

默认目标区间与 `lite` 项目的 SAE / VW-SAE 相同：

| 集合 | 样本数 | 原始样本索引（0 起，含端点） |
| --- | ---: | --- |
| 训练 | 800 | 4–803 |
| 验证 | 200 | 804–1003 |
| 测试 | 1390 | 1004–2393 |

先确定目标行划分，再构造各目标行的因果历史窗口。验证和测试可以使用之前已发生的
观测作为上下文，任何窗口都不访问目标行之后的数据。训练批次可以打乱，集合不打乱。
X / y 的均值和标准差只用训练目标行所对应的数据拟合，验证只用于早停和选最佳轮次，
测试只在最佳模型确定后评估。输入文件可能已经经过上游缩放；由于未提供上游处理参数，
本适配只保证自身处理不利用验证/测试统计量，不能证明原 CSV 的上游处理没有泄漏。
指标回到 CSV 的 y 数值尺度，不解释为原始物理浓度单位。

## 输入模式与模型

默认 `--features process`：7 个过程变量的历史窗口，不把任何 y 作为模型输入。
`--features causal13`：每个时刻构造以下特征，再对这些特征取历史窗口：

```text
u1(t), u2(t), u3(t), u4(t), u5(t),
u5(t-1), u5(t-2), u5(t-3), (u6(t)+u7(t))/2,
y(t-1), y(t-2), y(t-3), y(t-4)
```

causal13 不使用当前或未来 y，但假设每一步都能获取过去的真实质量测量值。
预测不会把自己的输出递归代回历史。历史 y 缺失时跳过无法构造当前特征的行；
后续有效行的窗口从最近一段连续有效历史重新开始，不压缩时间轴、不填补缺失标签。

**相同目标区间不等于完全相同的信息条件**：process 模式没有 SAE 的历史 y；
causal13 模式虽然每个时刻使用同样的 13 维特征，但 MOMENT 的多时刻窗口包含更多历史。
比较时必须同时报告特征模式与窗口长度，不能直接宣称输入信息完全公平。

默认模型为 `AutonLab/MOMENT-1-small`，支持替换为 Base/Large 或本地权重目录：

1. 训练集标准化后的窗口转为 `[batch, channels, 512]`，默认实际上下文为 32 步。
2. 上下文从右侧对齐；不足一个完整 patch 时因果重复最早已知特征补齐该 patch，
   其余左侧补零并使用 mask。补齐步长只为匹配原模型 patch，不读取未来、不丢弃起始测试行。
3. MOMENT 取 `reduction="none"`，按有效 patch 做平均，再拼接各通道的特征，保留变量身份。
4. 默认附加真实窗口的均值、标准差与最后观测值。MOMENT 的 RevIN 会移除局部水平与尺度，
   这些统计量为回归头保留相应信息；它们不包含人工补齐值。用 `--no-window-stats` 可关闭。
5. 使用一个新初始化的线性回归头预测标量 y。该头需要监督训练，不能称为零样本软测量。

`--mode probe`（默认）冻结全部 MOMENT 参数，并关闭主干 dropout，训练前仅缓存训练/验证
特征到 CPU 内存，然后只训练回归头；不会每轮重复编码这些样本。
`--mode finetune` 更新主干和回归头，分别使用 `--backbone-lr` 与 `--lr`。
`--gradient-checkpointing` 只在完整微调时生效。实现统一使用 float32，支持 CUDA / MPS / CPU；
本适配尚未在这些设备上执行验证。

基准线应标记完整配置，如 `MOMENT-Small + linear head + window statistics (process, context=32)`。
若希望观察预训练表示本身的作用，应另外报告关闭统计量的结果，并与相同窗口统计量的
简单回归模型比较。可以更改上下文，但它必须是 8 的倍数，范围 8–512。

## 后续 Linux 运行

本适配只使用已声明的核心依赖，不新增第三方包，不需要 PTB-XL 或 Timeseries-PILE。
在 `moment` 仓库根目录准备环境，然后将 CSV 放到主机可访问的位置：

```bash
uv sync --locked --extra cu128
```

冻结主干基准线（首次训练时会下载公开预训练权重）：

```bash
uv run --no-sync python scripts/debutanizer.py train \
  --data /path/to/debutanizer.csv \
  --output outputs/debutanizer/process_probe_seed42 \
  --device cuda --features process --mode probe --seed 42
```

使用 SAE 对应的 13 维特征：

```bash
uv run --no-sync python scripts/debutanizer.py train \
  --data /path/to/debutanizer.csv \
  --output outputs/debutanizer/causal13_probe_seed42 \
  --device cuda --features causal13 --mode probe --seed 42
```

完整微调示例（需按 GPU 内存调整批大小）：

```bash
uv run --no-sync python scripts/debutanizer.py train \
  --data /path/to/debutanizer.csv \
  --output outputs/debutanizer/process_finetune_seed42 \
  --device cuda --mode finetune --batch-size 8 --backbone-lr 1e-5 \
  --gradient-checkpointing
```

默认最多 100 轮、早停耐心 15、回归头学习率 1e-3、AdamW weight decay 0.01。
Small/Base/Large 可通过 `--model` 切换；用 `--revision` 固定 Hub commit，或提供本地
基础权重目录并加 `--local-files-only`。训练不会覆盖已有非空输出目录，避免实验结果混用。
`train` 子命令是必需的，直接打开脚本不会自动开始训练。

## 输出与加载

| 文件 | 内容 |
| --- | --- |
| `best.pt` | 最佳轮次的完整主干/回归头参数、模型配置、特征定义、训练配置、X/y scaler、数据 SHA-256 与版本 |
| `history.json` | 每轮训练/验证标准化 MSE |
| `metrics.json` | 时间划分、RMSE / MAE / MSE / R²、训练均值与上一时刻 y 基线 |
| `predictions.csv` | 测试行索引、真实 y、预测 y、训练均值和 persistence |

Persistence 使用真实 `y(t-1)`，process 模式的 MOMENT 不使用这个额外信息。
R² 为 `1-SSE/SST`；常量目标时记为 null。另输出 `sqrt(SSE/(N-1))`，便于对照 SAE 报告。

`evaluate` 只复评保存的测试区间，要求 CSV 内容哈希与训练时一致：

```bash
uv run --no-sync python scripts/debutanizer.py evaluate \
  --checkpoint outputs/debutanizer/process_probe_seed42/best.pt \
  --data /path/to/debutanizer.csv \
  --output outputs/debutanizer/process_probe_seed42_reeval --device cuda
```

`predict` 不拟合参数，可处理新 CSV；输出不自动作为独立测试报告。检查点包含完整参数，
加载时直接从保存的模型配置重建，不需要联网或重新下载基础权重。
可将 `best.pt` 复制到 Mac，在已安装核心依赖的环境中运行：

```bash
uv run --no-sync python scripts/debutanizer.py predict \
  --checkpoint /path/to/best.pt --data /path/to/debutanizer.csv \
  --output outputs/debutanizer/mac_predictions --device auto --batch-size 4
```

process 模式允许所有 y 留空；causal13 仍要求对应历史 y 可用。CSV 需保留 y 表头。
默认从样本索引 4 开始预测；可用 `--start-row` / `--stop-row` 指定区间（后者不含端点）。
需要预测单个新时刻时，应提供至少 5 行、最好包含完整上下文的连续观测，并指定最后一行。
索引指当前输入 CSV 的本地行号，不自动延续其他文件的时间索引。

## 未来验证

已新增 `tests/test_debutanizer.py`，覆盖特征因果性、时间区间、补齐、缺失 y、统计量、
冻结/微调梯度和自包含检查点。根据本次要求，这些测试尚未运行。
后续授权运行时，先通过安装命令的 `--group dev` 选项安装 pytest，
再执行 `uv run --no-sync python -m pytest tests/test_debutanizer.py`，
再在 Linux 上做小规模训练验证。当前没有 RMSE、运行耗时或实际设备兼容性的新增实测结论。
