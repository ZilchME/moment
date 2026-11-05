# 使用 uv 管理研究环境

本项目以 `pyproject.toml` 为唯一依赖声明，以 `uv.lock` 固定完整版本和下载来源。
默认 Python 为 3.11，支持范围为 3.11–3.12；此次整理面向 Linux x86_64 与
Apple Silicon macOS。Intel Mac、Windows、Linux ARM 不在此次锁定的平台范围内。

## 安装

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)（至少 0.9.27），
再在仓库根目录执行以下对应命令。`uv sync` 会创建或同步项目的 `.venv`，
不需要手动激活。第一次安装可能下载 Python 和依赖，但不会下载 MOMENT 模型权重。

| 使用场景 | 安装命令 |
| --- | --- |
| Mac 推理，仅核心库 | `uv sync --locked` |
| Mac 数据处理与软测量开发 | `uv sync --locked --extra data` |
| Linux CPU 推理 | `uv sync --locked --extra cpu` |
| Linux NVIDIA CUDA 12.8 微调 | `uv sync --locked --extra cu128 --extra finetune` |
| Linux NVIDIA CUDA 12.8 多卡 DeepSpeed | `uv sync --locked --extra cu128 --extra distributed` |
| Notebook 教程（Mac） | `uv sync --locked --extra notebooks` |
| Notebook 教程（Linux GPU） | `uv sync --locked --extra cu128 --extra notebooks` |
| 开发验证（Mac） | `uv sync --locked --extra finetune --extra notebooks --group dev` |

额外依赖可以组合，例如同时增加 `--extra notebooks --extra finetune`。
`cpu` 和 `cu128` 互斥，不要使用 `--all-extras`。Mac 使用 PyPI 的原生 PyTorch，
不需要 CUDA、torchvision 或 torchaudio；`cu128` 只为 Linux 配置 CUDA。
Linux 不指定后端时使用 PyPI 的 PyTorch，可能安装 CUDA 相关包，因此建议显式选择
`cpu` 或 `cu128`。DeepSpeed 仅在 Linux x86_64 安装，Mac 不提供该功能。

CUDA 安装方式遵循 [uv 官方 PyTorch 文档](https://docs.astral.sh/uv/guides/integration/pytorch/)。
GPU 主机需有支持 CUDA 12.8 的 NVIDIA 驱动；可先查看 `nvidia-smi`。
普通预编译 PyTorch 不要求单独安装 CUDA Toolkit；DeepSpeed 的部分算子需要编译器、
CUDA Toolkit 和与 PyTorch 匹配的编译环境。仓库原有 `ds.yaml` 是双 GPU 配置，
使用前需按实际主机修改。安装依赖不会自动运行它。

## 依赖分组

| 分组 | 内容 |
| --- | --- |
| 核心 | PyTorch、Transformers、Hugging Face Hub、NumPy、Safetensors |
| `data` | pandas、SciPy、scikit-learn、joblib、tqdm |
| `notebooks` | `data` 加 Matplotlib、Jupyter、内核和 widgets |
| `finetune` | `data` 加 Accelerate、PEFT、wfdb（官方 PTB-XL 示例所需） |
| `distributed` | `finetune` 加 Linux DeepSpeed |
| `dev` 开发组 | pytest |

当前采用 PyTorch 2.8.0 / Transformers 4.55.x / PEFT 0.17.x 作为兼容基线，
避免不受控地升级模型相关接口。实际安装版本由锁文件决定。
PyTorch 自身支持普通微调，不使用 Accelerate/LoRA/PTB-XL 时可仅安装核心与 `data`。

`requirements.txt` 只保留 `pip install -r requirements.txt` 的兼容入口，
不再维护第二份版本列表；该入口不使用 uv 的平台后端配置和锁文件。
构建统一由 Hatchling 与 `pyproject.toml` 管理，不再使用旧 `setup.py`。

## 检查环境，不训练

安装完成后，用 `--no-sync` 保留刚才选择的依赖组和后端：

```bash
uv run --no-sync python scripts/check_environment.py
uv run --no-sync python scripts/check_environment.py --smoke-test --device cpu
```

第一条仅检查版本与设备；第二条创建一个单层、隐藏维度为 16 的随机模型，
用一条长度为 512 的序列做前向计算。不下载预训练模型，不计算反向传播，
不执行优化器更新，也不代表模型精度验证。

Apple Silicon 可额外验证 MPS；Linux GPU 可将设备改为 `cuda`：

```bash
uv run --no-sync python scripts/check_environment.py --smoke-test --device mps
```

安装开发组后执行离线检查：

```bash
uv run --no-sync python -m pytest
```

默认测试覆盖重构、特征提取、预测头、分类头和本地权重保存/加载，均使用微型模型。
原有 `tests/test_inference.py` 会下载多个预训练模型，已标记为 `pretrained`，
不包含在默认检查中。确需验证真实权重时，可显式运行
`uv run --no-sync python -m pytest -m pretrained tests/test_inference.py`。

## 跨平台预训练推理

下面是需要时才执行的示例。首次执行会下载 Small 权重；也可以将模型名改为本地权重目录。

```python
import torch
from momentfm import MOMENTPipeline
from momentfm.utils.device import resolve_device

device = resolve_device("auto")  # CUDA → MPS → CPU
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-small",
    model_kwargs={"task_name": "embedding"},
)
model.init()
model = model.eval().to(device=device, dtype=torch.float32)
x = torch.randn(1, 3, 512, device=device, dtype=torch.float32)

with torch.inference_mode():
    output = model(x_enc=x, reduction="none")
embeddings = output.embeddings.cpu()
```

Mac 推理优先使用 float32 和小批量。显式请求 `mps`/`cuda` 而设备不可用时会报错，
可用 `cpu` 重试；不会把这种失败当作训练或推理成功。
如 MPS 遇到不支持的算子，可使用 CPU，或在启动进程前设置
`PYTORCH_ENABLE_MPS_FALLBACK=1`，允许部分算子回退到 CPU。
原有 Notebook 中的硬编码 `cuda`/`cuda:1` 仍需替换为实际设备；本次没有批量重写教程。

## 更新与复现

将 `pyproject.toml`、`.python-version` 和 `uv.lock` 一起提交到版本控制。
Linux 与 Mac 共享锁文件，各自创建自己的 `.venv`，不要跨系统复制虚拟环境。

```bash
uv lock --check                 # 检查声明与锁文件一致
uv lock --upgrade-package peft  # 在声明范围内有意更新某个依赖
uv sync --locked --extra cu128 --extra finetune
```

升级前检查差异，并重新运行离线检查。需要不同 CUDA 版本时，应先修改后端来源与
对应 PyTorch 版本并重新锁定，避免手动安装后被下一次 `uv sync` 覆盖。

## 本次验证范围

已在 Apple M4 / Python 3.11.14 上安装锁定的 Notebook 与微调依赖，
通过 12 项离线检查，覆盖 CPU 与 MPS 上的各任务前向计算、本地权重往返加载及设备选择。
源码包和 wheel 均可构建。Linux CPU、CUDA 12.8、DeepSpeed 已通过跨平台安装计划检查，
尚未在实际 Linux GPU 主机上运行；Python 3.12 也尚未进行实际运行验证。
没有启动训练，没有下载或运行官方预训练权重，也没有验证完整 Notebook 的精度结果。
