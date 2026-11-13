# Likelihood Weighting — Junyi Xu

`lw.py` 实现团队 `experiment.py` 中的 `Sampler` 接口，使用仓库已有的 Python 环境和依赖。

在团队运行入口中导入：

```python
from alarm_inference.samplers.lw import lw
```

RS 和 Gibbs 实现也已导入后，沿用仓库根目录 README 的公共入口：

```python
from alarm_inference.experiment import run_experiment

run_experiment("main", {"rs": rs, "lw": lw, "gibbs": gibbs})
```

第一个参数可选 `smoke`、`pilot` 或 `main`。每次调用运行所选阶段中配置的全部方法和实验组合。
主实验中 LW 共运行 480 次：四个场景、六个 N、每组 20 次。N 为生成样本数。
团队主实验需先完成 Gibbs 的 pilot 决策，填写相应设置，并将主实验状态改为 `ready`。

LW 返回按 BIF 状态名标记的后验、状态和 ESS 等权重诊断；全零权重时返回 `zero_total_weight` 和空后验。
公共 runner 负责计时、计算 Exact 参考值与 TVD，并保存 `runs.jsonl` 和 `metadata.json`。
实现使用传入的随机数生成器、8192 个样本的固定分块和对数权重，不平滑模型中的零概率。

已使用未修改的公共 runner 验证真实 LW 调用。代码实现与检查使用了 AI 辅助。

