# Gibbs Sampling — Yidong Shi

`gibbs.py` 实现团队 `experiment.py` 中的 `Sampler` 接口，使用仓库已有的 Python 环境和依赖。

在团队运行入口中导入：

```python
from alarm_inference.samplers.gibbs import gibbs
```

与 RS、LW 一起沿用仓库根目录 README 的公共入口：

先从 `CSE250A/` 运行 `PYTHONPATH=src .venv/bin/python`，再输入下面的 Python 代码。完整 shell 命令见[仓库 README](../../../README.md)。

```python
from alarm_inference.experiment import run_experiment
from alarm_inference.samplers.rs import rs
from alarm_inference.samplers.lw import lw
from alarm_inference.samplers.gibbs import gibbs

run_experiment("smoke", {"rs": rs, "lw": lw, "gibbs": gibbs})
```

第一个参数可选 `smoke`、`pilot` 或 `main`。每次调用运行所选阶段中配置的全部方法和实验组合。
重复运行需指定新的 `output_dir`。
主实验需先记录 pilot 决策，填写 `burn_in_sweeps`、`init_max_attempts`，并将状态改为 `ready`。

## 算法

固定证据，以第一个正权重 LW 完整赋值初始化。每轮按词典序拓扑顺序更新全部未观测节点，
局部条件概率正比于该节点的 CPT 与所有子节点 CPT 因子的乘积，使用本轮最新的其他节点取值。
因子在对数空间计算，不平滑零概率，保持 BIF 的父节点轴与状态顺序。

- N 为丢弃 burn-in 后保留的完整 sweep 数；每轮记录一次查询状态。主实验共 480 次：四个场景、六个 N、每组 20 次。
- 使用传入的随机数生成器。条件 CDF 按 Markov blanket 赋值缓存，每个节点最多 4096 项；每次调用重新建立缓存。
- 成功返回按 BIF 状态名标记的后验，未出现的状态概率为 `0.0`。
- 初始化达到尝试上限时返回 `gibbs_init_failed` 和空后验；这不代表证据概率必为零。
- `record_trace=True` 时返回长度为 N 的 `query_trace`，否则为 `None`；记录 trace 不改变随机流。
- 诊断包含初始化尝试数、初始对数权重、初末状态、更新顺序、burn-in、保留及完成轮数、变量更新次数。
- 非法输入抛出 `ValueError`；有效初始化后遇到无支持的局部条件分布会抛出 `RuntimeError`。

公共 runner 负责计时、计算 Exact 参考值与 TVD，并保存 `runs.jsonl` 和 `metadata.json`。
方法内部的准备、初始化、burn-in 和后验估计均计入耗时。

## 验证

`tests/test_gibbs.py` 检查手算后验、枚举得到的局部条件分布、证据固定、零概率、
初始化重试与失败、极小概率、burn-in、可复现性、输入校验和 ALARM 集成。
`tests/test_gibbs_long.py` 连续运行两条十万步 ALARM 链并核对状态、计数与重跑前缀。

```sh
.venv/bin/python -m pytest -q tests/test_gibbs.py tests/test_gibbs_long.py
.venv/bin/python -m pytest -q
```

集成测试直接调用现有 `run_experiment("smoke", ...)`，同时注册真实 RS、LW、Gibbs，
检查公共 runner 保存的种子、预算、后验和诊断。无需新增运行脚本或修改公共配置。

额外验证使用独立长链 pilot 设置：seed namespace 3、每链 100,000 步、零 burn-in 和 5000 次初始化上限；
这些是已完成验证的设置，不会覆盖仓库默认 pilot 配置。
已在 Ubuntu / Python 3.12.14、NumPy 2.5.4、pgmpy 1.1.2 下通过 51 项测试，
并用未修改的公共 runner 完成四个场景各五条十万步 Gibbs 链。

代码实现与检查使用了 AI 辅助。
