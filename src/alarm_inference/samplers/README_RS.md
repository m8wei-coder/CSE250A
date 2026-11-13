# Rejection Sampling — Sai Leng

`rs.py` 实现团队 `experiment.py` 中的 `Sampler` 接口，使用仓库已有的 Python 环境和依赖。

在团队运行入口中导入：

```python
from alarm_inference.samplers.rs import rs
```

与 LW、Gibbs 一起沿用仓库根目录 README 的公共入口：

```python
from alarm_inference.experiment import run_experiment

run_experiment("main", {"rs": rs, "lw": lw, "gibbs": gibbs})
```

## 算法

按词典序拓扑顺序对全部 37 个节点做前向（祖先）采样，证据节点也正常采样而不固定取值；只保留与证据完全一致的样本，用其中查询变量各状态的频率作为后验。
不剪枝无关节点，也不在证据不符时提前终止，使每个生成样本的计算量与 LW 一致，运行时间可以公平比较。

- N 为生成样本数（不是接受数）。主实验中 RS 共运行 480 次：四个场景、六个 N、每组 20 次。
- 成功时返回按 BIF 状态名标记的后验，未出现的查询状态概率为 `0.0`。
- 没有任何样本被接受时返回 `zero_accepts` 和空后验，这是预期的采样失败，由 runner 计入失败率。
- 诊断字段：`num_samples`、`accepted`、`rejected`、`acceptance_rate`、`chunk_size`、`scan_order`。
- 使用传入的随机数生成器和 8192 个样本的固定分块；改变 `chunk_size` 会改变随机流的消耗顺序。
- 不返回 `query_trace`，非法输入抛出 `ValueError`。

## 验证

`tests/test_rs.py` 在玩具网络上检查后验、接受率、零概率状态、`zero_accepts`、可复现性和输入校验，并在 ALARM smoke 场景上与 VE 对比。

```sh
.venv/bin/python -m pytest -q tests/test_rs.py
```

额外用未修改的公共 runner 跑通 smoke 与 pilot 阶段。每个正式场景生成 3,000,000 个样本时，接受率与精确 P(e) 的偏差都在 1σ 以内，TVD 为 0.0003–0.0070。
`lvedvolume_rare` 在 N=100 和 N=1000 下各跑 2000 次，零接受比例分别为 0.8315 和 0.1625，理论值 \((1-P(e))^N\) 为 0.8345 和 0.1637。

代码实现与检查使用了 AI 辅助。
