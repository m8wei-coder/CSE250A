# Query and Evidence Selection

## Selection criteria

We selected two single-variable queries from ALARM. `LVEDVOLUME` asks what the observed cardiac output and stroke volume tell us about left ventricular end-diastolic volume. `VENTALV` asks what ventilation, pressure, and expired carbon dioxide readings tell us about alveolar ventilation. The two queries cover different parts of the network and yield different posterior shapes.

For each query, the common and rare cases observe the same variables and differ in exactly one observed state. We used exact evidence probabilities to screen for common cases with \(P(e)\geq 0.10\) and rare cases with \(0.0002\leq P(e)\leq 0.002\). We required each posterior to differ from its prior by at least `0.10` in total variation distance (TVD). Among qualifying pairs, we preferred similar posteriors so the two cases would answer the same inference question with comparable conclusions.

## Strategies considered

| Strategy | ALARM example | Advantage | Tradeoff |
| --- | --- | --- | --- |
| Change a single informative reading | Query `LVFAILURE` and change `HISTORY` from `FALSE` to `TRUE` | The pair is easy to explain and uses one observed variable. | The two readings give very different conclusions. The posterior probability of `LVFAILURE=TRUE` rises from `0.53%` to `82.57%`. The less common evidence still has probability `5.45%`. |
| Change agreement between two informative readings | Query `LVEDVOLUME`, keep `CVP=LOW`, and change `PCWP` from `LOW` to `HIGH` | The second assignment captures a less common combination of two related readings. | The most likely query state changes from `LOW` to `HIGH`. The posterior TVD between cases is about `0.7155`. |
| Add an observation to the rare case | Query `LVEDVOLUME` given `STROKEVOLUME=LOW` and add `CO=HIGH` in the rare case | Evidence probability falls from `18.08%` to `0.1808%` while the posterior stays the same. | The cases use different evidence variable sets. |
| Change a background reading while keeping the main observation | Query `LVEDVOLUME`, keep `STROKEVOLUME=LOW`, and change `ERRCAUTER` from `FALSE` to `TRUE` | Both cases use the same evidence variables and have the same posterior. | `ERRCAUTER` is less directly related to the cardiac volume question. The less common assignment still has probability `1.808%`. |
| Match the observations and compare exact posteriors | Keep the same evidence variables, change one state, and compare both the evidence probabilities and posteriors | The pair can span a large difference in evidence probability while preserving the inference question and keeping the posteriors close. | It takes more screening to find a pair that meets both conditions. |

## Selected cases

| Query | Case | Evidence | Exact \(P(e)\), rounded | TVD from prior |
| --- | --- | --- | --- | --- |
| `LVEDVOLUME` | Common | `STROKEVOLUME=LOW`, `CO=LOW` | `0.14973626` | `0.4612` |
| `LVEDVOLUME` | Rare | `STROKEVOLUME=LOW`, `CO=HIGH` | `0.00180800` | `0.4612` |
| `VENTALV` | Common | `VENTLUNG=ZERO`, `PRESS=LOW`, `EXPCO2=LOW` | `0.18650268` | `0.2507` |
| `VENTALV` | Rare | `VENTLUNG=ZERO`, `PRESS=LOW`, `EXPCO2=HIGH` | `0.00196229` | `0.2420` |

For `LVEDVOLUME`, the main observation is `STROKEVOLUME=LOW`. Stroke volume and left ventricular end-diastolic volume share the parents `HYPOVOLEMIA` and `LVFAILURE` in ALARM, so observing one informs the other. Given `STROKEVOLUME`, `CO` is d-separated from `LVEDVOLUME`. High cardiac output is common marginally, with probability about `64.3%`, but has probability `1%` given low stroke volume. Changing `CO` therefore makes the full evidence about 83 times less likely while leaving the posterior identical.

For `VENTALV`, both cases keep lung ventilation and pressure fixed. `EXPCO2` can provide information about `VENTALV` through `ARTCO2`. Changing `EXPCO2` from `LOW` to `HIGH` makes the full evidence about 95 times less likely. Both posteriors favor `VENTALV=ZERO`, and their TVD is `0.0099`. The posterior for this query is more concentrated than the one for `LVEDVOLUME`.

## Reference posteriors

| Query | Case | `ZERO` | `LOW` | `NORMAL` | `HIGH` |
| --- | --- | --- | --- | --- | --- |
| `LVEDVOLUME` | Common and rare | | `0.273230` | `0.240719` | `0.486051` |
| `VENTALV` | Common | `0.938345` | `0.012110` | `0.048430` | `0.001115` |
| `VENTALV` | Rare | `0.928607` | `0.011984` | `0.049409` | `0.010000` |

## Experimental Validation

We evaluated the four selected cases on the validated [ALARM model](ALARM_MODEL_VALIDATION.md) using the [exact inference implementation](../src/alarm_inference/exact.py). For each pair, we also computed the joint marginal over the query and all observed variables. We extracted the entries matching the evidence and summed over the query states to obtain \(P(e)\). Dividing those entries by their sum gave a second calculation of the posterior.

The two calculations agreed for all four cases. Every case met its evidence probability range and the minimum TVD from its prior. The table compares the common and rare cases within each query.

| Query | Common-to-rare evidence probability ratio | TVD between the two posteriors |
| --- | --- | --- |
| `LVEDVOLUME` | `82.82` | `0.000000` |
| `VENTALV` | `95.04` | `0.009864` |

### Reproduction

Run from the `CSE250A/` repository root with the project virtual environment.

```sh
PYTHONPATH=src .venv/bin/python - <<'PY'
from math import isclose

from pgmpy.inference import VariableElimination

from alarm_inference.exact import exact_inference
from alarm_inference.model import load_alarm_model

model, _ = load_alarm_model("data/alarm.bif")
ve = VariableElimination(model)
pairs = [
    ("LVEDVOLUME", {"STROKEVOLUME": "LOW"}, "CO"),
    ("VENTALV", {"VENTLUNG": "ZERO", "PRESS": "LOW"}, "EXPCO2"),
]

def tvd(left, right):
    return 0.5 * sum(abs(left[state] - right[state]) for state in left)

for query, fixed, changed in pairs:
    prior = exact_inference(model, query).posterior
    joint = ve.query([query, *fixed, changed], show_progress=False)
    results = []
    for label in ("LOW", "HIGH"):
        evidence = {**fixed, changed: label}
        result = exact_inference(model, query, evidence)
        masses = {state: joint.get_value(**{query: state, **evidence}) for state in prior}
        total = sum(masses.values())
        assert total > 0
        assert isclose(result.p_evidence, total, rel_tol=1e-12)
        assert tvd(result.posterior, {state: mass / total for state, mass in masses.items()}) < 1e-12
        assert tvd(prior, result.posterior) >= 0.10
        results.append(result)

    common, rare = results
    assert common.p_evidence >= 0.10
    assert 0.0002 <= rare.p_evidence <= 0.002
    ratio = common.p_evidence / rare.p_evidence
    print(f"{query}  P(e) ratio {ratio:.2f}  posterior TVD {tvd(common.posterior, rare.posterior):.6f}")
PY
```
