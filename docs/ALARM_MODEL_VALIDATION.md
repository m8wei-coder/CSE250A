# ALARM Model Validation Report

## Source Data Validation

The loader verifies the SHA-256 checksum of [`data/alarm.bif`](../data/alarm.bif) before passing the same bytes to pgmpy. The checksum matches the value in the README.

```text
701e6c561f71b55669070c29614f0724b761289aa2c4a35bcc97b638ee881fa2
```

The network is acyclic and contains 37 nodes and 46 directed edges. Each node has one conditional probability table (CPT). Together, the tables contain 243 parent assignments and 752 probability values. The CPT dimensions, parent variables, and state labels are consistent. Every probability is finite and lies in `[0, 1]`. For each parent assignment, the probabilities sum to within `1e-6` of `1`.

To check the parent-axis mapping, the test looks up `LVEDVOLUME=HIGH` by label. The probability is `0.90` for `HYPOVOLEMIA=TRUE, LVFAILURE=FALSE` and `0.01` when those two parent values are reversed. Both values match the BIF.

## Rounding Corrections

`HREKG` and `HRSAT` each have three parent assignments with rounded probabilities. For each assignment, the BIF lists three probabilities of `0.3333333`, which sum to `0.9999999` in decimal arithmetic.

| CPT | `ERRCAUTER` | `HR` | Original sum | Corrected sum |
| --- | --- | --- | --- | --- |
| `HREKG` | `TRUE` | `LOW` | `0.9999999` | `1.0` |
| `HREKG` | `FALSE` | `LOW` | `0.9999999` | `1.0` |
| `HREKG` | `TRUE` | `NORMAL` | `0.9999999` | `1.0` |
| `HRSAT` | `TRUE` | `LOW` | `0.9999999` | `1.0` |
| `HRSAT` | `FALSE` | `LOW` | `0.9999999` | `1.0` |
| `HRSAT` | `TRUE` | `NORMAL` | `0.9999999` | `1.0` |

After validation, the loader uses the reported parent labels to locate the six assignments and divides each probability by its original sum in memory. The other 237 assignments retain exactly the probabilities read from the BIF. The loader returns the corrected model for variable elimination and sampling, along with the report of the original sums. The BIF checksum remains the same.

## Zero-Probability Entries

All five exact zeros occur in `PVSAT`. Their positions are the same before and after correction and are recorded for Gibbs sampling checks.

| `FIO2` | `VENTALV` | `PVSAT` states with zero probability |
| --- | --- | --- |
| `LOW` | `ZERO` | `NORMAL`, `HIGH` |
| `LOW` | `NORMAL` | `NORMAL`, `HIGH` |
| `NORMAL` | `ZERO` | `HIGH` |

## Reproduction

Run the following command from the `CSE250A/` repository root. Validation used Python `3.12.16` and pgmpy `1.1.2`.

```sh
.venv/bin/python -m pytest -q
```

The tests use the ALARM BIF to verify the label mapping, validation report, and six corrected assignments. They compare all other CPT values exactly against the values read directly from the BIF.
