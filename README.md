# Fixed 560-case ML sampling set

`final_sampling.py` creates a reproducible set of 560 design cases for the ML daylight workflow. It is separate from the existing manually configured CNN datasets. The earlier 50-case files remain in this folder as a workflow pilot and are not overwritten.

## Sampling logic

- A fixed random seed (`20260924`) makes every `model_001` to `model_560` reproducible.
- Fourteen unique façade patterns are used: four single-façade patterns, four adjacent-façade patterns, two opposite-façade patterns, and four three-façade patterns.
- Each façade pattern is combined with two window types and two shading conditions, giving 56 discrete design strata.
- Each stratum contains 10 LHS samples for window width, conventional-window height where applicable, and overhang depth where applicable.
- A floor-to-ceiling window has a 0.00 m sill height and a fixed 3.00 m height.
- A conventional window has a 0.90 m sill height and an LHS-sampled height of 0.60–2.10 m.
- The overhang length is equal to its associated window width, as defined in Section 4.2.
- All saved geometric dimensions are rounded to two decimal places (0.01 m).

The final dataset therefore has 560 cases. Every defined discrete design stratum appears exactly 10 times.

## Create the fixed case files

```powershell
python .\final_sampling.py
```

The script produces:

- `fixed_design_cases_560.csv` — a flat table for review and later ML preprocessing;
- `fixed_design_cases_560.json` — the same cases with window and overhang geometry for later simulation scripts.

After the files are created, the script refuses to overwrite them. This protects the meaning of every `model_XX` identifier. Only use `--overwrite` before starting simulations and only when you deliberately want a new fixed set.
