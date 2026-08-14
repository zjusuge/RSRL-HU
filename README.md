# RSRL-HU: One-Day-Ahead Landslide Velocity Forecasting

This repository provides the core Python implementation of **Robust State Residual Learning Informed by Hydrology with Prequential Uncertainty Quantification (RSRL-HU)** for one-day-ahead landslide velocity forecasting.

The workflow integrates robust causal motion-state estimation, following-day groundwater forecasting, a hydrology-informed dynamic velocity prior, shrinkage-controlled nonlinear residual learning, and rolling prequential prediction intervals.

Each monitoring day is treated as a forecast origin. All predictors are constructed exclusively from rainfall, groundwater-depth, and cumulative-displacement observations available at or before that forecast origin. The reference latent-velocity target is obtained only after the following-day displacement observation becomes available.

## Main Workflow

The executable workflow is implemented in:

```text
RSRL_HU.py
```

The workflow includes:

1. Validation of the complete daily monitoring record.
2. Robust causal displacement-velocity state estimation.
3. Construction of motion-state and state-uncertainty predictors.
4. Construction of antecedent rainfall and groundwater predictors.
5. Following-day groundwater-depth-change forecasting.
6. Forward cross-fitting of development-period groundwater predictions.
7. Temporally external selection of groundwater persistence shrinkage.
8. Estimation of a hydrology-informed dynamic velocity prior.
9. Nonlinear residual learning with residual shrinkage.
10. Nested temporal selection of residual-model complexity.
11. Four rolling-origin external evaluation blocks.
12. One prespecified late-period validation block.
13. Rolling prequential empirical prediction intervals.
14. An optional operational forecast for the day following the final observation.

## Repository Contents

The repository contains:

```text
RSRL-HU/
|-- README.md
|-- RSRL_HU.py
|-- requirements.txt
`-- data.xlsx
```

The repository intentionally excludes generated figures, serialized models, intermediate tables, supplementary experiments, and benchmark-model implementations.

All calculations are completed in memory. Numerical results are printed directly to the terminal, and the main script does not create output files.

## Dataset

The default input workbook is:

```text
data.xlsx
```

The workbook must be located in the same directory as `RSRL_HU.py` unless an alternative path is supplied using the `--data` argument.

The monitoring dataset contains **720 consecutive daily observations** covering the period from **12 January 2016 to 31 December 2017**.

The required input columns are:

| Column      | Description                                                 | Unit         |
| ----------- | ----------------------------------------------------------- | ------------ |
| `Date`      | Monitoring date                                             | `YYYY-MM-DD` |
| `RL (mm/d)` | Daily rainfall                                              | mm day^-1    |
| `GL (m)`    | Groundwater depth measured downward from the ground surface | m            |
| `SD (mm)`   | Cumulative GNSS surface displacement                        | mm           |

The input record must contain:

- no missing observations;
- no duplicated dates;
- no temporal discontinuities;
- no negative rainfall values; and
- no non-finite numerical values.

Groundwater depth is measured downward from the ground surface. Therefore, a smaller groundwater-depth value represents a shallower water table.

## Forecasting Task

For a forecast issued at origin day \(t\), RSRL-HU predicts the reference latent landslide velocity for day \(t+1\).

The following-day observation is used only after the forecast has been issued. It is used for:

- construction of the reference latent-velocity target;
- external forecast evaluation; and
- prequential forecast-error updating.

It is not used to construct predictors for the forecast issued at day \(t\).

## Chronological Information Boundary

The implementation is designed to preserve temporal causality.

At forecast origin \(t\), every predictor is constructed exclusively from observations available at or before day \(t\).

The chronological information boundary is maintained through:

1. strictly causal motion-state estimation;
2. backward-looking rolling rainfall and groundwater features;
3. forward cross-fitting of development-period groundwater predictions;
4. development-period-only model and hyperparameter selection;
5. frozen external calibration and test predictions; and
6. prequential updating of prediction-interval error histories.

External calibration and test observations are not used to construct predictors or fit the corresponding frozen forecasting model.

## Robust Causal State Estimation

The cumulative-displacement and velocity states are estimated using a causal two-state transition model.

Student-t innovation weighting is used to reduce the influence of atypical displacement observations.

The latent state contains:

- cumulative displacement; and
- daily landslide velocity.

The state-estimation procedure also provides:

- prior velocity;
- displacement innovation;
- standardized innovation;
- robust observation weight;
- observation outlier score;
- latent displacement variance; and
- latent velocity variance.

These variables provide both motion-state information and an explicit representation of state-estimation uncertainty.

## Following-Day Groundwater Forecasting

Following-day groundwater-depth change is estimated using a Huber gradient boosting regressor.

The groundwater predictor set includes:

- current groundwater depth;
- recent groundwater rise;
- rolling groundwater statistics;
- cumulative rainfall over multiple antecedent windows;
- maximum recent rainfall;
- antecedent precipitation indices;
- wet-day counts; and
- seasonal calendar variables.

Development-period groundwater predictions used by downstream components are generated through forward cross-fitting.

Each cross-fitted prediction block is produced by a model trained only on observations preceding that block.

A persistence-shrinkage coefficient is selected using temporally external development-period predictions:

```text
shrunk groundwater forecast
    = persistence forecast
    + groundwater shrinkage
    * (raw model forecast - persistence forecast)
```

This procedure controls potentially unstable following-day groundwater changes while preserving informative hydrological variation.

## Hydrology-Informed Dynamic Prior

The dynamic velocity prior combines:

- persistent latent velocity;
- a bounded daily drift term; and
- conditional groundwater forcing.

The general prior structure is:

```text
dynamic velocity prior
    = velocity persistence
    + bounded drift
    + activated groundwater contribution
```

Groundwater forcing becomes active when the predicted groundwater depth crosses a prespecified activation depth.

The effective groundwater activation head is transformed into a bounded nonlinear forcing term.

Dynamic-prior parameters are estimated using bounded robust nonlinear least squares and temporally external groundwater predictions.

The fitted contribution of the explicit groundwater-forcing term may vary among temporal evaluation blocks.

## Residual Learning

The final one-day-ahead velocity forecast is decomposed into a hydrology-informed dynamic prior and a nonlinear residual correction:

```text
final velocity forecast
    = dynamic velocity prior
    + residual shrinkage * predicted residual
```

The residual model uses a compact Huber gradient boosting regressor.

Residual predictors include:

- the hydrology-informed velocity prior;
- current latent velocity;
- recent motion dynamics;
- state-estimation uncertainty;
- groundwater-history variables;
- predicted following-day groundwater depth;
- predicted groundwater rise;
- groundwater activation head; and
- normalized groundwater forcing.

Residual-model complexity and residual shrinkage are selected through nested temporal validation within each external development period.

## Prequential Prediction Intervals

RSRL-HU constructs symmetric empirical prediction intervals from absolute historical forecast errors.

Before each external prediction interval is issued, the interval half-width is estimated using only:

- errors from the initial calibration period; and
- errors from preceding external forecasts.

The current external forecast error is added to the error history only after the corresponding prediction interval has been issued.

This ordering preserves the prequential information boundary.

The default nominal prediction-interval coverage is:

```text
90%
```

The interval is calculated as:

```text
lower bound = point forecast - prequential quantile
upper bound = point forecast + prequential quantile
```

Because the interval is symmetric and empirical, its lower velocity bound may be negative when the predicted velocity is close to zero.

## Temporal Evaluation Design

The primary evaluation consists of four rolling-origin blocks and one prespecified late-period validation block.

| Evaluation                | Fold | Development period       | Calibration period       | External target period   | External targets |
| ------------------------- | ---: | ------------------------ | ------------------------ | ------------------------ | ---------------: |
| Rolling-origin evaluation |    1 | 2016-02-02 to 2017-02-18 | 2017-02-19 to 2017-04-04 | 2017-04-06 to 2017-05-20 |               45 |
| Rolling-origin evaluation |    2 | 2016-02-02 to 2017-04-04 | 2017-04-05 to 2017-05-19 | 2017-05-21 to 2017-07-04 |               45 |
| Rolling-origin evaluation |    3 | 2016-02-02 to 2017-05-19 | 2017-05-20 to 2017-07-03 | 2017-07-05 to 2017-08-18 |               45 |
| Rolling-origin evaluation |    4 | 2016-02-02 to 2017-07-03 | 2017-07-04 to 2017-08-17 | 2017-08-19 to 2017-10-02 |               45 |
| Late-period validation    |    1 | 2016-02-02 to 2017-08-18 | 2017-08-19 to 2017-10-02 | 2017-10-03 to 2017-12-31 |               90 |

The rolling-origin experiment contains:

```text
180 external forecasts
```

The late-period validation contains:

```text
90 external forecasts
```

## Evaluation Metrics

Point-forecast performance is evaluated using:

- root mean squared error;
- mean absolute error;
- coefficient of determination; and
- mean signed bias.

Following-day groundwater performance is evaluated using:

- root mean squared error;
- mean absolute error;
- coefficient of determination;
- mean signed bias;
- RMSE skill relative to groundwater persistence; and
- MAE skill relative to groundwater persistence.

Prediction-interval performance is evaluated using:

- prediction interval coverage probability;
- mean prediction interval width;
- median prediction interval width; and
- Winkler interval score.

## Installation

Python 3.12 is recommended for reproducing the tested execution environment.

### Option 1: Using pip and venv

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Option 2: Using Conda

```bash
conda create -n rsrl-hu python=3.12 -y
conda activate rsrl-hu
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Tested Software Environment

The workflow was successfully executed using:

| Component        | Version    |
| ---------------- | ---------- |
| Operating system | Windows 11 |
| Python           | 3.12.11    |
| NumPy            | 2.2.6      |
| pandas           | 2.3.2      |
| SciPy            | 1.16.1     |
| scikit-learn     | 1.7.1      |
| openpyxl         | 3.1.5      |

Minor numerical differences may occur across operating systems, processor architectures, Python versions, and numerical-library builds.

## Quick Start

Place `data.xlsx` in the repository root and run:

```bash
python RSRL_HU.py
```

The script automatically:

1. validates the monitoring workbook;
2. constructs the causal motion-state representation;
3. constructs rainfall and groundwater predictors;
4. performs nested temporal model selection;
5. evaluates the four rolling-origin blocks;
6. evaluates the late-period validation block;
7. calculates velocity forecast metrics;
8. calculates groundwater forecast metrics;
9. calculates prediction-interval metrics; and
10. generates the optional operational forecast.

## Using an Alternative Workbook Path

An alternative workbook can be supplied using:

```bash
python RSRL_HU.py --data /path/to/data.xlsx
```

On Windows, a quoted absolute path can be used:

```powershell
python RSRL_HU.py --data "D:\path\to\data.xlsx"
```

The alternative workbook must preserve the required column names, chronological structure, observation period, and record length expected by the implementation.

## Omitting the Operational Forecast

To perform only the primary temporal evaluation, run:

```bash
python RSRL_HU.py --skip-operational
```

## Command-Line Arguments

| Argument             |     Default | Description                                                  |
| -------------------- | ----------: | ------------------------------------------------------------ |
| `--data`             | `data.xlsx` | Path to the input monitoring workbook                        |
| `--skip-operational` |     `false` | Skip the operational forecast following the final observation |

Command-line help can be displayed using:

```bash
python RSRL_HU.py --help
```

## Terminal Output

The program prints:

- the RSRL-HU software version;
- the detected runtime environment;
- the monitoring-data summary;
- the external temporal partitions;
- selected residual configurations;
- selected residual shrinkage coefficients;
- selected groundwater shrinkage coefficients;
- fitted dynamic-prior parameters;
- rolling-origin velocity metrics;
- late-period velocity metrics;
- groundwater forecast metrics;
- prediction-interval metrics; and
- the optional operational following-day forecast.

The implementation does not create figures or result files.

## Reference Execution

A successful reference execution produced the following latent-velocity results:

| Evaluation                |    N | RMSE (mm/day) | MAE (mm/day) |     R² | Bias (mm/day) |   PICP | MPIW (mm/day) |
| ------------------------- | ---: | ------------: | -----------: | -----: | ------------: | -----: | ------------: |
| Rolling-origin evaluation |  180 |      0.009120 |     0.007249 | 0.8045 |      0.002584 | 88.33% |      0.029428 |
| Late-period validation    |   90 |      0.004317 |     0.003065 | 0.6933 |      0.000670 | 98.89% |      0.022670 |

The following-day groundwater model produced:

| Evaluation                | RMSE (m) |  MAE (m) |     R² | RMSE skill relative to persistence | MAE skill relative to persistence |
| ------------------------- | -------: | -------: | -----: | ---------------------------------: | --------------------------------: |
| Rolling-origin evaluation | 0.165720 | 0.095449 | 0.6210 |                             38.44% |                            44.35% |
| Late-period validation    | 0.277758 | 0.111914 | 0.2796 |                             15.74% |                            39.19% |

These values are provided as a reference execution rather than a guarantee of bitwise-identical results across all software and hardware environments.

## Operational Forecast

When operational forecasting is enabled, the script generates a forecast for the day immediately following the final monitoring observation.

For the supplied 720-day dataset:

```text
Forecast origin: 2017-12-31
Forecast target: 2018-01-01
```

The operational output includes:

- current latent velocity;
- raw following-day groundwater forecast;
- persistence-shrunk groundwater forecast;
- fitted hydrology-informed velocity prior;
- selected residual configuration;
- selected residual shrinkage;
- final one-day-ahead velocity forecast;
- 90% velocity prediction interval;
- predicted latent cumulative displacement;
- 90% displacement prediction interval; and
- prequential interval half-width.

The operational forecast is conditional on the supplied monitoring record and should not be interpreted as an independent external validation result.

## Reproducibility Notes

The workflow is designed to preserve chronological causality.

- State estimation is performed causally.
- Predictors are constructed only from current and preceding observations.
- Following-day targets are not used as same-day predictors.
- Development-period groundwater predictions are generated through forward cross-fitting.
- Hyperparameter selection is performed within the corresponding development period.
- External calibration and test observations are not used to fit the frozen external forecasting model.
- Prediction-interval widths are updated prequentially.
- Fixed random seeds are used for stochastic estimators.

For the closest possible numerical reproduction, retain:

- the dependency versions in `requirements.txt`;
- the original chronological observation order;
- the predefined temporal partitions;
- the model constants;
- the feature definitions; and
- the random seed.

## Scope of the Repository

`RSRL_HU.py` implements the principal forecasting framework and its primary temporal evaluation.

The repository does not include:

- graphical visualization;
- automatic output-file generation;
- benchmark-model comparison;
- deep-learning comparison;
- moving-block bootstrap analysis;
- permutation analysis;
- feature-importance analysis;
- ablation experiments;
- supplementary sensitivity experiments; or
- serialized model export.

These exclusions are intentional and keep the public implementation focused on the principal RSRL-HU forecasting workflow.

## Data Availability

The dataset used in this study, comprising 720 consecutive daily observations of rainfall, groundwater depth, and cumulative GNSS displacement at the Qili landslide from 12 January 2016 to 31 December 2017, is publicly available in Zenodo under the Creative Commons Attribution 4.0 International license (Version 1.0.0), doi:10.5281/zenodo.21916040.

## Contact

For questions concerning the scientific study or software implementation, please contact the author identified in the associated article.

```text
Tianlong Wang
Email: tianlong_wang@zju.edu.cn
```
