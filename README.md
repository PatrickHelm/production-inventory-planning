# Production-Inventory Planning in High-Tech Supply Chains: One Policy to Rule Them All?

Code for the paper

> van Dijck, T., Helm, P., Fleuren, T., Temizöz, T., & van Jaarsveld, W. (2026). *Production-Inventory Planning in High-Tech Supply Chains: One Policy to Rule Them All?* Available at SSRN: https://ssrn.com/abstract=6735067, doi:[10.2139/ssrn.6735067](http://dx.doi.org/10.2139/ssrn.6735067)

The problem, the case data and the simulation framework are described in the paper and in the repository of the research challenge: [Research Challenge: Production-Inventory Planning at ASML](https://github.com/tjumvandijck/Research-Challenge-Production-Inventory-Planning-at-ASML). This repository contains the seven policies compared in the paper, the scripts that tune and evaluate them, and the input data of the case.

## Citation

```bibtex
@misc{vanDijck2026,
  author = {van Dijck, Tjum and Helm, Patrick and Fleuren, Tijn and Temiz{\"o}z, Tarkan and van Jaarsveld, Willem},
  title  = {Production-Inventory Planning in High-Tech Supply Chains: One Policy to Rule Them All?},
  year   = {2026},
  month  = may,
  note   = {Available at SSRN: https://ssrn.com/abstract=6735067},
  doi    = {10.2139/ssrn.6735067},
  url    = {https://ssrn.com/abstract=6735067}
}
```

## Contents

- [Policies and settings](#policies-and-settings)
- [Setup](#setup)
- [Reproducing the experiments](#reproducing-the-experiments)
- [License](#license)

## Policies and settings

| name in the code | label in the paper | policy |
|---|---|---|
| `dynamic_item` | DBS–Item | dynamic base-stock policy with item-specific safety-stock factors |
| `static_item` | SBS–Item | static base-stock policy with item-specific safety-stock factors |
| `dynamic_global` | DBS–Global | dynamic base-stock policy with one global safety-stock factor |
| `static_global` | SBS–Global | static base-stock policy with one global safety-stock factor |
| `multi_stage` | MSS–PIP | rolling-horizon multi-stage stochastic program over a scenario tree (default tree `6-4-3-2-2`, then no branching) |
| `two_stage` | TSS–PIP | rolling-horizon two-stage stochastic program over demand scenarios (by default 1000) |
| `deterministic` | D–PIP | rolling-horizon deterministic integer program over a hedged point forecast |

The base-stock policies are in [src/base_stock.py](src/base_stock.py) and are tuned by [src/hedging_optimizer.py](src/hedging_optimizer.py). The optimization policies are in [src/optimization_policies.py](src/optimization_policies.py), and their Gurobi models in [src/optimization_models.py](src/optimization_models.py). In the paper, the stochastic programs use LP relaxation plus rounding (`--round`) and a planning horizon of 30 periods.

Every policy is evaluated in five settings. Each variant changes one aspect of the base case:

| setting | row in the paper's tables | flags | change |
|---|---|---|---|
| `scaling1` | Base case | `--scaling 1` | none |
| `scaling1_stationary_demand` | Stationary demand | `--scaling 1 --stationary_demand` | i.i.d. aggregate demand (AR coefficients 0, variance 13.5), with its own demand paths |
| `scaling1_higher_backorder_costs` | Higher backorder costs | `--scaling 1 --higher_backorder_costs` | backlog penalty 19 times the end product's holding cost (critical ratio 0.95) |
| `scaling1_lower_capacities` | Lower capacities | `--scaling 1 --lower_capacities` | every capacity minus one |
| `scaling3` | Less stochastic split | `--scaling 3` | gamma shape times 3 and scale divided by 3: the same expected product shares with less variance |

The setting name is also the name of its result folder, e.g. `results/base_stock/dynamic_item/scaling1_lower_capacities/`.

## Setup

The experiments run in Docker. The image contains Python 3.9 and the package versions the paper's results were computed with, pinned in [requirements.txt](requirements.txt).

```bash
docker build -t production-inventory-planning .
```

- `run.sh` starts containers from the image in `$IMAGE`, which defaults to `production-inventory-planning`.
- The code is not copied into the image: `run.sh` mounts the repository at `/mnt/mydata`. Code changes need no rebuild; changes to `requirements.txt` do.
- `run.sh` pins every container to specific host CPUs with `--cpuset-cpus`, so it needs a Linux host where your user can run `docker` without `sudo`.

**Without Docker:** use Python 3.9 and `pip install -r requirements.txt`. Run the scripts from the repository root with `python scripts/<script>.py [args] --n_jobs 1 --job_id 1`. `run.sh` normally appends these two arguments.

### Gurobi license

The optimization policies need [Gurobi](https://www.gurobi.com), which is free for academic use. The base-stock policies run without a license.

1. Paste your license into the `gurobi.lic` in the repository root. Because the repository is mounted into the container, no rebuild is needed.
2. Keep git from picking up your license:

   ```bash
   git update-index --skip-worktree gurobi.lic
   ```

`run.sh` hands `gurobi.lic` to Gurobi only when the file is not empty. The license has to work inside a container, e.g. a Web License Service (WLS) or token-server license. Named-user and single-machine licenses are tied to one host and generally don't work in a container. Without a license, Gurobi falls back to the size-limited license that ships with `gurobipy`, and the optimization policies stop with `Model too large for size-limited license`.

To check that the license works inside the container:

```bash
docker run --rm -v "$PWD":/mnt/mydata -e GRB_LICENSE_FILE=/mnt/mydata/gurobi.lic \
    production-inventory-planning python -c "import gurobipy as gp; gp.Model()"
```

## Reproducing the experiments

### Pipeline overview

Run every command from the repository root.

| step | command | needs |
|---|---|---|
| 1 | `./run_setup_base_stock.sh N` | nothing |
| 2 | `./run_deterministic.sh N` | step 1 |
| 2 | `./run_stochastic.sh two_stage N` | step 1 |
| 2 | `./run_stochastic.sh multi_stage N` | step 1 |
| 3 | `./run_ablation_optimization.sh [CPU_START] [NUM_JOBS]` | step 1 (scaling-1 files) |

The step-2 commands and the ablation don't depend on each other and can run at the same time on disjoint CPU ranges. Steps 1 and 2 run all five settings in parallel. The ablation uses the base case only.

All runs write into `results/`, and reruns overwrite earlier outputs: step 1 always reruns the base-stock evaluation, and the deterministic evaluation always writes to `results/optimization/deterministic/<setting>/`, whatever its arguments.

### N, CPU ranges and environment variables

`N` is the number of CPUs per setting. Setting `i` (0–4, in the order of the settings table) runs on CPUs `CPU_START + i*N` to `CPU_START + (i+1)*N - 1`, so each script uses `5*N` CPUs.

| variable | default | scripts | meaning |
|---|---|---|---|
| `CPU_START` | `0` | setup, deterministic, stochastic | first host CPU index |
| `NUM_TRAJECTORIES` | `1000` | setup, deterministic, stochastic | evaluation trajectories per setting |
| `LOG_DIR` | `logs/ablation/<script>-<timestamp>` | setup, deterministic, stochastic | log folder |
| `TUNING_TRAJECTORIES` | `1000` | setup, deterministic | trajectories per point of the global and the deterministic hedge grids |
| `INITIAL_STATE_TRAJECTORIES` | `NUM_TRAJECTORIES` | setup | test demand paths and initial states to generate, at least `NUM_TRAJECTORIES` |
| `NUM_SCENARIOS` | `1000` | stochastic | scenarios of the two-stage policy |
| `IMAGE` | `production-inventory-planning` | all | Docker image |

- `N` can't exceed `NUM_TRAJECTORIES`. Sharded evaluations give each of the `N` jobs `NUM_TRAJECTORIES // N` trajectories, so pick an `N` that divides `NUM_TRAJECTORIES`. Otherwise the leftover trajectories are silently skipped.
- Each job seeds the scenario sampling of the stochastic policies with its job id. To reproduce the paper's numbers exactly, use the job counts of the reference runs: `N=5` for the stochastic policies, `N=1` for the deterministic policy, and 5 jobs per ablation configuration.
- The deterministic and stochastic runs use a planning horizon of 30.
- Every script stays in the foreground until all settings are done, which can take days. Start long runs with `nohup`:

```bash
mkdir -p logs
CPU_START=0 nohup ./run_stochastic.sh two_stage 5 > logs/two_stage.out 2>&1 &
```

### Resources

The reference runs used a server with 2x Intel Xeon Platinum 8280L (56 cores, 112 threads) and about 6 TB RAM, with up to about 56 cores in use at the same time. All numbers below were measured there and are rough. The whole pipeline takes about 3,000 CPU-hours, most of it in the stochastic evaluations. Memory is not the bottleneck.

**Time** (one CPU per job, measured in the reference runs)

| run | rough cost |
|---|---|
| `run_setup_base_stock.sh` | about 6 CPU-hours; a base-stock evaluation without njit takes about 15 ms per trajectory |
| `run_deterministic.sh` | about 20 h per setting for the hedge grid, which runs serially whatever `N` is, plus about 3.5 s per trajectory for the evaluation |
| `run_stochastic.sh two_stage` | about 12 min per trajectory, about 200 CPU-hours per setting at 1000 trajectories |
| `run_stochastic.sh multi_stage` | about 15 min per trajectory, about 260 CPU-hours per setting |
| ablation | 3.5 min (two-stage, 500 scenarios) to 2 h (multi-stage MIP) per trajectory |

The wall time of a sharded evaluation is roughly `NUM_TRAJECTORIES / N` times the time per trajectory. With `N=5`, the two-stage run takes about 1.7 days and the multi-stage run about 2.1 days. With the ablation defaults, the multi-stage MIP takes about 40 h, the two-stage policy with 2000 scenarios and the tree `6-4-3-2-2-2` about 11 h each, and the other configurations less than 8 h.

### Step 1: base-stock setup

```bash
./run_setup_base_stock.sh 8
```

Runs four stages, each after the previous one has finished:

1. **Shared precomputations,** one container after another on the first `N` CPUs: rounding variance, demand-share variance and static demand distribution for scaling 1 and 3, and the static demand distribution for stationary demand. Output: `data/precomputed/`.
2. **Base-stock tuning per setting:** the global grid (`enumerate_global_hedging.py`), then the item-level tuning (`run_optimization.py`), which starts from the global result. The dynamic and static variants run side by side on half of the setting's CPUs each, or one after the other if `N=1`. Output in `results/base_stock/{dynamic,static}_{global,item}/<setting>/`: `hedging_parameters.txt`, plus `grid_costs.txt` (global) or `mean_history.txt` and `sigma_history.txt` (item).
3. **Test demand and initial states** (`precompute_initial_states.py`, warming up the tuned `dynamic_item` policy). Output: `data/demand_realizations_*.npy`, `data/demand_realizations_warmup_*.npy` and `data/initial_*.npy`. The scaling-1 baseline runs first, because the lower-capacity and higher-backorder settings reuse its demand paths.
4. **Base-stock evaluation** of `dynamic_item`, `static_item`, `dynamic_global` and `static_global`, one CPU each and at most `N` at a time per setting. Output in `results/base_stock/<policy>/<setting>/`: `all_costs.txt`, `costs_mean_std.txt`, `fill_rate.txt`, `holding_cost.txt`, `backorder_cost.txt` and `inventory_distribution.txt`.

Stages 1–3 skip a step when all its output files exist (the log says `Skipping ...: outputs exist`). A fresh clone contains none of these outputs, so all four stages run. Delete outputs to recompute them. The lower-capacity and higher-backorder initial states are the exception: they are rebuilt unless `data/initial_states_scaling1_<suffix>_baseline_demand.txt` shows they were generated from at least `INITIAL_STATE_TRAJECTORIES` baseline demand paths.

### Step 2: optimization policies

The step-2 scripts check the step-1 files first, and stop with a message pointing to `run_setup_base_stock.sh` if something is missing or out of date. Give concurrent runs disjoint CPU ranges, for example:

```bash
CPU_START=0  nohup ./run_deterministic.sh 1          > logs/deterministic.out 2>&1 &   # CPUs 0-4
CPU_START=5  nohup ./run_stochastic.sh multi_stage 5 > logs/multi_stage.out   2>&1 &   # CPUs 5-29
CPU_START=30 nohup ./run_stochastic.sh two_stage 5   > logs/two_stage.out     2>&1 &   # CPUs 30-54
```

**Deterministic.** Per setting, `run_deterministic.sh`:

1. Sweeps the hedge over `0.20, 0.21, ..., 0.32` with `enumerate_deterministic_hedging.py`, using `TUNING_TRAJECTORIES` trajectories per value. It writes the best value to `results/optimization/deterministic/<setting>/hedging_parameters.txt`, and the sweep to `all_costs.txt` in the same folder. This step is skipped when both files exist.
2. Evaluates the chosen hedge with `evaluate_optimization_policy.py`, split into `N` jobs, and writes the per-job files to the same folder.

The sweep runs in a single process, so `N` only speeds up the evaluation, and `N=1` is usually enough.

**Stochastic.** `run_stochastic.sh {two_stage|multi_stage} N` evaluates the policy with LP + rounding (`--round`) and horizon 30, split into `N` jobs per setting. The two-stage policy uses `NUM_SCENARIOS` scenarios. The multi-stage policy uses the default scenario tree `6-4-3-2-2`, with no branching in the remaining periods.

Output: `results/optimization/{two_stage,multi_stage}/<setting>_round_h30/`. A non-default `NUM_SCENARIOS` adds `_s<NUM_SCENARIOS>` to the folder name (two-stage only), and a `NUM_TRAJECTORIES` other than 1000 adds `_n<NUM_TRAJECTORIES>`.

The deterministic and stochastic evaluations write per-job files, with `<job>` from 1 to `N`: `all_costs_<job>.txt`, `costs_mean_std_<job>.txt`, `fill_rate_<job>.txt`, `holding_cost_<job>.txt`, `backorder_cost_<job>.txt` and `inventory_distribution_<job>.txt`.

### Step 3: ablation of the stochastic policies

```bash
./run_ablation_optimization.sh [CPU_START] [NUM_JOBS]    # defaults: 0 and 5
DRY_RUN=1 ./run_ablation_optimization.sh 0 5             # only print the commands
```

Twelve configurations on the base case. Each changes one factor relative to its policy's baseline (LP + rounding, horizon 30, 1000 scenarios or tree `6-4-3-2-2`):

| configuration | change | folder under `results/optimization/<policy>/` |
|---|---|---|
| `<policy>_baseline` | none | `scaling1_round_n100/` |
| `<policy>_mip` | MIP instead of LP + rounding | `scaling1_n100/` |
| `<policy>_h20` | horizon 20 | `scaling1_round_h20_n100/` |
| `<policy>_h40` | horizon 40 | `scaling1_round_h40_n100/` |
| `two_stage_s500`, `two_stage_s2000` | 500 or 2000 scenarios | `scaling1_round_s500_n100/`, `scaling1_round_s2000_n100/` |
| `multi_stage_tree6-4-3-2`, `multi_stage_tree6-4-3-2-2-2` | scenario tree `6-4-3-2` or `6-4-3-2-2-2` | `scaling1_round_tree6-4-3-2_n100/`, `scaling1_round_tree6-4-3-2-2-2_n100/` |

Each configuration evaluates 100 trajectories with `NUM_JOBS` single-CPU jobs. `./run_ablation_optimization.sh -h` lists the arguments and environment overrides.

### Running a single script with `run.sh`

All run scripts go through `run.sh`, which runs one script from `scripts/` in one or more containers. Use it directly for single steps or for other experiments:

```bash
./run.sh [wrapper options] SCRIPT [python args...]
```

`./run.sh -h` lists the wrapper options. Everything after the script name is passed to the Python script; its arguments are defined in [src/parser.py](src/parser.py).

For example, one ablation configuration on its own:

```bash
# -> results/optimization/multi_stage/scaling1_round_tree6-4-3-2_n100/
./run.sh -i 0 -j 5 evaluate_optimization_policy.py --policy multi_stage --scaling 1 --round --scenario_tree 6-4-3-2 --num_trajectories 100
```

## License

The code is released under the [MIT License](LICENSE). If you use it, please cite the paper (see [Citation](#citation)).
