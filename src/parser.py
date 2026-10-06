import argparse


def get_ablation_kwargs(args):
    return {
        "stationary_demand": args.stationary_demand,
        "higher_backorder_costs": args.higher_backorder_costs,
        "lower_capacities": args.lower_capacities,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Optimization Policy")
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=1000,
        help="Number of trajectories to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["multi_stage", "two_stage", "deterministic", "dynamic_global", "dynamic_item", "static_global", "static_item", ""],
        default="dynamic_item",
        help="Policy type",
    )
    parser.add_argument(
        "--hedging",
        type=float,
        default=0.3,
        help="Hedging parameter",
    )
    parser.add_argument(
        "--num_scenarios",
        type=int,
        default=1000,
        help="Number of scenarios",
    )
    parser.add_argument(
        "--planning_horizon",
        type=int,
        default=None,
        help="Planning horizon length for optimization policies; None keeps the policy default",
    )
    parser.add_argument(
        "--scenario_tree",
        type=str,
        default=None,
        help="Branching structure of the multi-stage scenario tree as dash-separated "
             "integers, e.g. 6-4-3-2-2; None keeps the policy default",
    )
    parser.add_argument(
        "--scaling",
        type=int,
        default=1,
        help="Scaling factor",
    )
    parser.add_argument(
        "--job_id",
        type=int,
        default=0,
        help="Job ID",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of jobs",
    )
    parser.add_argument(
        "--round",
        action="store_true",
        help="Use LP + rounding in optimization",
    )
    parser.add_argument(
        "--stationary_demand",
        action="store_true",
        help="Use stationary aggregate demand with variance 13.5",
    )
    parser.add_argument(
        "--higher_backorder_costs",
        action="store_true",
        help="Set the backorder critical ratio to 0.95",
    )
    parser.add_argument(
        "--lower_capacities",
        action="store_true",
        help="Subtract one from every capacity",
    )
    return parser.parse_args()
