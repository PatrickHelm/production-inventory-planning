#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: ./run_stochastic.sh {two_stage|multi_stage} N  (N: CPUs per setting)" >&2
    exit 2
}

if [[ $# -ne 2 ]]; then
    usage
fi

policy="$1"
case "$policy" in
    two_stage|multi_stage) ;;
    *) usage ;;
esac

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ab_run_stochastic "$policy" "$2"
