#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/advanced-control-mpl"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/advanced-control-cache"
matlab -batch "addpath('matlab'); run_local_suite; run_local_suite(0.001,'local_suite_dt1ms',11);"
.venv/bin/python python/build_cartesian_reference.py
.venv/bin/python python/run_panda_local.py --video
.venv/bin/python python/run_panda_local.py --cartesian
.venv/bin/python python/verify_local.py
.venv/bin/python python/analyze_local.py
