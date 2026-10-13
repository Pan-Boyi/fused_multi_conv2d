#!/bin/bash
# No-NPU graph-runner checks. --out-root and --case-dir select complete bound bundles.
HERE="$(cd "$(dirname "$0")" && pwd)"
exec "${PYTHON:-python3}" "$HERE/run_stub_test.py" "$@"
