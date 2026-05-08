#!/bin/bash
set -e

cd "$(dirname "$0")"

python -m weather.main
python -m EET.main
python -m electricity.main
python -m exchange_rate.main
python -m ILI.main
python -m traffic.main
