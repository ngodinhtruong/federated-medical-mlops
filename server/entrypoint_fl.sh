#!/bin/sh

echo "Checking flwr version..."

python -c "import flwr; print('[SERVER] flwr before:', flwr.__version__)" || true

pip uninstall -y flwr || true
pip install --no-cache-dir flwr==1.7.0

python -c "import flwr; print('[SERVER] flwr after :', flwr.__version__)"

echo "Starting FL server..."

exec python /opt/fl/server.py
