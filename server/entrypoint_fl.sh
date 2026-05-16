python -c "import flwr; print('[SERVER] flwr after :', flwr.__version__)"

echo "Preparing data..."
python /opt/fl/data/downloadData.py

echo "Starting FL server..."

exec python /opt/fl/server_flower.py
