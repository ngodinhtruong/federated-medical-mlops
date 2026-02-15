#!/bin/bash
set -e

echo " Client Airflow starting..."

echo " Init / migrate DB"
airflow db migrate || true

echo " Create user (if not exists)"
airflow users create \
  --username ${AF_USER:-changeme} \
  --password ${AF_PASS:-changeme} \
  --firstname Client \
  --lastname Node \
  --role Admin \
  --email client@example.com || true

echo " ENV:"
echo "CLIENT_ID=$CLIENT_ID"
echo "CLIENT_SEED=$CLIENT_SEED"
echo "FL_SERVER_ADDRESS=$FL_SERVER_ADDRESS"

echo " Start scheduler"
airflow scheduler &

echo " Start webserver"
exec airflow webserver
