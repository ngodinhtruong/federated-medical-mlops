#!/bin/bash
set -e

echo "Waiting for Postgres..."

until airflow db check; do
  sleep 2
done

echo "Postgres ready"

echo "Migrating DB"
airflow db migrate || airflow db init

echo "👤 Creating admin user (if not exists)"
airflow users create \
  --username ${AF_USER:-admin} \
  --password ${AF_PASS:-admin} \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com || true

echo "Starting scheduler..."
airflow scheduler &

echo "Starting webserver..."
exec airflow webserver
