#!/bin/bash
# Executado pelo postgres no primeiro boot (docker-entrypoint-initdb.d).
# Cria o database "mlflow" além do database principal definido em POSTGRES_DB.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE mlflow;
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO "$POSTGRES_USER";
EOSQL
