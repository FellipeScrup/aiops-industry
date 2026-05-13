-- Bronze → Silver DDL
-- Tabela: smartfactory_logs (eventos NDJSON do dataset Smart Factory Logs 14441997)

CREATE TABLE IF NOT EXISTS smartfactory_logs (
    id              VARCHAR(36) PRIMARY KEY,
    station         VARCHAR(20) NOT NULL,
    event_timestamp TIMESTAMP NOT NULL,
    current_state   VARCHAR(20) NOT NULL,
    current_task    TEXT,
    current_task_duration FLOAT,
    current_sub_task TEXT,
    sensors         JSONB,
    split           VARCHAR(10) NOT NULL  -- 'train' ou 'test'
);

CREATE INDEX IF NOT EXISTS idx_sf_station ON smartfactory_logs(station);
CREATE INDEX IF NOT EXISTS idx_sf_state   ON smartfactory_logs(current_state);
CREATE INDEX IF NOT EXISTS idx_sf_split   ON smartfactory_logs(split);
