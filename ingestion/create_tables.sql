-- Bronze → Silver DDL
-- Tabela única: piade_telemetry (janelas de 1h do dataset PIADE sequences)

CREATE TABLE IF NOT EXISTS piade_telemetry (
    id                 SERIAL PRIMARY KEY,
    equipment_id       VARCHAR(20)   NOT NULL,
    interval_start     TIMESTAMPTZ   NOT NULL,
    count_sum          FLOAT,
    num_changes        FLOAT,
    pct_idle           FLOAT,
    pct_production     FLOAT,
    pct_downtime       FLOAT,
    pct_perf_loss      FLOAT,
    pct_sched_downtime FLOAT,
    ingested_at        TIMESTAMPTZ   DEFAULT NOW(),
    CONSTRAINT uq_piade_telemetry UNIQUE (equipment_id, interval_start)
);

CREATE INDEX IF NOT EXISTS idx_piade_equipment_id    ON piade_telemetry (equipment_id);
CREATE INDEX IF NOT EXISTS idx_piade_interval_start  ON piade_telemetry (interval_start);
CREATE INDEX IF NOT EXISTS idx_piade_pct_downtime    ON piade_telemetry (pct_downtime);
