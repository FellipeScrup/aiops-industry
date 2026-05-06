-- Bronze → Silver DDL
-- Tabelas: logs (camada Silver normalizada) e alarm_dictionary (contribuição TCC)

CREATE TABLE IF NOT EXISTS logs (
    id            SERIAL PRIMARY KEY,
    source        VARCHAR(10)  NOT NULL,
    machine_id    VARCHAR(20)  NOT NULL,
    alarm_code    VARCHAR(20)  NOT NULL,
    event_type    VARCHAR(30),
    duration_ms   BIGINT,
    raw_timestamp TIMESTAMPTZ  NOT NULL,
    ingested_at   TIMESTAMPTZ  DEFAULT NOW(),
    CONSTRAINT uq_logs_dedup UNIQUE (source, machine_id, alarm_code, raw_timestamp)
);

CREATE INDEX IF NOT EXISTS idx_logs_source      ON logs (source);
CREATE INDEX IF NOT EXISTS idx_logs_machine_id  ON logs (machine_id);
CREATE INDEX IF NOT EXISTS idx_logs_alarm_code  ON logs (alarm_code);
CREATE INDEX IF NOT EXISTS idx_logs_timestamp   ON logs (raw_timestamp);

CREATE TABLE IF NOT EXISTS alarm_dictionary (
    id                 SERIAL PRIMARY KEY,
    alarm_code         VARCHAR(20) UNIQUE NOT NULL,
    source_dataset     VARCHAR(10),
    category           VARCHAR(50),
    severity           VARCHAR(20),
    title              VARCHAR(200),
    description        TEXT,
    probable_causes    TEXT[],
    corrective_actions TEXT[],
    created_at         TIMESTAMP DEFAULT NOW(),
    updated_at         TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alarm_dict_code     ON alarm_dictionary (alarm_code);
CREATE INDEX IF NOT EXISTS idx_alarm_dict_category ON alarm_dictionary (category);
CREATE INDEX IF NOT EXISTS idx_alarm_dict_severity ON alarm_dictionary (severity);
