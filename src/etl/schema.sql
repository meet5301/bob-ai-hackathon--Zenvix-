-- ============================================================
-- Grid Guard — Neon Postgres Schema
-- Team Maverick, IBM Bobathon U1
-- ============================================================

-- ------------------------------------------------------------
-- assets
-- Master registry of grid equipment being monitored.
-- Every other table references asset_id from here.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS assets (
    asset_id        TEXT        PRIMARY KEY,
    asset_type      TEXT        NOT NULL,           -- transformer | substation | breaker
    region          TEXT        NOT NULL,
    lat             NUMERIC(9,6) NOT NULL,
    lon             NUMERIC(9,6) NOT NULL,
    install_year    SMALLINT    NOT NULL,
    customers_served INTEGER    NOT NULL,
    criticality_tier SMALLINT   NOT NULL            -- 1 (highest) .. 3 (lowest)
);

COMMENT ON TABLE assets IS
    'Master registry of monitored grid assets (transformers, substations, breakers).';

-- ------------------------------------------------------------
-- sensor_readings
-- Daily IoT/SCADA sensor snapshots per asset.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sensor_readings (
    id                  BIGSERIAL   PRIMARY KEY,
    asset_id            TEXT        NOT NULL
                            REFERENCES assets(asset_id) ON DELETE CASCADE,
    date                DATE        NOT NULL,
    temperature         NUMERIC(7,3),
    vibration           NUMERIC(7,3),
    oil_quality         NUMERIC(7,3),
    partial_discharge   NUMERIC(7,3),
    failed              SMALLINT    NOT NULL DEFAULT 0  -- 0 = healthy, 1 = failure event
);

COMMENT ON TABLE sensor_readings IS
    'Daily sensor measurements (temperature, vibration, oil quality, partial discharge) '
    'and a binary failure flag per asset.';

CREATE INDEX IF NOT EXISTS idx_sensor_asset_id ON sensor_readings(asset_id);
CREATE INDEX IF NOT EXISTS idx_sensor_date     ON sensor_readings(date);

-- ------------------------------------------------------------
-- weather
-- Daily regional weather pulled from Open-Meteo.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS weather (
    id                  BIGSERIAL   PRIMARY KEY,
    region              TEXT        NOT NULL,
    date                DATE        NOT NULL,
    temperature_max     NUMERIC(6,2),
    precipitation_sum   NUMERIC(7,2),
    windspeed_max       NUMERIC(6,2)
);

COMMENT ON TABLE weather IS
    'Daily regional weather data (max temperature, precipitation, max wind speed) '
    'sourced from the Open-Meteo archive API.';

CREATE INDEX IF NOT EXISTS idx_weather_region ON weather(region);
CREATE INDEX IF NOT EXISTS idx_weather_date   ON weather(date);

-- ------------------------------------------------------------
-- incidents
-- Recorded outage/failure incidents linked to an asset.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS incidents (
    incident_id         TEXT        PRIMARY KEY,
    asset_id            TEXT        NOT NULL
                            REFERENCES assets(asset_id) ON DELETE CASCADE,
    date                DATE        NOT NULL,
    cause               TEXT,
    duration_hours      NUMERIC(6,2),
    customers_affected  INTEGER
);

COMMENT ON TABLE incidents IS
    'Logged power-outage or equipment-failure incidents, including cause, '
    'duration, and number of customers affected.';

CREATE INDEX IF NOT EXISTS idx_incidents_asset_id ON incidents(asset_id);
CREATE INDEX IF NOT EXISTS idx_incidents_date     ON incidents(date);
