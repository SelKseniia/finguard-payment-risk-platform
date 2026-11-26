-- ============================================================================
-- FinGuard Payment Risk & Margin Intelligence: ClickHouse DWH Schema
-- ============================================================================

CREATE DATABASE IF NOT EXISTS finguard;

-- 1. Справочник мерчантов (ReplacingMergeTree для версионирования тарифов и статусов)
CREATE TABLE IF NOT EXISTS finguard.dim_merchants
(
    merchant_id       LowCardinality(String),
    merchant_name     String,
    country           LowCardinality(FixedString(2)),
    region            LowCardinality(String),
    mcc_category      LowCardinality(String),
    contract_fee_pct  Decimal(6, 4),
    ip_subnet         String,
    anomaly_tag       LowCardinality(String),
    updated_at        DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (merchant_id);

-- 2. Таблица фактов транзакций (партиционирование по месяцам, оптимальный ключ сортировки)
CREATE TABLE IF NOT EXISTS finguard.fact_transactions
(
    transaction_id      UInt64,
    event_time          DateTime,
    merchant_id         LowCardinality(String),
    country             LowCardinality(FixedString(2)),
    region              LowCardinality(String),
    mcc_category        LowCardinality(String),
    amount_eur          Decimal(12, 2),
    card4               LowCardinality(Nullable(String)),
    card6               LowCardinality(Nullable(String)),
    device_type         LowCardinality(Nullable(String)),
    device_info         Nullable(String),
    ip_address          String,
    merchant_fee_eur    Decimal(10, 2),
    interchange_fee_eur Decimal(10, 2),
    gateway_cost_eur    Decimal(10, 2),
    is_fraud            UInt8,
    is_chargeback       UInt8,
    chargeback_loss_eur Decimal(12, 2),
    net_margin_eur      Decimal(12, 2)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_time)
ORDER BY (region, country, merchant_id, event_time, transaction_id)
SETTINGS index_granularity = 8192;

-- 3. Аналитическая витрина: Расчет динамического резерва (Rolling Reserve) и контроль маржи
CREATE OR REPLACE VIEW finguard.mart_merchant_risk_exposure AS
WITH merchant_stats AS (
    SELECT
        t.merchant_id                                                       AS merchant_id,
        any(m.merchant_name)                                                AS merchant_name,
        t.region                                                            AS region,
        t.country                                                           AS country,
        t.mcc_category                                                      AS mcc_category,
        any(m.contract_fee_pct)                                             AS contract_fee_pct,
        count()                                                             AS total_tx_count,
        sum(t.amount_eur)                                                   AS gross_tpv_eur,
        sum(t.net_margin_eur)                                               AS net_margin_eur,
        round(sum(t.net_margin_eur) / nullIf(sum(t.amount_eur), 0) * 100, 2) AS net_margin_pct,
        sum(t.chargeback_loss_eur)                                          AS total_cb_loss_eur,
        round(countIf(t.is_chargeback = 1) / count() * 100, 2)              AS cb_rate_pct,
        -- Метрика скорости роста потерь за последние 14 дней окна (Velocity Check)
        sumIf(t.chargeback_loss_eur, t.event_time >= (SELECT max(event_time) - INTERVAL 14 DAY FROM finguard.fact_transactions)) AS cb_loss_14d_eur,
        sumIf(t.amount_eur, t.event_time >= (SELECT max(event_time) - INTERVAL 14 DAY FROM finguard.fact_transactions))          AS tpv_14d_eur
    FROM finguard.fact_transactions AS t
    LEFT JOIN finguard.dim_merchants AS m ON t.merchant_id = m.merchant_id
    GROUP BY t.merchant_id, t.region, t.country, t.mcc_category
)
SELECT
    *,
    round(cb_loss_14d_eur / nullIf(tpv_14d_eur, 0) * 100, 2) AS cb_exposure_14d_pct,
    -- Алгоритм динамического резерва (Rolling Reserve %):
    -- Базовый холд 5% для MENA / High-Risk MCC, 2% для EU + надбавка за скорость чарджбэков
    multiIf(
        cb_rate_pct >= 5.0 OR cb_exposure_14d_pct >= 8.0, 20.0, -- Критический риск: 20% заморозка выплат
        cb_rate_pct >= 2.0 OR cb_exposure_14d_pct >= 3.5, 12.0, -- Повышенный риск: 12% заморозка
        region = 'MENA' OR mcc_category IN ('Travel & Airlines', 'Online Gaming'), 5.0,
        2.0                                                     -- Стандартный низкорисковый мерчант EU
    ) AS recommended_rolling_reserve_pct,
    -- Сумма средств в резерве (покрытие потенциальных диспутов)
    round(tpv_14d_eur * (recommended_rolling_reserve_pct / 100.0), 2) AS required_reserve_hold_eur,
    -- Флаг причины алерта для команды риск- и фин-аналитиков
    multiIf(
        net_margin_eur < 0 AND cb_rate_pct < 2.0, 'MARGIN_LEAKAGE_PRICING',
        cb_rate_pct >= 5.0,                       'HIGH_FRAUD_EXPOSURE',
        cb_exposure_14d_pct > cb_rate_pct * 1.5,  'VELOCITY_SPIKE_14D',
        'HEALTHY'
    ) AS risk_action_flag
FROM merchant_stats
ORDER BY net_margin_eur ASC;
