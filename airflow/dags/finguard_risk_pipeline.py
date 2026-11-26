from datetime import datetime, timedelta
import logging
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException

default_args = {
    "owner": "kseniia.seliutina",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

DATA_PATH = "/opt/airflow/data/fact_transactions.csv"
OUTPUT_ALERTS_PATH = "/opt/airflow/data/daily_risk_alerts.csv"


def run_data_quality_checks(**context):
    """
    Data Quality Gate: блокирует пайплайн при нарушении целостности финансовых данных
    (дубли TransactionID, отрицательные суммы или аномальный % пропусков).
    """
    logging.info("Starting Data Quality checks on incoming payment batch...")
    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise AirflowException("DQ Failed: Incoming transaction batch is empty!")

    duplicate_count = df["TransactionID"].duplicated().sum()
    if duplicate_count > 0:
        raise AirflowException(f"DQ Failed: Found {duplicate_count} duplicate TransactionIDs!")

    if (df["amount_eur"] <= 0).any():
        raise AirflowException("DQ Failed: Detected non-positive transaction amounts!")

    logging.info(f"DQ Passed successfully. Validated {len(df):,} transactions.")


def compute_rolling_reserves_and_alerts(**context):
    """
    Рассчитывает риск-экспозицию мерчантов, утечку маржи и формирует очередь алертов
    для AI Risk Copilot и команды риск-мониторинга.
    """
    df = pd.read_csv(DATA_PATH, parse_dates=["event_time"])
    max_date = df["event_time"].max()
    cutoff_14d = max_date - timedelta(days=14)

    # Агрегация метрик по мерчантам
    agg = df.groupby(["merchant_id", "region", "country", "mcc_category"], as_index=False).agg(
        total_tx=("TransactionID", "count"),
        gross_tpv_eur=("amount_eur", "sum"),
        net_margin_eur=("net_margin_eur", "sum"),
        cb_count=("is_chargeback", "sum"),
        cb_loss_eur=("chargeback_loss_eur", "sum"),
    )

    # Расчет 14-дневного скользящего окна (Velocity)
    recent = df[df["event_time"] >= cutoff_14d].groupby("merchant_id", as_index=False).agg(
        tpv_14d_eur=("amount_eur", "sum"),
        cb_loss_14d_eur=("chargeback_loss_eur", "sum"),
    )

    marts = agg.merge(recent, on="merchant_id", how="left").fillna(0)
    marts["cb_rate_pct"] = (marts["cb_count"] / marts["total_tx"] * 100).round(2)
    marts["cb_exposure_14d_pct"] = (
        (marts["cb_loss_14d_eur"] / marts["tpv_14d_eur"].replace(0, float("nan"))) * 100
    ).fillna(0).round(2)

    # Логика динамического резерва (Rolling Reserve)
    def assign_reserve(row):
        if row["cb_rate_pct"] >= 5.0 or row["cb_exposure_14d_pct"] >= 8.0:
            return 20.0
        if row["cb_rate_pct"] >= 2.0 or row["cb_exposure_14d_pct"] >= 3.5:
            return 12.0
        if row["region"] == "MENA" or row["mcc_category"] in ("Travel & Airlines", "Online Gaming"):
            return 5.0
        return 2.0

    marts["recommended_reserve_pct"] = marts.apply(assign_reserve, axis=1)
    marts["required_hold_eur"] = (marts["tpv_14d_eur"] * marts["recommended_reserve_pct"] / 100).round(2)

    # Выделяем критические алерты (отрицательная маржа или высокий уровень чарджбэков)
    alerts = marts[(marts["net_margin_eur"] < 0) | (marts["recommended_reserve_pct"] >= 12.0)].copy()
    alerts = alerts.sort_values("net_margin_eur", ascending=True)
    alerts.to_csv(OUTPUT_ALERTS_PATH, index=False)

    logging.info(
        f"Pipeline finished. Flagged {len(alerts)} high-risk/margin-leakage merchants. "
        f"Total recommended reserve hold: €{marts['required_hold_eur'].sum():,.2f}"
    )


with DAG(
    dag_id="finguard_daily_risk_and_reserve_pipeline",
    default_args=default_args,
    description="Daily ETL, Data Quality Gate & Dynamic Rolling Reserve Calculation",
    schedule_interval="0 6 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["fintech", "risk-analytics", "clickhouse", "rolling-reserve"],
) as dag:

    dq_gate = PythonOperator(
        task_id="data_quality_validation_gate",
        python_callable=run_data_quality_checks,
    )

    risk_calculation = PythonOperator(
        task_id="calculate_dynamic_rolling_reserves",
        python_callable=compute_rolling_reserves_and_alerts,
    )

    dq_gate >> risk_calculation
