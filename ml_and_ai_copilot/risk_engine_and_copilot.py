import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score

# Определяем пути относительно корня проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "ml_and_ai_copilot")

print("1. Загружаем витрину транзакций и справочник мерчантов...")
tx = pd.read_csv(os.path.join(DATA_DIR, "fact_transactions.csv"), parse_dates=["event_time"])
merchants = pd.read_csv(os.path.join(DATA_DIR, "dim_merchants.csv"))

# ============================================================================
# БЛОК А: ПОИСК СВЯЗАННЫХ МЕРЧАНТОВ ПО IP-ПОДСЕТЯМ (FRAUD RING DETECTION)
# ============================================================================
print("2. Запускаем кластеризацию IP-подсетей для поиска связанных аккаунтов...")
tx["ip_subnet_24"] = tx["ip_address"].apply(lambda x: ".".join(str(x).split(".")[:3]))

subnet_clusters = (
    tx.groupby("ip_subnet_24")
    .agg(
        linked_merchants_count=("merchant_id", "nunique"),
        linked_merchant_ids=("merchant_id", lambda x: ", ".join(sorted(set(x)))),
        total_subnet_tpv=("amount_eur", "sum"),
        subnet_cb_rate=("is_chargeback", "mean"),
    )
    .reset_index()
)

# Выявляем кластеры, где больше 1 мерчанта делят одну /24 подсеть
fraud_rings = subnet_clusters[subnet_clusters["linked_merchants_count"] > 1].copy()
linked_merchant_set = set()
for ids_str in fraud_rings["linked_merchant_ids"]:
    linked_merchant_set.update(ids_str.split(", "))

print(f"   -> Найдено кластеров сговора: {len(fraud_rings)} (вовлечено мерчантов: {len(linked_merchant_set)})")

# ============================================================================
# БЛОК Б: ML-СКОРИНГ ТРАНЗАКЦИЙ И ОЦЕНКА ФИНАНСОВОГО ЭФФЕКТА (ROI)
# ============================================================================
print("3. Обучаем ML-модель (Random Forest) для предсказания вероятности чарджбэка...")
features_df = tx.copy()
features_df["hour"] = features_df["event_time"].dt.hour
features_df["is_credit"] = (features_df["card6"] == "credit").astype(int)
features_df["is_mena"] = (features_df["region"] == "MENA").astype(int)
features_df["is_linked_ip"] = features_df["merchant_id"].isin(linked_merchant_set).astype(int)
features_df["fee_spread"] = features_df["merchant_fee_eur"] - features_df["interchange_fee_eur"]

feature_cols = ["amount_eur", "hour", "is_credit", "is_mena", "is_linked_ip", "fee_spread"]
X = features_df[feature_cols].fillna(0)
y = features_df["is_chargeback"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Оцениваем качество модели на тестовой выборке
test_probs = clf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, test_probs)
print(f"   -> Качество ML-модели на тесте: ROC-AUC = {roc_auc:.3f}")

# Проставляем ML Risk Score (от 0 до 100) на все транзакции
tx["ml_risk_score"] = (clf.predict_proba(X)[:, 1] * 100).round(1)

# ============================================================================
# БЛОК В: АГРЕГАЦИЯ ВИТРИНЫ ДЛЯ TABLEAU И AI RISK COPILOT
# ============================================================================
print("4. Собираем итоговую витрину мерчантов с рекомендациями по резервам...")
max_date = tx["event_time"].max()
cutoff_14d = max_date - pd.Timedelta(days=14)

m_summary = tx.groupby(["merchant_id", "region", "country", "mcc_category"], as_index=False).agg(
    total_tx=("TransactionID", "count"),
    gross_tpv_eur=("amount_eur", "sum"),
    net_margin_eur=("net_margin_eur", "sum"),
    cb_count=("is_chargeback", "sum"),
    cb_loss_eur=("chargeback_loss_eur", "sum"),
    avg_ml_score=("ml_risk_score", "mean"),
)

recent_14d = tx[tx["event_time"] >= cutoff_14d].groupby("merchant_id", as_index=False).agg(
    tpv_14d_eur=("amount_eur", "sum"),
    cb_loss_14d_eur=("chargeback_loss_eur", "sum"),
)

m_summary = m_summary.merge(merchants[["merchant_id", "merchant_name", "contract_fee_pct", "ip_subnet"]], on="merchant_id")
m_summary = m_summary.merge(recent_14d, on="merchant_id", how="left").fillna(0)

m_summary["cb_rate_pct"] = (m_summary["cb_count"] / m_summary["total_tx"] * 100).round(2)
m_summary["cb_exposure_14d_pct"] = (
    (m_summary["cb_loss_14d_eur"] / m_summary["tpv_14d_eur"].replace(0, np.nan)) * 100
).fillna(0).round(2)
m_summary["is_ip_linked"] = m_summary["merchant_id"].isin(linked_merchant_set)

# Назначаем категорию риска, динамический резерв и рекомендацию
def evaluate_merchant(row):
    if row["is_ip_linked"]:
        return "FRAUD_RING_LINKED", 25.0, "IMMEDIATE FREEZE (25%) + KYB Investigation: Linked via shared /24 IP subnet."
    if row["cb_rate_pct"] >= 5.0 or row["cb_exposure_14d_pct"] >= 8.0:
        return "HIGH_FRAUD_VELOCITY", 20.0, "INCREASE HOLD TO 20%: Abnormal chargeback spike detected."
    if row["net_margin_eur"] < 0 and row["cb_rate_pct"] < 2.0:
        return "MARGIN_LEAKAGE", 5.0, "REPRICE TARIFF: Contract fee below Interchange + Gateway cost."
    if row["region"] == "MENA" or row["mcc_category"] in ("Travel & Airlines", "Online Gaming"):
        return "ELEVATED_BASELINE", 5.0, "STANDARD HIGH-RISK HOLD (5%): Monitor 14d velocity."
    return "HEALTHY", 2.0, "STANDARD HOLD (2%): Healthy unit economics."

eval_results = m_summary.apply(evaluate_merchant, axis=1)
m_summary["risk_status"] = [r[0] for r in eval_results]
m_summary["recommended_reserve_pct"] = [r[1] for r in eval_results]
m_summary["copilot_action"] = [r[2] for r in eval_results]
m_summary["required_reserve_eur"] = (m_summary["tpv_14d_eur"] * m_summary["recommended_reserve_pct"] / 100).round(2)

# Сохраняем обогащенную витрину мерчантов (идеально для Tableau Public!)
tableau_mart_path = os.path.join(DATA_DIR, "tableau_merchant_risk_mart.csv")
m_summary.to_csv(tableau_mart_path, index=False)

# ============================================================================
# БЛОК Г: AI RISK COPILOT (ГЕНЕРАЦИЯ ИСПОЛНИТЕЛЬНЫХ РИСК-МЕМОРАНДУМОВ)
# ============================================================================
def generate_llm_risk_memo(row):
    """
    Формирует структурированный риск-меморандум (Structured Output / RAG-шаблон,
    готовый к подключению OpenAI / Anthropic API в продакшн-контуре).
    """
    return f"""### Risk Investigation Memo: {row['merchant_name']} (`{row['merchant_id']}`)
- **Alert Category:** `{row['risk_status']}` | **Region / MCC:** {row['region']} ({row['country']}) — {row['mcc_category']}
- **Financial Impact:** Gross TPV: `€{row['gross_tpv_eur']:,.2f}` | Net Margin: `€{row['net_margin_eur']:,.2f}` | Chargeback Losses: `€{row['cb_loss_eur']:,.2f}`
- **Risk Telemetry:** Chargeback Rate: `{row['cb_rate_pct']}%` | 14d Exposure: `{row['cb_exposure_14d_pct']}%` | Avg ML Risk Score: `{row['avg_ml_score']:.1f}/100`
- **Network / Pricing Context:** IP Subnet `{row['ip_subnet']}.x` (Linked Ring: `{row['is_ip_linked']}`) | Contract Fee: `{row['contract_fee_pct']*100:.2f}%`
- **AI Copilot Recommendation:** **{row['copilot_action']}** (Target Rolling Reserve Hold: `€{row['required_reserve_eur']:,.2f}`)
---"""

flagged = m_summary[m_summary["risk_status"].isin(["FRAUD_RING_LINKED", "HIGH_FRAUD_VELOCITY", "MARGIN_LEAKAGE"])].sort_values("net_margin_eur")

memo_path = os.path.join(OUT_DIR, "AI_COPILOT_SAMPLE_MEMOS.md")
with open(memo_path, "w", encoding="utf-8") as f:
    f.write("# FinGuard AI Risk Copilot — Automated Investigation Memos\n\n")
    f.write("> *Generated automatically from ClickHouse risk marts & ML scoring telemetry.*\n\n")
    for _, r in flagged.iterrows():
        f.write(generate_llm_risk_memo(r) + "\n\n")

print(f"5. Готово! Сохранены артефакты:")
print(f"   -> Витрина для Tableau Public: data/tableau_merchant_risk_mart.csv")
print(f"   -> Отчеты AI Risk Copilot: ml_and_ai_copilot/AI_COPILOT_SAMPLE_MEMOS.md (выявлено {len(flagged)} критических мерчантов)")
print(f"   -> Сумма рекомендуемых заморозок (Rolling Reserve) по портфелю: €{m_summary['required_reserve_eur'].sum():,.2f}")
