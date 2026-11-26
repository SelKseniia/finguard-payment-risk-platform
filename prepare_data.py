import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. Загрузка и сэмплирование реальных транзакций IEEE-CIS
tx = pd.read_csv("train_transaction.csv.zip", usecols=[
    "TransactionID", "TransactionDT", "TransactionAmt", "ProductCD",
    "card4", "card6", "P_emaildomain", "isFraud"
])
identity = pd.read_csv("train_identity.csv.zip", usecols=[
    "TransactionID", "DeviceType", "DeviceInfo", "id_30", "id_31"
])

df = tx.merge(identity, on="TransactionID", how="left")

# Берем срез 180 000 транзакций, сохраняя пропорцию фрода (чтобы Tableau работал быстро)
df = df.sample(n=180000, random_state=42).sort_values("TransactionDT").reset_index(drop=True)

# Превращаем секунды TransactionDT в реальные даты (например, 6 месяцев 2025 года)
start_date = datetime(2025, 1, 1)
df["event_time"] = df["TransactionDT"].apply(lambda x: start_date + timedelta(seconds=int(x)))
df["amount_eur"] = (df["TransactionAmt"] * 0.92).round(2)  # Переводим в EUR

# 2. Генерируем справочник из 200 мерчантов (EU & MENA)
np.random.seed(42)
n_merchants = 200

countries_config = [
    ("DE", "EU", 0.003, 0.012),    # Германия: низкий interchange (регулируется ЕС)
    ("ES", "EU", 0.003, 0.014),    # Испания
    ("NL", "EU", 0.003, 0.013),    # Нидерланды
    ("CY", "EU", 0.004, 0.018),    # Кипр (часто high-risk/fintech)
    ("AE", "MENA", 0.016, 0.028),  # ОАЭ: высокий cross-border interchange и высокая маржа
    ("SA", "MENA", 0.018, 0.030),  # Саудовская Аравия
    ("EG", "MENA", 0.019, 0.032),  # Египет
]

mccs = ["E-commerce Retail", "Digital Goods", "Travel & Airlines", "SaaS & Subscriptions", "Online Gaming"]

merchants = []
for i in range(1, n_merchants + 1):
    m_id = f"MERCH_{i:04d}"
    country, region, base_ic, base_take_rate = countries_config[i % len(countries_config)]
    mcc = mccs[i % len(mccs)]
    
    # Базовая комиссия, которую мы берем с мерчанта (merchant_fee_pct)
    take_rate = round(base_take_rate + np.random.uniform(-0.002, 0.006), 4)
    # Базовый IP-префикс мерчанта (для дальнейшей кластеризации)
    ip_subnet = f"185.{np.random.randint(10, 250)}.{np.random.randint(1, 250)}"
    
    merchants.append({
        "merchant_id": m_id,
        "merchant_name": f"{mcc.split()[0]}_Partner_{i}",
        "country": country,
        "region": region,
        "mcc_category": mcc,
        "contract_fee_pct": take_rate,
        "ip_subnet": ip_subnet,
        "anomaly_tag": "Normal"
    })

merchants_df = pd.DataFrame(merchants)

# 3. Зашиваем 3 боевые аномалии в справочник мерчантов
# Аномалия А: Кольцо фрода (4 мерчанта сидят на одной IP-подсети)
fraud_ring_ids = ["MERCH_0014", "MERCH_0028", "MERCH_0042", "MERCH_0056"]
merchants_df.loc[merchants_df["merchant_id"].isin(fraud_ring_ids), "ip_subnet"] = "185.220.101"
merchants_df.loc[merchants_df["merchant_id"].isin(fraud_ring_ids), "anomaly_tag"] = "Fraud_Ring_IP"

# Аномалия Б: Утечка маржи (сейлзы дали слишком низкий тариф 0.6% в дорогом регионе)
leakage_ids = ["MERCH_0005", "MERCH_0019", "MERCH_0033"]
merchants_df.loc[merchants_df["merchant_id"].isin(leakage_ids), "contract_fee_pct"] = 0.0060
merchants_df.loc[merchants_df["merchant_id"].isin(leakage_ids), "anomaly_tag"] = "Margin_Leakage"

# Аномалия В: Всплеск чарджбэков в Travel (нужен срочный пересмотр Rolling Reserve)
spike_ids = ["MERCH_0007", "MERCH_0021"]
merchants_df.loc[merchants_df["merchant_id"].isin(spike_ids), "anomaly_tag"] = "Chargeback_Spike"

# 4. Распределяем транзакции по мерчантам и считаем экономику каждой операции
# Фродовые транзакции из IEEE-CIS чаще направляем в Fraud_Ring и Chargeback_Spike
normal_merchants = merchants_df[merchants_df["anomaly_tag"] == "Normal"]["merchant_id"].values
risky_merchants = merchants_df[merchants_df["anomaly_tag"].isin(["Fraud_Ring_IP", "Chargeback_Spike"])]["merchant_id"].values

df["merchant_id"] = np.where(
    (df["isFraud"] == 1) & (np.random.rand(len(df)) < 0.45),
    np.random.choice(risky_merchants, size=len(df)),
    np.random.choice(normal_merchants, size=len(df))
)

# Объединяем транзакции с данными мерчантов
full_df = df.merge(merchants_df, on="merchant_id", how="left")

# Усиливаем аномалию В: в последние 30 дней у spike_ids резко растет доля фрода/чарджбэков
max_date = full_df["event_time"].max()
recent_mask = (full_df["merchant_id"].isin(spike_ids)) & (full_df["event_time"] >= max_date - timedelta(days=30))
full_df.loc[recent_mask, "isFraud"] = np.where(np.random.rand(recent_mask.sum()) < 0.18, 1, full_df.loc[recent_mask, "isFraud"])

# Генерируем конкретный IP-адрес клиента/мерчанта внутри подсети
full_df["ip_address"] = full_df["ip_subnet"] + "." + np.random.randint(1, 254, size=len(full_df)).astype(str)

# Считаем реальную экономику (Interchange + Gateway Cost vs Наша комиссия)
ic_map = dict(zip(merchants_df["country"], [c[2] for c in countries_config]))
full_df["interchange_rate"] = full_df["country"].map(ic_map)
# Корпоративные/кредитные карты дороже в обработке (+0.4%)
full_df["interchange_rate"] += np.where(full_df["card6"] == "credit", 0.004, 0.0)

full_df["interchange_fee_eur"] = (full_df["amount_eur"] * full_df["interchange_rate"]).round(2)
full_df["gateway_cost_eur"] = 0.12  # Фиксированная себестоимость транзакции (12 евроцентов)
full_df["merchant_fee_eur"] = (full_df["amount_eur"] * full_df["contract_fee_pct"]).round(2)

# Статус чарджбэка (около 85% от фрода превращается в официальный чарджбэк + штраф 15 EUR)
full_df["is_chargeback"] = np.where((full_df["isFraud"] == 1) & (np.random.rand(len(full_df)) < 0.85), 1, 0)
full_df["chargeback_loss_eur"] = np.where(full_df["is_chargeback"] == 1, full_df["amount_eur"] + 15.0, 0.0).round(2)

# Чистая маржа шлюза (Net Margin) по транзакции
full_df["net_margin_eur"] = (
    full_df["merchant_fee_eur"] - full_df["interchange_fee_eur"] - full_df["gateway_cost_eur"] - full_df["chargeback_loss_eur"]
).round(2)

# 5. Сохраняем итоговые таблицы для ClickHouse, ML и Tableau
merchants_df.to_csv("dim_merchants.csv", index=False)

cols_to_export = [
    "TransactionID", "event_time", "merchant_id", "country", "region", "mcc_category",
    "amount_eur", "card4", "card6", "DeviceType", "DeviceInfo", "ip_address",
    "merchant_fee_eur", "interchange_fee_eur", "gateway_cost_eur",
    "isFraud", "is_chargeback", "chargeback_loss_eur", "net_margin_eur"
]
full_df[cols_to_export].to_csv("fact_transactions.csv", index=False)

print("Готово! Созданы файлы:")
print(f"1. dim_merchants.csv ({len(merchants_df)} мерчантов)")
print(f"2. fact_transactions.csv ({len(full_df)} транзакций)")
print(f"Общий оборот: €{full_df['amount_eur'].sum():,.2f} | Чистая маржа: €{full_df['net_margin_eur'].sum():,.2f}")
