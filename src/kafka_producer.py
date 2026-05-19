"""
Phase 4 - T4.1
Kafka Producer: يحاكي stream حقيقي من النصوص العربية
"""
import time
import json
import pandas as pd
from kafka import KafkaProducer

KAFKA_BROKER = "localhost:9092"
TOPIC        = "arabic-texts"
DATA_PATH    = "data/raw/raw_data.parquet"

def main():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8")
    )

    df = pd.read_parquet(DATA_PATH)
    sample = df.sample(n=100, random_state=42).reset_index(drop=True)

    print(f"Sending {len(sample)} messages to topic '{TOPIC}' ...")
    for i, row in sample.iterrows():
        msg = {
            "text"  : str(row["text"])[:500],
            "label" : str(row["label"]),
            "id"    : i
        }
        producer.send(TOPIC, value=msg)
        print(f"Sent [{i+1}/100]: label={msg['label']}")
        time.sleep(0.5)

    producer.flush()
    print("Done sending all messages!")

main()
