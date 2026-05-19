"""
Phase 4 - T4.2
Spark Structured Streaming: يقرأ من Kafka ويصنّف النصوص
"""
import time
import re
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, FloatType, IntegerType, StructType, StructField

KAFKA_BROKER   = "localhost:9092"
TOPIC          = "arabic-texts"
CHECKPOINT_DIR = "/tmp/spark_checkpoint"

def main():
    spark = (SparkSession.builder
             .appName("ArabicAIDetection_Streaming")
             .config("spark.driver.memory", "4g")
             .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # دوال المعالجة
    def preprocess(text):
        if not text:
            return ""
        text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', str(text))
        text = re.sub(r'[إأآا]', 'ا', text)
        text = re.sub(r'ى', 'ي', text)
        text = re.sub(r'ة', 'ه', text)
        text = re.sub(r'[^؀-ۿ ]+', ' ', text)
        return text.strip()

    TOP50 = {'درس','بحث','علم','جمع','هدف','حلل','عمل','جزير','خدم','اثر',
             'حقق','فهم','عزز','فعل','قدم','ركز','عبر','ظهر','دول','تيج'}

    def simple_classify(text):
        if not text:
            return "unknown"
        words = set(text.split())
        overlap = len(words & TOP50)
        avg_len = sum(len(w) for w in words) / max(len(words), 1)
        score = overlap * 0.6 + (1 / max(avg_len, 1)) * 10
        return "ai" if score > 3.5 else "human"

    preprocess_udf = F.udf(preprocess, StringType())
    classify_udf   = F.udf(simple_classify, StringType())

    # قراءة من Kafka
    print("Starting Spark Structured Streaming from Kafka...")
    kafka_df = (spark.readStream
                .format("kafka")
                .option("kafka.bootstrap.servers", KAFKA_BROKER)
                .option("subscribe", TOPIC)
                .option("startingOffsets", "latest")
                .load())

    # تحليل الـ JSON
    schema = StructType([
        StructField("text",  StringType()),
        StructField("label", StringType()),
        StructField("id",    IntegerType()),
    ])

    parsed_df = (kafka_df
                 .select(F.from_json(F.col("value").cast("string"), schema).alias("data"))
                 .select("data.*"))

    # معالجة وتصنيف
    result_df = (parsed_df
                 .withColumn("cleaned", preprocess_udf(F.col("text")))
                 .withColumn("prediction", classify_udf(F.col("cleaned")))
                 .withColumn("timestamp", F.current_timestamp())
                 .select("id", "label", "prediction", "timestamp"))

    # طباعة النتائج
    query = (result_df.writeStream
             .outputMode("append")
             .format("console")
             .option("truncate", False)
             .option("checkpointLocation", CHECKPOINT_DIR)
             .start())

    print("Streaming started! Waiting for messages...")
    query.awaitTermination(timeout=90)
    print("Streaming completed!")
    spark.stop()

main()
