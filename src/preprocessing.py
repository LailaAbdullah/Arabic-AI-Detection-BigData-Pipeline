"""
Phase 2 - Arabic Text Preprocessing with PySpark
T2.1: Preprocessing pipeline
T2.2: Save as Parquet to HDFS
T2.3: EDA (n-gram frequency, TTR)
MapReduce: Word Count + N-Gram using Spark RDD
"""

import re
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from nltk.stem.isri import ISRIStemmer

# ── Config ──
HDFS_RAW       = "hdfs://localhost:9000/user/arabic_ai_detection/data/raw/raw_data.parquet"
HDFS_PROCESSED = "hdfs://localhost:9000/user/arabic_ai_detection/data/processed/processed_data.parquet"
STOPWORDS_PATH = "data/arabic_stopwords.txt"

# ── تحميل الـ stopwords ──
with open(STOPWORDS_PATH, "r") as f:
    STOPWORDS = set(f.read().strip().split())

stemmer = ISRIStemmer()

# ── دوال المعالجة ──
arabic_diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652]')

def remove_diacritics(text):
    return re.sub(arabic_diacritics, '', text)

def normalize_arabic(text):
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[^؀-ۿ ]+', ' ', text)
    return text

def preprocess(text):
    if not text:
        return ""
    text = str(text)
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOPWORDS]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

preprocess_udf = F.udf(preprocess, StringType())


def main():
    spark = (SparkSession.builder
             .appName("ArabicAIDetection_Phase2")
             .config("spark.driver.memory", "4g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # ── T2.1: قراءة البيانات من HDFS ──
    print("=" * 50)
    print("T2.1 - Reading data from HDFS...")
    sdf = spark.read.parquet(HDFS_RAW)
    print(f"Rows loaded: {sdf.count():,}")

    # ── تطبيق المعالجة ──
    print("Applying Arabic preprocessing...")
    sdf = sdf.withColumn("cleaned_text", preprocess_udf(F.col("text")))
    sdf = sdf.filter(F.col("cleaned_text") != "")
    print("Preprocessing done!")

    # ── T2.2: حفظ Parquet على HDFS ──
    print("=" * 50)
    print("T2.2 - Saving processed data to HDFS as Parquet...")
    (sdf.write
        .mode("overwrite")
        .option("compression", "snappy")
        .parquet(HDFS_PROCESSED))
    print(f"Saved → {HDFS_PROCESSED}")

    # ── T2.3: EDA ──
    print("=" * 50)
    print("T2.3 - EDA on processed data")

    # Type-Token Ratio (TTR)
    def compute_ttr(text):
        words = text.split()
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    ttr_udf = F.udf(compute_ttr)
    sdf_ttr = sdf.withColumn("ttr", ttr_udf(F.col("cleaned_text")))
    print("\n── TTR by label ──")
    sdf_ttr.groupBy("label").agg(
        F.round(F.avg("ttr"), 4).alias("avg_TTR")
    ).show()

    # ── MapReduce: Word Count ──
    print("=" * 50)
    print("MapReduce Job 1 - Word Count")
    words_rdd = (sdf.select("cleaned_text").rdd
                 .flatMap(lambda row: row[0].split() if row[0] else []))
    word_count = (words_rdd
                  .map(lambda w: (w, 1))
                  .reduceByKey(lambda a, b: a + b)
                  .sortBy(lambda x: x[1], ascending=False))
    top20 = word_count.take(20)
    print("\nTop 20 words:")
    for word, count in top20:
        print(f"  {word}: {count:,}")

    # ── MapReduce: Bigram Frequency ──
    print("=" * 50)
    print("MapReduce Job 2 - Bigram Frequency")
    def extract_bigrams(text):
        words = text.split() if text else []
        return [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]

    bigram_count = (sdf.select("cleaned_text").rdd
                    .flatMap(lambda row: extract_bigrams(row[0]) if row[0] else [])
                    .map(lambda bg: (bg, 1))
                    .reduceByKey(lambda a, b: a + b)
                    .sortBy(lambda x: x[1], ascending=False))
    top10_bigrams = bigram_count.take(10)
    print("\nTop 10 Bigrams:")
    for bg, count in top10_bigrams:
        print(f"  '{bg}': {count:,}")

    # ── إحصائيات نهائية ──
    print("=" * 50)
    print("Final Stats:")
    sdf.groupBy("label").agg(
        F.count("*").alias("count"),
        F.round(F.avg(F.length("cleaned_text")), 1).alias("avg_cleaned_length")
    ).show()

    spark.stop()
    print("Phase 2 completed!")

main()
