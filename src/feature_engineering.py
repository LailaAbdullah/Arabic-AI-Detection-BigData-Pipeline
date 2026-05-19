"""
Phase 3 - T3.1 & T3.2
Feature Engineering with PySpark
Features: f15 (word length dist), f36 (avg S/P), f57 (top50 embedding), f78 (perplexity)
Advanced: TF-IDF via Spark MLlib
"""

import re
from collections import Counter
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

HDFS_PROCESSED = "hdfs://localhost:9000/user/arabic_ai_detection/data/processed/processed_data.parquet"
HDFS_FEATURES  = "hdfs://localhost:9000/user/arabic_ai_detection/data/processed/features.parquet"

# ── f15: Word Length Variance (توزيع أطوال الكلمات) ──
def word_length_variance(text):
    words = str(text).split() if text else []
    if len(words) < 2:
        return 0.0
    lengths = [len(w) for w in words]
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    return float(variance)

# ── f36: Average Sentences per Paragraph ──
def avg_sentences_per_paragraph(text):
    text = str(text).strip() if text else ""
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    total_sentences = 0
    for p in paragraphs:
        sentences = re.split(r'[.!?؟،]', p)
        total_sentences += len([s for s in sentences if s.strip()])
    return float(total_sentences / len(paragraphs)) if paragraphs else 0.0

# ── f57: عدد الكلمات الشائعة (بديل top-50 embedding) ──
TOP50_WORDS = {
    'درس', 'بحث', 'علم', 'جمع', 'هدف', 'حلل', 'عمل', 'جزير',
    'خدم', 'اثر', 'حقق', 'فهم', 'عزز', 'فعل', 'قدم', 'ركز',
    'عبر', 'ظهر', 'دول', 'تيج', 'نتج', 'وجد', 'كشف', 'طور',
    'استخدم', 'اقترح', 'قيس', 'وصف', 'فسر', 'حدد', 'اختبر',
    'قارن', 'صمم', 'بنى', 'نفذ', 'قيم', 'ادار', 'عالج', 'حسن',
    'طبق', 'وضح', 'اثبت', 'اكتشف', 'فحص', 'راجع', 'لخص',
    'استنتج', 'اقيس', 'نمذج', 'صنف'
}

def count_top50_words(text):
    words = set(str(text).split()) if text else set()
    return int(len(words & TOP50_WORDS))

# ── f78: Perplexity تقريبي (entropy-based) ──
def approx_perplexity(text):
    import math
    words = str(text).split() if text else []
    if not words:
        return 0.0
    freq = Counter(words)
    total = len(words)
    entropy = -sum((c/total) * math.log2(c/total) for c in freq.values())
    return float(entropy)


def main():
    spark = (SparkSession.builder
             .appName("ArabicAIDetection_Phase3_Features")
             .config("spark.driver.memory", "4g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # UDFs
    wlv_udf   = F.udf(word_length_variance, FloatType())
    asp_udf   = F.udf(avg_sentences_per_paragraph, FloatType())
    top50_udf = F.udf(count_top50_words, IntegerType())
    ppl_udf   = F.udf(approx_perplexity, FloatType())

    # قراءة البيانات
    print("Reading processed data from HDFS...")
    sdf = spark.read.parquet(HDFS_PROCESSED)
    print(f"Rows: {sdf.count():,}")

    # ── T3.1: حساب الفيتشرز ──
    print("Computing features f15, f36, f57, f78 ...")
    sdf = (sdf
           .withColumn("f15_word_len_variance",  wlv_udf(F.col("cleaned_text")))
           .withColumn("f36_avg_sent_per_para",  asp_udf(F.col("text")))
           .withColumn("f57_top50_words_count",  top50_udf(F.col("cleaned_text")))
           .withColumn("f78_perplexity",         ppl_udf(F.col("cleaned_text"))))

    print("\nFeature stats:")
    sdf.select("label",
               "f15_word_len_variance",
               "f36_avg_sent_per_para",
               "f57_top50_words_count",
               "f78_perplexity").groupBy("label").agg(
        F.round(F.avg("f15_word_len_variance"), 3).alias("avg_f15"),
        F.round(F.avg("f36_avg_sent_per_para"), 3).alias("avg_f36"),
        F.round(F.avg("f57_top50_words_count"), 3).alias("avg_f57"),
        F.round(F.avg("f78_perplexity"), 3).alias("avg_f78"),
    ).show()

    # ── T3.2: TF-IDF بـ Spark MLlib ──
    print("Computing TF-IDF with Spark MLlib...")
    tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="words")
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=5000)
    idf       = IDF(inputCol="rawFeatures", outputCol="tfidf_features")
    indexer   = StringIndexer(inputCol="label", outputCol="label_idx")

    pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, indexer])
    model    = pipeline.fit(sdf)
    sdf      = model.transform(sdf)

    print("TF-IDF computed!")
    print(f"Feature vector size: 5000")

    # ── T3.3: تقسيم البيانات 70/15/15 ──
    print("\nSplitting data 70/15/15 ...")
    train_df, val_df, test_df = sdf.randomSplit([0.70, 0.15, 0.15], seed=42)
    print(f"Train : {train_df.count():,}")
    print(f"Val   : {val_df.count():,}")
    print(f"Test  : {test_df.count():,}")

    # حفظ التقسيمات
    train_df.write.mode("overwrite").parquet(
        "hdfs://localhost:9000/user/arabic_ai_detection/data/processed/train.parquet")
    val_df.write.mode("overwrite").parquet(
        "hdfs://localhost:9000/user/arabic_ai_detection/data/processed/val.parquet")
    test_df.write.mode("overwrite").parquet(
        "hdfs://localhost:9000/user/arabic_ai_detection/data/processed/test.parquet")

    print("Train/Val/Test saved to HDFS!")

    spark.stop()
    print("\n Phase 3 Feature Engineering completed!")

main()
