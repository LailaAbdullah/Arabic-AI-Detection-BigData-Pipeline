"""
Task 4.4 - Scalability Test
Benchmarks batch processing time vs number of executors
"""
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

HDFS_PROCESSED = "hdfs://localhost:9000/user/arabic_ai_detection/data/processed/processed_data.parquet"

def run_benchmark(num_cores):
    print(f"\n{'='*50}")
    print(f"Running with local[{num_cores}] cores...")

    spark = (SparkSession.builder
             .appName(f"Scalability_Test_{num_cores}cores")
             .master(f"local[{num_cores}]")
             .config("spark.driver.memory", "4g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    start = time.time()

    sdf = spark.read.parquet(HDFS_PROCESSED)
    count = sdf.count()

    sdf = sdf.withColumn("word_count", F.size(F.split(F.col("cleaned_text"), " ")))
    avg_words = sdf.groupBy("label").agg(F.avg("word_count")).collect()

    end = time.time()
    elapsed = round(end - start, 2)

    print(f"Cores: {num_cores} | Rows: {count:,} | Time: {elapsed}s")
    for row in avg_words:
        print(f"  Label={row['label']} | Avg words={round(row['avg(word_count)'],2)}")

    spark.stop()
    return elapsed

results = {}
for cores in [1, 2, 4]:
    t = run_benchmark(cores)
    results[cores] = t

print("\n" + "="*50)
print("SCALABILITY SUMMARY")
print("="*50)
print(f"{'Cores':<10} {'Time (s)':<12} {'Speedup':<10}")
print("-"*32)
base = results[1]
for cores, t in results.items():
    speedup = round(base / t, 2)
    print(f"{cores:<10} {t:<12} {speedup}x")
print("\nTask 4.4 Scalability Test Complete!")
