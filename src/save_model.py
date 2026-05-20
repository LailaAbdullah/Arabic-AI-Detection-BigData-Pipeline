from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.appName("SaveModel").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

train_df = spark.read.parquet("hdfs://localhost:9000/user/arabic_ai_detection/data/processed/train.parquet")

assembler = VectorAssembler(
    inputCols=["tfidf_features","f15_word_len_variance","f36_avg_sent_per_para","f57_top50_words_count","f78_perplexity"],
    outputCol="features")

train_df = assembler.transform(train_df)
lr = LogisticRegression(featuresCol="features", labelCol="label_idx", maxIter=100, regParam=0.01)
lr_model = lr.fit(train_df)
lr_model.save("models/lr_model")
print("Model saved to models/lr_model")
spark.stop()
