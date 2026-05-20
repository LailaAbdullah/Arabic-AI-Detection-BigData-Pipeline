"""
Task 3.5 - FFNN using Spark MLlib MultilayerPerceptronClassifier
"""
from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics

spark = SparkSession.builder.appName("FFNN_Spark").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

print("Loading data...")
train_df = spark.read.parquet("hdfs://localhost:9000/user/arabic_ai_detection/data/processed/train.parquet")
test_df  = spark.read.parquet("hdfs://localhost:9000/user/arabic_ai_detection/data/processed/test.parquet")

assembler = VectorAssembler(
    inputCols=["tfidf_features","f15_word_len_variance","f36_avg_sent_per_para","f57_top50_words_count","f78_perplexity"],
    outputCol="features")

train_df = assembler.transform(train_df)
test_df  = assembler.transform(test_df)

# FFNN: input=5004, hidden=[256, 128], output=2
layers = [5004, 256, 128, 2]

mlp = MultilayerPerceptronClassifier(
    featuresCol="features",
    labelCol="label_idx",
    layers=layers,
    maxIter=50,
    seed=42,
    blockSize=128
)

print("Training FFNN (MLP) with Spark MLlib...")
mlp_model = mlp_model = mlp.fit(train_df)
predictions = mlp_model.transform(test_df)

multi_eval = MulticlassClassificationEvaluator(labelCol="label_idx", predictionCol="prediction")
binary_eval = BinaryClassificationEvaluator(labelCol="label_idx", rawPredictionCol="rawPrediction")

accuracy  = multi_eval.evaluate(predictions, {multi_eval.metricName: "accuracy"})
f1        = multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"})
precision = multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedPrecision"})
recall    = multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedRecall"})
auc       = binary_eval.evaluate(predictions)

print("\n" + "="*50)
print("FFNN (MLP) Results:")
print("="*50)
print(f"Accuracy  : {accuracy:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"AUC-ROC   : {auc:.4f}")

preds_labels = predictions.select("prediction","label_idx").rdd.map(lambda r: (r[0], r[1]))
from pyspark.mllib.evaluation import MulticlassMetrics
metrics = MulticlassMetrics(preds_labels)
print("\nConfusion Matrix:")
print(metrics.confusionMatrix().toArray().astype(int))

mlp_model.save("models/ffnn_model")
print("\nFFNN model saved to models/ffnn_model")
spark.stop()
