"""
Phase 3 - T3.4 & T3.5
Spark MLlib Models:
T3.4: Baseline - Logistic Regression
T3.5: Random Forest + Linear SVM
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import MulticlassMetrics

HDFS_BASE = "hdfs://localhost:9000/user/arabic_ai_detection/data/processed"

def evaluate_model(predictions, model_name):
    print(f"\n{'='*50}")
    print(f"Results: {model_name}")
    print('='*50)

    binary_eval = BinaryClassificationEvaluator(labelCol="label_idx", rawPredictionCol="rawPrediction")
    auc = binary_eval.evaluate(predictions)

    multi_eval = MulticlassClassificationEvaluator(labelCol="label_idx", predictionCol="prediction")
    accuracy = multi_eval.evaluate(predictions, {multi_eval.metricName: "accuracy"})
    f1       = multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"})
    precision= multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedPrecision"})
    recall   = multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedRecall"})

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"AUC-ROC   : {auc:.4f}")

    # Confusion Matrix
    preds_labels = predictions.select("prediction", "label_idx").rdd.map(lambda r: (r[0], r[1]))
    metrics = MulticlassMetrics(preds_labels)
    print("\nConfusion Matrix:")
    print(metrics.confusionMatrix().toArray().astype(int))

    return accuracy, f1, auc

def main():
    spark = (SparkSession.builder
             .appName("ArabicAIDetection_Phase3_Modeling")
             .config("spark.driver.memory", "4g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # قراءة البيانات
    print("Loading train/val/test from HDFS...")
    train_df = spark.read.parquet(f"{HDFS_BASE}/train.parquet")
    val_df   = spark.read.parquet(f"{HDFS_BASE}/val.parquet")
    test_df  = spark.read.parquet(f"{HDFS_BASE}/test.parquet")

    # دمج الفيتشرز
    assembler = VectorAssembler(
        inputCols=["tfidf_features", "f15_word_len_variance",
                   "f36_avg_sent_per_para", "f57_top50_words_count", "f78_perplexity"],
        outputCol="features"
    )
    train_df = assembler.transform(train_df)
    val_df   = assembler.transform(val_df)
    test_df  = assembler.transform(test_df)

    results = {}

    # ── T3.4: Baseline - Logistic Regression ──
    print("\n" + "="*50)
    print("T3.4 - Baseline: Logistic Regression")
    lr = LogisticRegression(featuresCol="features", labelCol="label_idx",
                            maxIter=100, regParam=0.01)
    lr_model = lr.fit(train_df)
    lr_preds = lr_model.transform(test_df)
    acc, f1, auc = evaluate_model(lr_preds, "Logistic Regression")
    results["Logistic Regression"] = (acc, f1, auc)

    # ── T3.5a: Random Forest ──
    print("\n" + "="*50)
    print("T3.5 - Random Forest")
    rf = RandomForestClassifier(featuresCol="features", labelCol="label_idx",
                                numTrees=100, seed=42)
    rf_model = rf.fit(train_df)
    rf_preds = rf_model.transform(test_df)
    acc, f1, auc = evaluate_model(rf_preds, "Random Forest")
    results["Random Forest"] = (acc, f1, auc)

    # Feature Importance
    print("\nTop 5 Feature Importances (RF):")
    importances = rf_model.featureImportances
    feat_names  = ["tfidf"] * 5000 + ["f15_word_len_var", "f36_avg_s_p", "f57_top50", "f78_perplexity"]
    top_idx = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)[:5]
    for idx in top_idx:
        name = feat_names[idx] if idx < len(feat_names) else f"feat_{idx}"
        print(f"  [{idx}] {name}: {importances[idx]:.4f}")

    # ── T3.5b: Linear SVM ──
    print("\n" + "="*50)
    print("T3.5 - Linear SVM")
    svm = LinearSVC(featuresCol="features", labelCol="label_idx", maxIter=100)
    svm_model = svm.fit(train_df)
    svm_preds = svm_model.transform(test_df)
    acc, f1, auc = evaluate_model(svm_preds, "Linear SVM")
    results["Linear SVM"] = (acc, f1, auc)

    # ── مقارنة النماذج ──
    print("\n" + "="*50)
    print("Model Comparison Summary")
    print("="*50)
    print(f"{'Model':<25} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    print("-"*55)
    for name, (acc, f1, auc) in results.items():
        print(f"{name:<25} {acc:>10.4f} {f1:>10.4f} {auc:>10.4f}")

    # حفظ أفضل نموذج
    best = max(results, key=lambda k: results[k][1])
    print(f"\nBest model: {best}")

    if best == "Logistic Regression":
        lr_model.save("hdfs://localhost:9000/user/arabic_ai_detection/models/lr_model")
    elif best == "Random Forest":
        rf_model.save("hdfs://localhost:9000/user/arabic_ai_detection/models/rf_model")
    else:
        svm_model.save("hdfs://localhost:9000/user/arabic_ai_detection/models/svm_model")

    print(f"Best model saved to HDFS!")
    spark.stop()
    print("\n Phase 3 Modeling completed!")

main()
