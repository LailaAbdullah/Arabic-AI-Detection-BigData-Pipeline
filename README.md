# Scalable Real-time Detection of AI-Generated Arabic Text

## Project Overview
A distributed Big Data pipeline for detecting AI-generated Arabic text using Apache Spark, Kafka, and HDFS.

## Dataset
KFUPM-JRCAI/arabic-generated-abstracts (HuggingFace)
- Total: 41,940 samples (8,388 human + 33,552 AI)
- Generation methods: by_polishing, from_title, from_title_and_content

## Pipeline Architecture
## Features Engineered (Student: Laila - Position 15)
| Feature | Description |
|---------|-------------|
| f15 - Word Length Variance | Distribution of word lengths |
| f36 - Avg Sentences/Paragraph | Structural complexity |
| f57 - Top50 Embedding Words | Semantic richness |
| f78 - Perplexity Score | Text predictability |

## Model Results
| Model | Accuracy | F1 | AUC-ROC |
|-------|----------|-----|---------|
| Logistic Regression | 96.7% | 96.7% | 99.0% |
| Linear SVM | 95.9% | 96.0% | 98.3% |
| Random Forest | 79.8% | 70.8% | 94.7% |

## How to Run
```bash
start-dfs.sh
spark-submit --master local[*] src/data_acquisition.py
spark-submit --master local[*] src/preprocessing.py
spark-submit --master local[*] src/feature_engineering.py
spark-submit --master local[*] src/modeling.py
```

## Environment
- Apache Spark 3.5.1
- Hadoop 3.x (HDFS)
- Apache Kafka 3.7.0
- Python 3.12
- CentOS Linux aarch64
