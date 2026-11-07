from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, sum, regexp_replace, to_date,
    dayofmonth, month, dayofweek,
    when
)

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)

from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier
)

from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

OUTPUT_DIR = (
    "E:/New folder/New folder-7/INT315/CA2/"
    "CreditCardFraudDetection/Visualizations"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# 1. CREATE SPARK SESSION
# ==================================================

spark = SparkSession.builder \
    .appName("CreditCardFraudDetection") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("\n========== CREDIT CARD FRAUD DETECTION ==========")

# ==================================================
# 2. LOADING DATASET
# ==================================================

data_path = "E:/New folder/New folder-7/INT315/CA2/CreditCardFraudDetection/Dataset/CreditCardData.csv"

df = spark.read.csv(
    data_path,
    header=True,
    inferSchema=True
)

# Display Dataset
print("\nOriginal Dataset:")
df.show(10, truncate=False)

# Display Schema
print("\nDataset Schema:")
df.printSchema()

# Display Total Rows and Columns
print("\nTotal Rows:", df.count())
print("Total Columns:", len(df.columns))

# Display Column Names
print("\nColumn Names:")
print(df.columns)

# Check Missing Values
print("\nMissing Values:")

missing_values = df.select([
    sum(
        when(col(c).isNull(), 1).otherwise(0)
    ).alias(c)
    for c in df.columns
])

missing_values.show(truncate=False)

# Check Fraud Distribution
print("\nFraud Distribution:")

df.groupBy("Fraud").count().orderBy("Fraud").show()

fraud_data = df.groupBy("Fraud").count().toPandas()

plt.figure(figsize=(8, 5))

sns.barplot(
    data=fraud_data,
    x="Fraud",
    y="count"
)

plt.title("Credit Card Fraud Distribution")
plt.xlabel("Transaction Type (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.xticks([0, 1], ["Legitimate", "Fraudulent"])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_fraud_distribution.png"),dpi=300)
plt.show()

# Basic Statistical Analysis
print("\nStatistical Summary:")

df.describe("Age", "Time").show()

print("\nDataset Exploration Completed!")

# ==================================================
# 3. DATA PREPROCESSING
# ==================================================

# Remove Transaction ID
df = df.drop("Transaction ID")

# Clean Amount
df = df.withColumn(
    "Amount",
    regexp_replace(col("Amount"), "[^0-9.]", "").cast("double")
)

# Convert Date
df = df.withColumn(
    "Date",
    to_date(col("Date"), "d-MMM-yy")
)

# Categorical columns
categorical_cols = [
    "Day of Week",
    "Type of Card",
    "Entry Mode",
    "Type of Transaction",
    "Merchant Group",
    "Country of Transaction",
    "Shipping Address",
    "Country of Residence",
    "Gender",
    "Bank"
]

# Fill missing categorical values
for c in categorical_cols:
    df = df.fillna({c: "Unknown"})

# Fill missing Amount with median
amount_median = df.approxQuantile("Amount", [0.5], 0.01)[0]
df = df.fillna({"Amount": amount_median})

# Extract date features
df = df.withColumn("Transaction_Day", dayofmonth(col("Date")))
df = df.withColumn("Transaction_Month", month(col("Date")))
df = df.withColumn("Transaction_Weekday", dayofweek(col("Date")))

# Remove original Date
df = df.drop("Date")

# Remove duplicates
df = df.dropDuplicates()

# Convert target column
df = df.withColumn("Fraud", col("Fraud").cast("double"))

print("\n========== PREPROCESSING RESULTS ==========")

print("Cleaned Records:", df.count())

amount_data = df.select(
    "Fraud", "Amount"
).sample(
    False, 0.5, seed=42
).toPandas()

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=amount_data,
    x="Fraud",
    y="Amount"
)

plt.title("Transaction Amount by Fraud Status")
plt.xlabel("Fraud (0 = Legitimate, 1 = Fraudulent)")
plt.ylabel("Transaction Amount (£)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"02_transaction_amount.png"), dpi=300)
plt.show()

print("\nMissing Values:")
df.select([
    sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in df.columns
]).show(truncate=False)

print("\nFraud Distribution:")
df.groupBy("Fraud").count().show()

entry_data = df.groupBy(
    "Entry Mode", "Fraud"
).count().toPandas()

plt.figure(figsize=(9, 5))

sns.barplot(
    data=entry_data,
    x="Entry Mode",
    y="count",
    hue="Fraud",
    hue_order=[0, 1],
    palette={0: "steelblue", 1: "darkorange"}
)

plt.title("Fraud Distribution by Entry Mode")
plt.xlabel("Entry Mode")
plt.ylabel("Number of Transactions")
plt.legend(
    title="Fraud",
    labels=["Legitimate", "Fraudulent"]
)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"03_fraud_by_entry_mode.png"), dpi=300)
plt.show()

# ==================================================
# 4. FEATURE ENGINEERING
# ==================================================

numeric_cols = [
    "Time",
    "Amount",
    "Age",
    "Transaction_Day",
    "Transaction_Month",
    "Transaction_Weekday"
]

# Split dataset
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# String Indexing
indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c.replace(" ", "_") + "_Index",
        handleInvalid="keep"
    )
    for c in categorical_cols
]

# One Hot Encoding
encoder = OneHotEncoder(
    inputCols=[c.getOutputCol() for c in indexers],
    outputCols=[
        c.replace(" ", "_") + "_Encoded"
        for c in categorical_cols
    ],
    handleInvalid="keep"
)

encoded_cols = [
    c.replace(" ", "_") + "_Encoded"
    for c in categorical_cols
]

# Assemble features
assembler = VectorAssembler(
    inputCols=numeric_cols + encoded_cols,
    outputCol="features",
    handleInvalid="keep"
)

feature_pipeline = Pipeline(
    stages=indexers + [encoder, assembler]
)

feature_model = feature_pipeline.fit(train_df)

train_data = feature_model.transform(train_df)
test_data = feature_model.transform(test_df)

print("\n========== FEATURE ENGINEERING RESULTS ==========")

print("Training Records:", train_data.count())
print("Testing Records:", test_data.count())

# ==================================================
# 5. MODEL TRAINING
# ==================================================

models = {
    "Logistic Regression": LogisticRegression(
        featuresCol="features",
        labelCol="Fraud"
    ),

    "Decision Tree": DecisionTreeClassifier(
        featuresCol="features",
        labelCol="Fraud",
        seed=42
    ),

    "Random Forest": RandomForestClassifier(
        featuresCol="features",
        labelCol="Fraud",
        numTrees=100,
        seed=42
    )
}

# ==================================================
# 6. MODEL EVALUATION
# ==================================================

accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="Fraud",
    predictionCol="prediction",
    metricName="accuracy"
)

precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="Fraud",
    predictionCol="prediction",
    metricName="weightedPrecision"
)

recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="Fraud",
    predictionCol="prediction",
    metricName="weightedRecall"
)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="Fraud",
    predictionCol="prediction",
    metricName="f1"
)

auc_evaluator = BinaryClassificationEvaluator(
    labelCol="Fraud",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

results = []

print("\n========== MODEL COMPARISON ==========")

for name, algorithm in models.items():

    print("\nTraining:", name)

    model = algorithm.fit(train_data)

    predictions = model.transform(test_data)

    accuracy = accuracy_evaluator.evaluate(predictions)
    precision = precision_evaluator.evaluate(predictions)
    recall = recall_evaluator.evaluate(predictions)
    f1 = f1_evaluator.evaluate(predictions)
    auc = auc_evaluator.evaluate(predictions)

    results.append((
        name,
        accuracy,
        precision,
        recall,
        f1,
        auc
    ))

    print("\nConfusion Matrix:")
    predictions.groupBy("Fraud", "prediction").count().show()

    cm_data = predictions.select(
        "Fraud", "prediction"
    ).toPandas()

    cm = pd.crosstab(
        cm_data["Fraud"],
        cm_data["prediction"]
    ).reindex(
        index=[0.0, 1.0],
        columns=[0.0, 1.0],
        fill_value=0
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"]
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.tight_layout()

    filename = name.lower().replace(" ", "_") + "_confusion_matrix.png"

    plt.savefig(os.path.join(OUTPUT_DIR,filename), dpi=300)
    plt.show()

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
)

plt.figure(figsize=(12, 6))

results_plot = results_df.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score"
)

sns.barplot(
    data=results_plot,
    x="Model",
    y="Score",
    hue="Metric"
)

plt.title("Machine Learning Model Performance Comparison")
plt.xlabel("Machine Learning Algorithms")
plt.ylabel("Performance Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=10)
plt.legend(title="Evaluation Metrics")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"04_model_performance.png"), dpi=300)
plt.show()

# ==================================================
# 7. FINAL MODEL COMPARISON
# ==================================================

print("\n========== FINAL MODEL RESULTS ==========")

print(
    f"{'Model':<25}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1 Score':<12}"
    f"{'ROC-AUC':<12}"
)

print("-" * 85)

for result in results:
    name, accuracy, precision, recall, f1, auc = result

    print(
        f"{name:<25}"
        f"{accuracy:<12.4f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{auc:<12.4f}"
    )

# ==================================================
# 8. FINAL MODEL SELECTION
# ==================================================

# Select the model with the highest F1 Score
best_model = max(results, key=lambda x: x[4])

best_model_name = best_model[0]

print("\n========== SELECTED BEST MODEL ==========")

print("Selected Model:", best_model_name)
print("Accuracy:", round(best_model[1], 4))
print("Precision:", round(best_model[2], 4))
print("Recall:", round(best_model[3], 4))
print("F1 Score:", round(best_model[4], 4))
print("ROC-AUC:", round(best_model[5], 4))

print("\nBest Model Selected Successfully!")

# ==================================================
# 9. STOP SPARK SESSION
# ==================================================

spark.stop()