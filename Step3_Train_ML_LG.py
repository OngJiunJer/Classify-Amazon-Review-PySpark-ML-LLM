# ML: TF-IDF + Logistic Regression
#Import Library
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, array_max, count, when
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import HashingTF, IDF, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

############################################
# Data Splittin
############################################
def data_splitting(df_preprocessed: DataFrame, Train_Ratio: float) -> DataFrame:
    train_df, test_df = df_preprocessed.randomSplit([Train_Ratio, 1-Train_Ratio], seed=42)
    return train_df, test_df


############################################
# Check Class Imbalance (Handle Imbalance Data)
############################################
def check_imbalance(train_df: DataFrame, target_column: str, imbalance_ratio_threshold: int) -> DataFrame:

    # Colect Count From each Class
    class_counts_df = train_df.groupBy(target_column).count().collect()
    class_counts = {row[target_column]: row['count'] for row in class_counts_df}

    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count

    if imbalance_ratio >= imbalance_ratio_threshold:
        print(f"Start random under sampling... (imbalance ratio: {imbalance_ratio:.2f})")
        
        # calculated fraction for each class
        fractions = {c: min_count / count for c, count in class_counts.items()}

        # use SampleBy in order to made sure each class have same count of sample
        balanced_df = train_df.sampleBy(target_column, fractions, seed=42)

        print("Balanced dataset class counts:")
        balanced_df.groupBy(target_column).count().show()
        
        return balanced_df

    return train_df



############################################
# Embed (TF-IDF)
############################################
def embed_TF_IDF(df: DataFrame, col_names: list) -> DataFrame:
    for c in col_names:
        hashingTF = HashingTF(inputCol=f"{c}_tokens_stemmed", outputCol=f"{c}_tf", numFeatures=2000)
        df = hashingTF.transform(df)
        
        idf = IDF(inputCol=f"{c}_tf", outputCol=f"{c}_tf_idf")
        idf_model = idf.fit(df)        # <-- Must fit first
        df = idf_model.transform(df)
    
    # Drop temporary columns after loop
    temp_cols = [f"{c}_tokens_stemmed" for c in col_names] + [f"{c}_tf" for c in col_names]
    df = df.drop(*temp_cols)

    return df


############################################
# Train & Prediction Logistic Regression 
############################################
def train_lr(train_df: DataFrame, test_df: DataFrame, col_names: list, target_column: str):
    feature_cols = [f"{c}_tf_idf" for c in col_names]  # TF-IDF columns
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_df = assembler.transform(train_df)
    test_df = assembler.transform(test_df)

    lr = LogisticRegression(featuresCol="features", labelCol=target_column,  predictionCol="ml_Prediction", maxIter=20)
    lr_model = lr.fit(train_df)
    predictions = lr_model.transform(test_df)
    predictions = predictions.withColumn("confidence",  array_max(vector_to_array(col("probability"))))
    
    return predictions


############################################
# Model Evaluation
############################################
def evaluate_classfication_report(predictions: DataFrame, target_column: str, prediction_column: str):
    print("##### RUN evaluate_classfication_report FUNCTION #####")
    # Overall matric
    evaluator_acc = MulticlassClassificationEvaluator(labelCol=target_column, predictionCol=prediction_column, metricName="accuracy")
    accuracy = evaluator_acc.evaluate(predictions)

    evaluator_f1 = MulticlassClassificationEvaluator(labelCol=target_column, predictionCol=prediction_column, metricName="f1")
    f1 = evaluator_f1.evaluate(predictions)

    evaluator_precision = MulticlassClassificationEvaluator(labelCol=target_column, predictionCol=prediction_column, metricName="weightedPrecision")
    precision = evaluator_precision.evaluate(predictions)

    evaluator_recall = MulticlassClassificationEvaluator(labelCol=target_column, predictionCol=prediction_column, metricName="weightedRecall")
    recall = evaluator_recall.evaluate(predictions)

    print("=== Overall Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}\n")
    print("##### FINISH evaluate_classfication_report FUNCTION #####")


def evaluate_accuracy_each_class(predictions: DataFrame, target_column: str, prediction_column: str):
    """
    Calculate accuracy for each class in the predictions.
    Accuracy = (# correct predictions for class) / (# total instances of class)
    """
    # Group by the target class
    class_accuracy = (
        predictions.groupBy(target_column)
        .agg(
            count("*").alias("total"),  # total rows for this class
            count(when(col(target_column) == col(prediction_column), True)).alias("correct")  # correct predictions
        )
        .withColumn("accuracy", col("correct") / col("total"))
    )

    class_accuracy.show(truncate=False)



############################################
# Train ML Pipeline
############################################
def train_lr_pipeline(
    df_pyspark,                # Spark DataFrame
    col_names,                  # list of columns to embed
    target_column,              # label column
    train_ratio=0.8,            # train/test split ratio
    imbalance_ratio_threshold=20 # threshold for under-sampling
):
    # Keep column needed to avoid too much unecassary column input function
    df_pyspark_small = df_pyspark.select(*([col(target_column)] + [col(f"{c}") for c in col_names] + [col(f"{c}_tokens_stemmed") for c in col_names]))

    # Split data
    print("##### RUN data_splitting FUNCTION #####")
    train_df, test_df = data_splitting(df_pyspark_small, train_ratio)
    print("##### FINISH data_splitting FUNCTION #####")

    # Handle class imbalance (under-sampling if needed)
    print("##### RUN check_imbalance FUNCTION #####")
    train_df = check_imbalance(train_df, target_column, imbalance_ratio_threshold)
    print("##### FINISH check_imbalance FUNCTION #####")

    # TF-IDF embedding for specified columns
    print("##### RUN embed_TF_IDF FUNCTION #####")
    train_df = embed_TF_IDF(train_df, col_names)
    test_df = embed_TF_IDF(test_df, col_names)  # also embed test set
    print("##### FINISH embed_TF_IDF FUNCTION #####")

    # Train Logistic Regression
    print("##### RUN train_lr FUNCTION #####")
    prediction_result = train_lr(train_df, test_df, col_names, target_column)
    print("##### FINISH train_lr FUNCTION #####")

    return prediction_result