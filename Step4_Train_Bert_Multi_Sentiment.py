# install terminal: pip install torch transformers sentencepiece "accelerate>=0.26.0"

#Import Library
from pyspark.sql import DataFrame
from pyspark.sql.functions import concat_ws, col, lit
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import re
from tqdm import tqdm  # pip install tqdm

# Set up for LLM
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    llm = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    return llm

# LLM Classify Function
def llm_classify_review(llm, pdf: pd.DataFrame, text_col: str) -> pd.DataFrame:
    results = []

    for review in tqdm(pdf[text_col], desc="LLM Classifying Reviews"):
        try:
            output = llm(review)  # raw text
            label = output[0]["label"]  # e.g., '3 stars'
            match = re.search(r"\d", label)
            rating = int(match.group()) if match else None
        except:
            rating = None
            score = 0.0

        results.append(rating)

    pdf["llm_prediction"] = results
    return pdf



# Seperate High & Low Confidence Prediction
def split_high_low_confidence(predictions: DataFrame) -> tuple[DataFrame, DataFrame]:
    predictions_high_confidence = predictions.filter(col("confidence") >= 0.80)
    predictions_low_confidence = predictions.filter(col("confidence") < 0.80)

    # Count rows
    high_count = predictions_high_confidence.count()
    low_count = predictions_low_confidence.count()

    print(f"High confidence rows: {high_count}")
    print(f"Low confidence rows: {low_count}")

    return predictions_high_confidence, predictions_low_confidence

# Combine Multiple Text Col
def add_combined_text_col(df: DataFrame, col_names: list) -> DataFrame:
    labeled_cols = []

    for c in col_names:
        labeled_cols.append(lit(f"{c}:\n"))
        labeled_cols.append(col(c))
        labeled_cols.append(lit("\n\n"))

    return df.withColumn(
        "text_col",
        concat_ws("", *labeled_cols)
    )


# LLM Pipeline to Classify Low Confidence Prediction From ML
def llm_classify_pipeline(predictions: DataFrame, col_names: list, target_column: str, spark) -> DataFrame:

    if len(col_names) > 1:
        predictions = add_combined_text_col(predictions, col_names)
    else:
        predictions = predictions.withColumnRenamed(col_names[0], "text_col")
    

    # Keep only necessary columns
    predictions_small = predictions.select(
        "text_col",
        target_column,
        "ml_Prediction",
        "confidence"
    )

    high_df, low_df = split_high_low_confidence(predictions_small)

    # Convert LOW confidence to Pandas
    low_pd = low_df.toPandas()

    if len(low_pd) > 0:
        llm = load_llm()
        low_pd = llm_classify_review(llm, low_pd, "text_col")
        low_spark = spark.createDataFrame(low_pd)
    else:
        low_spark = low_df.withColumn("llm_prediction", col("ml_Prediction"))

    high_df = high_df.withColumn(
        "llm_prediction",
        col("ml_Prediction")
    )

    return high_df.unionByName(low_spark)
    