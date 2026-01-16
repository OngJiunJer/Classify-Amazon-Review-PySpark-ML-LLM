# install terminal: pip install torch transformers sentencepiece "accelerate>=0.26.0"

#Import Library
from pyspark.sql import DataFrame
from pyspark.sql.functions import concat_ws, col, lit
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd
import re
from tqdm import tqdm  # pip install tqdm

# Set up for LLM
def load_llm():
    model_name = "google/flan-t5-small" # Can change to this "google/flan-t5-small" for small scale

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_model =AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")

    # Ensure inputs are truncated to the tokenizer's max length
    tokenizer.model_max_length = 512

    llm = pipeline(
        "text2text-generation",
        model=llm_model,
        tokenizer=tokenizer,
        max_new_tokens=1   # controls how many tokens are generated
    )

    return llm

# LLM Classify Function
def llm_classify_review(llm, pdf: pd.DataFrame, text_col: str) -> pd.DataFrame:

    results = []

    for review in tqdm(pdf[text_col], desc="LLM Classifying Reviews"):
        prompt = f"""
You are a sentiment analysis assistant.

Classify the following Amazon fine food review into a star rating (1–5):
1 = very negative
2 = negative
3 = neutral
4 = positive
5 = very positive

Return ONLY one digit (1, 2, 3, 4, or 5).
Do not explain.

Review:
\"\"\"{review}\"\"\"
Rating:
"""
        
        try:
            output = llm(prompt)
            generated_text = output[0]["generated_text"]
            match = re.search(r"\b[1-5]\b", generated_text)
            rating = int(match.group()) if match else None
        except:
            rating = None

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
def llm_classify_pieline(predictions: DataFrame, col_names: list, target_column: str, spark) -> DataFrame:

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
    