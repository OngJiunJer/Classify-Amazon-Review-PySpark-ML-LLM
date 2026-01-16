# Import Library
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, when, trim, lower, udf, size
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.sql.types import ArrayType, StringType

import nltk
from nltk.stem import PorterStemmer
# Run once: nltk.download('punkt')

######################
# Remove Null Data
######################
def remove_Na(df_pyspark: DataFrame) -> DataFrame:
    # Calculate null counts for all columns
    null_counts = df_pyspark.select([count(when(col(c).isNull(), c)).alias(c) for c in df_pyspark.columns]).collect()[0].asDict()

    total = df_pyspark.count()

    # Loop each col na count
    for c, null_count in null_counts.items():
        ratio = null_count / total

        if ratio > 0.7: # Drop Column
            df_pyspark = df_pyspark.drop(c)
            print(f"Drop column '{c}' due to too many null values (ratio={ratio:.2f})")
        elif null_count > 0: # Drop Row
            df_pyspark = df_pyspark.na.drop(subset=[c])
            print(f"Removed nulls from column '{c}' (ratio={ratio:.2f})")
        else:
             print(f"Keep column '{c}' since no na value appear.")

    return df_pyspark

######################
# Remove Duplicate data
######################
def remove_duplicate(df_pyspark: DataFrame) -> DataFrame:
    df_pyspark = df_pyspark.dropDuplicates()
    return df_pyspark

######################
# Remove Extra Space
######################
def remove_extra_space(df_pyspark: DataFrame, col_names: list) -> DataFrame:
    for c in col_names:
        df_pyspark = df_pyspark.withColumn(c, trim(col(c)))

    return df_pyspark


######################
# Convert to Lowercase
######################
def convert_lowercase(df_pyspark: DataFrame, col_names: list) -> DataFrame:
    for c in col_names:
        df_pyspark = df_pyspark.withColumn(c, lower(col(c)))

    return df_pyspark

######################
# Tokenization (Specific for ML)
######################
def tokenize(df_pyspark: DataFrame, col_names: list) -> DataFrame:
    for c in col_names:
        tokenizer = RegexTokenizer(inputCol=c, outputCol=f"{c}_tokens", pattern="\\W")
        df_pyspark = tokenizer.transform(df_pyspark)

    return df_pyspark

######################
# Remove Stopword (Specific for ML)
######################
def remove_stop_word(df_pyspark: DataFrame, col_names: list) -> DataFrame:
    for c in col_names:
        remover = StopWordsRemover(inputCol=f"{c}_tokens", outputCol=f"{c}_tokens_clean")
        df_pyspark = remover.transform(df_pyspark)

    return df_pyspark

######################
# Remove text which are too short
######################
def remove_short_text(df_pyspark: DataFrame, col_names: list, size_range: int) -> DataFrame:
    for c in col_names:
        df_pyspark = df_pyspark.filter(size(col(f"{c}_tokens")) >= size_range)

    return df_pyspark

######################
# Stem (Specific for ML) 
######################
# Define PorterStemmer
stemmer = PorterStemmer()

# Steming each token function
def stem_tokens(tokens):
    if tokens is None:
        return []
    return [stemmer.stem(token) for token in tokens]

# user define function made python function (stem_tokens) able to use in pyspark feature (withColumn)
stem_udf = udf(stem_tokens, ArrayType(StringType())) 

# Function to stem multiple tokenized columns
def stemming(df_pyspark: DataFrame, col_names: list) -> DataFrame:
    for c in col_names:
        df_pyspark = df_pyspark.withColumn(f"{c}_tokens_stemmed", stem_udf(col(f"{c}_tokens_clean")))
    return df_pyspark

######################
# Convert Str & Int
######################
def convert_str_int(df_pyspark: DataFrame, col_name: str, convert_to: str) -> DataFrame:
    if convert_to == "int":
        df_pyspark = df_pyspark.withColumn(col_name, col(col_name).cast("int"))
        print(f"convert {col_name} column to int data type.")
    elif convert_to == "str":
        df_pyspark = df_pyspark.withColumn(col_name, col(col_name).cast("string"))
        print(f"convert {col_name} column to str data type.")
    else:
        print("Input Wrong Data.")

    return df_pyspark

######################
# Remove Column
######################
def remove_column(df_pyspark: DataFrame, col_names: list) -> DataFrame:
    df_temp = df_pyspark

    for col in col_names:
        df_temp = df_temp.drop(col)
        print(f"remove {col} column.")

    return df_temp

######################
# Preprocessing Pipeline
######################
def preprocessing_pipeline(df_pyspark: DataFrame, col_names: list) -> DataFrame:
    """
    Full preprocessing pipeline for Amazon Review dataset
    """
    # Remove Nulls & Duplicates
    print("##### RUN remove_Na & remove_duplicate FUNCTION #####")
    df_pyspark = remove_Na(df_pyspark)
    df_pyspark = remove_duplicate(df_pyspark)
    print("##### FINISH remove_Na & remove_duplicate FUNCTION #####")

    # Remove Extra Spaces
    print("##### RUN remove_extra_space FUNCTION #####")
    df_pyspark = remove_extra_space(df_pyspark, col_names)
    print("##### FINISH remove_extra_space FUNCTION #####")

    # Lowercase
    print("##### RUN convert_lowercase FUNCTION #####")
    df_pyspark = convert_lowercase(df_pyspark, col_names)
    print("##### FINISH convert_lowercase FUNCTION #####")

    # Tokenization
    print("##### RUN tokenize FUNCTION #####")
    df_pyspark = tokenize(df_pyspark, col_names)
    print("##### FINISH tokenize FUNCTION #####")

    # Remove Stopwords
    print("##### RUN remove_stop_word FUNCTION #####")
    df_pyspark = remove_stop_word(df_pyspark, col_names)
    print("##### FINISH remove_stop_word FUNCTION #####")

    # Remove Short Text
    print("##### RUN remove_short_text FUNCTION #####")
    df_pyspark = remove_short_text(df_pyspark, col_names, 20)
    print("##### FINISH remove_short_text FUNCTION #####")

    # Stemming
    print("##### RUN stemming FUNCTION #####")
    df_pyspark = stemming(df_pyspark, col_names)
    print("##### FINISH stemming FUNCTION #####")

    return df_pyspark


