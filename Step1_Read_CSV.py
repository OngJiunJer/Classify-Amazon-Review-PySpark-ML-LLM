# Import Library
from pyspark.sql import DataFrame

# -------------------------------
# Spark Session Initialization
# -------------------------------
# SparkSession.builder        : Entry point to configure Spark
# appName("Day1_Read_CSV")    : Identifies the Spark application in Spark UI and logs
# getOrCreate()               : Returns an existing SparkSession or creates a new one
#
# inferSchema=True            : Automatically detects column data types instead of
#                               treating all values as strings

# Create Read CSV Function from 
def Read_CSV_Dataset(spark, file_path: str) -> DataFrame:
    df_pyspark = spark.read.csv(
        file_path,
        header=True,       # First row as column names
        inferSchema=True,  # Automatically detect column types
        multiLine=True,    # Handles multi-line text in a cell
        escape='"',        # Escape character for quotes inside text
        quote='"'          # Quote character for text columns
    )

    df_pyspark.show(5)
    df_pyspark.printSchema()
    

    return df_pyspark