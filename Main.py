# Install 3.5.0 Spark because in PySpark 4.x, the class AccumulatorUnixServer always tries to inherit from socketserver.UnixStreamServer, even on Windows.: 
# pip uninstall pyspark 
# pip install pyspark==3.5.0
#Instal nltk: pip install nltk

# Import Library
from pyspark.sql import SparkSession

# Import Function
from Step1_Read_CSV import Read_CSV_Dataset
from Step2_Preprocessing import (
    convert_str_int,
    remove_column,
    preprocessing_pipeline
)
from Step3_Train_ML_LG import evaluate_classfication_report, train_lr_pipeline, evaluate_accuracy_each_class
from Step4_Train_LLM import llm_classify_pipeline

# Main
def main():
    # Craete a Spark Enviroment
    spark = SparkSession.builder.appName('Amazon Review').getOrCreate()

    # Read CSV Data
    df_pyspark = Read_CSV_Dataset(spark, "Reviews.csv")
    
    # Preprocessing Data
    col_column = ["text"] # Feature Column
    target_column = "Score"
    unwanted_column = ["Id", "ProductId", "UserId", "ProfileName", "HelpfulnessNumerator", "HelpfulnessDenominator", "Time", "Summary"]

    # Remove unwanted Col
    df_preprocessed = remove_column(df_pyspark, unwanted_column)

    # Ensure Score Target Variable is Int Data Type
    df_preprocessed = convert_str_int(df_preprocessed, target_column, "int")
    
    # Run All the preprocessing step
    df_preprocessed = preprocessing_pipeline(df_preprocessed, col_column)

    # Show the preprocessed Dataset
    df_preprocessed.show()

    # Run Train ML Pipeline
    prediction_result = train_lr_pipeline(df_preprocessed, col_column, target_column)

    # Evalue ML Performance
    evaluate_classfication_report(prediction_result, target_column, "ml_Prediction")
    evaluate_accuracy_each_class(prediction_result, target_column, "ml_Prediction")

    # Run Train ML Pipeline
    predictions_final = llm_classify_pipeline(prediction_result, col_column, target_column, spark)

    # Evalue ML Performance
    evaluate_classfication_report(predictions_final, target_column, "llm_prediction")
    evaluate_accuracy_each_class(predictions_final, target_column, "llm_prediction")

    spark.stop() # Stop pyspark

    

if __name__ == "__main__":
    main()

