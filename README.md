# Classify Amazon Review Pyspark, ML, and LLM
This project classifies Amazon Fine Food Reviews into 5-star ratings using a hybrid approach of Machine Learning (ML) and Large Language Models (LLMs).
The workflow first uses TF-IDF + Logistic Regression to classify reviews, then reclassifies low-confidence predictions using LLMs for improved performance.

## Main.py
- Initializes a Spark session
- Run read csv function
- Runs the preprocessing pipeline (cleaning, tokenization, stemming)
- Trains the ML pipeline (TF-IDF + Logistic Regression)
- Evaluates ML performance
- Uses LLM to reclassify low-confidence ML predictions
- Evaluates LLM-enhanced performance

## Step1_Read_CSV
- Reads the Reviews.csv dataset

## Step2_Preprocessing.py
- Remove nulls & duplicates
- Trim whitespace, lowercase text
- Tokenize, remove stopwords, stem tokens
- Full preprocessing pipeline applied before training

## Step3_Train_ML_LG.py
- TF-IDF feature embedding
- Train Logistic Regression classifier
- Handle class imbalance with under-sampling
- Evaluate overall and per-class accuracy

## Step4_Train_Bert_Multi_Sentiment.py & Step4_Train_LLM_Flan_T5.py
- Apply LLMs to low-confidence ML predictions
- BERT Multi-Sentiment: balanced performance across all classes
- Flan-T5: strong performance on extreme sentiment (1 & 5 stars)

## ML & LLM Performance
ML Pipeline (TF-IDF + Logistic Regression):
  - Overall Metrics:
    - Accuracy: 0.6924
    - Weighted F1 Score: 0.6370
    - Weighted Precision: 0.6313
    - Weighted Recall: 0.6924
  - Per-Class Accuracy:
    | Score | Accuracy |
    | ----- | -------- |
    | 1     | 0.528    |
    | 2     | 0.088    |
    | 3     | 0.174    |
    | 4     | 0.149    |
    | 5     | 0.952    |

Flan-T5 LLM:
  - Overall Metrics:
    - Accuracy: 0.6569
    - Weighted F1 Score: 0.6253
    - Weighted Precision: 0.6731
    - Weighted Recall: 0.6569
  - Per-Class Accuracy:
    | Score | Accuracy |
    | ----- | -------- |
    | 1     | 0.948    |
    | 2     | 0.002    |
    | 3     | 0.022    |
    | 4     | 0.220    |
    | 5     | 0.845    |

BERT-Multi-Sentiment LLM:
  - Overall Metrics:
    - Accuracy: 0.6710
    - Weighted F1 Score: 0.6933
    - Weighted Precision: 0.7280
    - Weighted Recall: 0.6710
  - Per-Class Accuracy:
    | Score | Accuracy |
    | ----- | -------- |
    | 1     | 0.567    |
    | 2     | 0.502    |
    | 3     | 0.470    |
    | 4     | 0.462    |
    | 5     | 0.771    |

Conclucion:
  - ML: Fast, effective for clear cases (5-star reviews), struggles with mid-range scores.
  - Flan-T5 LLM: Excellent for extreme sentiment detection (1 & 5 stars). However, realy bad on predict on mid-range scores.
  - BERT LLM: Balanced performance across all ratings, improves low-confidence ML predictions. However, not able to maintain high accuracy prediction on score 1 and 5.

## cLimitations:
- Dataset is heavily skewed toward 5-star reviews, making middle ratings harder to predict.
- LLMs can misclassify low-confidence or nuanced reviews.
- ML only uses review text, ignoring helpful metadata.
- LLMs are computationally expensive for large datasets.

## Future Improvements:
- Handle class imbalance with SMOTE or weighted loss.
- Add metadata features like review length, time, or product info.
- Combine ML + LLM predictions in an ensemble.
- Test other transformer models or fine-tune hyperparameters.
- Make the pipeline real-time deployable for new reviews.


