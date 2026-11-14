# Travel & Tourism Recommendation System Using Machine Learning

## 1. Project Overview

The Travel & Tourism Recommendation System is a Machine Learning-based project designed to help users discover similar tourist destinations across India.

The system recommends tourist attractions, predicts expected Google review ratings and classifies attractions into High Rating and Normal Rating categories.

## 2. Project Objectives

- Recommend similar tourist destinations.
- Predict expected Google review ratings.
- Classify tourist attractions based on their ratings.
- Provide an interactive web application for users.

## 3. Dataset Information

**Dataset:** Indian Tourist Attractions Dataset

- Total records: 321
- Original columns: 16
- States and Union Territories: 32
- Tourist attraction types: 66
- Geographical zones: 6

## 4. Machine Learning Algorithms

### Content-Based Recommendation System
- TF-IDF
- Cosine Similarity

Used to recommend similar tourist destinations based on their features.

### Random Forest Regression
Used to predict expected Google review ratings.

### Logistic Regression
Used to classify tourist attractions into:
- High Rating (4.5 and above)
- Normal Rating (below 4.5)

## 5. Model Performance

### Random Forest Regression

| Metric | Result |
|---|---:|
| MAE | 0.122 |
| MSE | 0.023 |
| RMSE | 0.153 |
| R² Score | 0.366 |

### Logistic Regression

| Metric | Result |
|---|---:|
| Accuracy | 72.3% |
| Precision | 75% |
| Recall | 85.7% |
| F1 Score | 80% |

## 6. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TF-IDF
- Cosine Similarity
- Joblib
- Streamlit

## 7. Project Structure

```text
Travel_Tourism_Recommendation_System/
│
├── dataset/
│   ├── Indian_Tourist_Attractions_Data.xlsx
│   └── Travel_Destinations_Cleaned.csv
│
├── notebooks/
│   └── 01_Project_Into.ipynb
│
├── models/
│   ├── tourism_recommendation_model.pkl
│   └── tourism_prediction_models.pkl
│
├── streamlit_app/
│   └── app.py
│
├── reports/
│
├── README.md
└── .gitignore
```

## 8. Installation and Execution

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit openpyxl
```

Run the Streamlit application from the project root directory:

```bash
python -m streamlit run streamlit_app/app.py
```

The application will open in your browser at:

`http://localhost:8501`

## 9. Application Modules

1. Home Dashboard
2. Destination Recommendations
3. Rating Prediction
4. About Project

## 10. Project Workflow

1. Project Understanding
2. Data Collection
3. Data Cleaning and Preprocessing
4. Exploratory Data Analysis
5. Machine Learning Model Development
6. Model Evaluation
7. Model Saving Using Joblib
8. Streamlit Application Development
9. Testing and Deployment

## 11. Future Enhancements

- Add more tourist destinations.
-# Travel-Tourism-Recommendation-System
Travel and Tourism Recommendation System using Machine Learning
