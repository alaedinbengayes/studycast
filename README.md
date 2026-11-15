# StudyCast – Exam Score Estimator

CSCI104 Introduction to AI – Final Project, Option B (University of Khorfakkan)

A Streamlit website that loads a trained scikit-learn model (`model.joblib`) and estimates a student's exam score from five inputs:
study hours per week, attendance (%), previous exam score, tutoring sessions per month and access to resources.

- **Dataset:** [Student Performance Factors – Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)
- **Model:** Linear regression in a scikit-learn pipeline (median imputer and one-hot encoder). Test MAE is 1.11 points.

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Files
| File | Purpose |
|---|---|
| `streamlit_app.py` | The website: form, validation, prediction, error handling |
| `model.joblib` | Trained pipeline exported from the Colab notebook |
| `requirements.txt` | Library versions the model was trained with (they must match) |
| `.streamlit/config.toml` | Theme colours |

Add `?test=model-failure` to the URL to simulate a model failure (used for test 4 in the report).
