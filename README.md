# StudyCast – Exam Score Estimator

CSCI104 Introduction to AI – Final Project, Option B (University of Khorfakkan)

**Live website:** https://studycast-ukf.vercel.app
**Colab notebook:** https://colab.research.google.com/github/alaedinbengayes/studycast/blob/main/notebook/Exam_Score_Model.ipynb
**Dataset:** [Student Performance Factors – Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)

The website takes five inputs (study hours per week, attendance %, previous exam score, tutoring sessions per month and access to resources) and estimates the exam score using a scikit-learn linear regression pipeline trained in the notebook. The test MAE is 1.11 points.

## How the website calls the model
`app.py` (Flask) serves `templates/index.html`. When the visitor presses **Predict Exam Score**, the page validates the inputs and sends them to `POST /api/predict`. The server loads `model.joblib` with `joblib.load`, validates the inputs again, builds a one-row pandas DataFrame with the training column names and returns `model.predict()`.

## Run locally
```bash
pip install -r requirements.txt
python app.py        # then open http://localhost:5000
```

## Files
| File | Purpose |
|---|---|
| `app.py` | Flask server: page and `/api/predict` endpoint, validation and error handling |
| `templates/index.html` | The web page: header and navigation, form, results card, client-side validation |
| `model.joblib` | Trained pipeline exported from the notebook |
| `requirements.txt` | Library versions the model was trained with |
| `notebook/` | Training notebook (outputs visible) and the Kaggle CSV |
| `streamlit_version/` | An earlier Streamlit version of the same site (not deployed) |

Add `?test=model-failure` to the URL to simulate a model failure (test 4 in the report).
