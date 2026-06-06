# 🧬 Gender & Age Group Classifier

A machine learning web app that predicts **Gender** and **Age Group** based on real-world CDC health measurements.

## 🌐 Live App
👉 [https://gender-age-classifier.streamlit.app](https://gender-age-classifier.streamlit.app)

## 📊 Dataset
- **Source:** CDC NHANES (National Health and Nutrition Examination Survey)
- **Size:** 2,278 records with 12 health features
- **Features:** BMI, Glucose, Insulin, Height, Weight, Waist, Arm measurements, etc.

## 🎯 Models
| Model | Algorithm | Accuracy |
|---|---|---|
| Gender Classifier | Random Forest | 86.4% |
| Age Group Classifier | Random Forest + SMOTE | 84.4% |

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, SMOTE
- Streamlit
- Deployed on Streamlit Cloud

## 🚀 How to Run Locally
```bash
git clone https://github.com/aayitijhyaganguly-ai/Gender-age-classifier.git
cd Gender-age-classifier
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure
```
├── Data/
├── Models/
├── notebook.ipynb
├── app.py
└── requirements.txt
```
