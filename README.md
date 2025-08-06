# 🏦 Loan Default Prediction System

This project predicts the likelihood of a loan applicant defaulting, using machine learning on structured/tabular financial data. It also includes an interactive Streamlit web app for real-time predictions.

---

## 🚀 Features

- Exploratory Data Analysis (EDA) and visualization
- Data preprocessing and feature engineering
- Multiple ML models (Logistic Regression, Random Forest, XGBoost)
- Model evaluation and explainability (e.g., SHAP)
- Interactive Streamlit web app for live predictions
- Well-organized codebase with modular structure

---

## 📁 Project Structure
loan-default-predictor/
│
├── app.py # Streamlit application
├── config.yaml # Configuration settings (optional)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
├── data/
│ ├── raw/ # Original datasets
│ └── processed/ # Cleaned and processed data
│
├── notebooks/
│ └── EDA_and_Modeling.ipynb # Analysis notebook
│
├── models/
│ └── model.pkl # Serialized trained model
│
├── src/
│ ├── preprocess.py # Data preprocessing
│ ├── train_model.py # Model training
│ └── utils.py # Helper functions
│
└── logs/ # System logs (optional)

text

---

## 🛠️ Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/loan-default-predictor.git
   cd loan-default-predictor
2. **Create and activate virtual environment:**:
   ```bash
   python -m venv venv
   Windows: 
   venv\Scripts\activate

   Mac/Linux:
   source venv/bin/activate
   
3. **Install dependencies:**:
pip install -r requirements.txt
3. **Run the application:**:
streamlit run app.py

**📊 Example Use Case**
The model processes financial inputs (income, loan amount, credit history, etc.) and returns:

🔴 "Will Default": High risk of non-payment

🟢 "Will Not Default": Low risk, safer to approve

**🤝 Credits**
Developer: Ali Sleiman
Inspiration: Real-world financial risk modeling systems
Technologies: Python, Scikit-learn, XGBoost, Streamlit

**📌 Disclaimer**
⚠️ Important: This is an educational project only. Not intended for real financial decision-making.

