# src/preprocess.py
import pandas as pd
import numpy as np

def clean_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw loan DataFrame and produce engineered features needed for training/inference.
    - Imputes missing values (LoanAmount, Credit_History, Self_Employed, Gender, Married, Dependents, Loan_Amount_Term)
    - Maps Dependents '3+' -> 3 and converts to int
    - Creates TotalIncome and Income_to_Loan_Ratio (safe from div-by-zero)
    - Caps ApplicantIncome (99th percentile) and LoanAmount (IQR method)
    - Encodes simple binary columns to 0/1 and one-hot encodes Property_Area
    Returns a new DataFrame (does not modify input).
    """
    df = df.copy()

    # ---------------------
    # 1) Basic imputations
    # ---------------------
    # Numeric imputations
    if 'LoanAmount' in df.columns:
        df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())

    if 'ApplicantIncome' in df.columns:
        df['ApplicantIncome'] = df['ApplicantIncome'].fillna(0)

    if 'CoapplicantIncome' in df.columns:
        df['CoapplicantIncome'] = df['CoapplicantIncome'].fillna(0)

    if 'Loan_Amount_Term' in df.columns:
        df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])

    # Categorical imputations (use mode or domain default)
    if 'Credit_History' in df.columns:
        if not df['Credit_History'].mode().empty:
            df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
        else:
            df['Credit_History'] = df['Credit_History'].fillna(1.0)

    if 'Self_Employed' in df.columns:
        df['Self_Employed'] = df['Self_Employed'].fillna('No')

    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])

    if 'Married' in df.columns:
        df['Married'] = df['Married'].fillna(df['Married'].mode()[0])

    if 'Dependents' in df.columns:
        df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])

    # ---------------------
    # 2) Dependents cleanup
    # ---------------------
    if 'Dependents' in df.columns:
        # Convert '3+' to '3' then to int
        df['Dependents'] = df['Dependents'].astype(str).replace('3+', '3')
        # If some non-numeric values remain, coerce and fill with mode
        df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce')
        if df['Dependents'].isnull().any():
            df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
        df['Dependents'] = df['Dependents'].astype(int)

    # ---------------------
    # 3) Engineered numeric features (after imputations)
    # ---------------------
    # Total income
    if {'ApplicantIncome', 'CoapplicantIncome'}.issubset(df.columns):
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    else:
        df['TotalIncome'] = np.nan

    # Ensure LoanAmount is not zero/NaN (we already filled with median)
    if 'LoanAmount' in df.columns:
        # replace any zero with median to avoid divide by zero
        loan_median = df['LoanAmount'].median() if not df['LoanAmount'].isnull().all() else 1.0
        df['LoanAmount'] = df['LoanAmount'].replace(0, loan_median)
        # Income to loan ratio
        df['Income_to_Loan_Ratio'] = df['TotalIncome'] / df['LoanAmount']
    else:
        df['Income_to_Loan_Ratio'] = np.nan

    # safe applicant/coapplicant ratio (applicant / coapplicant). coapplicant may be zero -> use np.where
    if {'ApplicantIncome','CoapplicantIncome'}.issubset(df.columns):
        df['Applicant_to_Coapp_Ratio'] = np.where(
            df['CoapplicantIncome'] == 0,
            df['ApplicantIncome'],  # if no coapplicant income, ratio = applicant income (or set to a big number)
            df['ApplicantIncome'] / df['CoapplicantIncome']
        )

    # ---------------------
    # 4) Outlier capping
    # ---------------------
    # ApplicantIncome: cap at 99th percentile
    if 'ApplicantIncome' in df.columns:
        cap_val = df['ApplicantIncome'].quantile(0.99)
        df.loc[df['ApplicantIncome'] > cap_val, 'ApplicantIncome'] = int(cap_val)

    # LoanAmount: cap using IQR
    if 'LoanAmount' in df.columns:
        Q1 = df['LoanAmount'].quantile(0.25)
        Q3 = df['LoanAmount'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df.loc[df['LoanAmount'] < lower, 'LoanAmount'] = lower
        df.loc[df['LoanAmount'] > upper, 'LoanAmount'] = upper

    # ---------------------
    # 5) Categorical mappings (clear & consistent)
    # ---------------------
    # Binary maps (choose convention: 1 = positive / Yes / Male)
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0}).fillna(0).astype(int)

    if 'Married' in df.columns:
        df['Married'] = df['Married'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    if 'Education' in df.columns:
        df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0}).fillna(0).astype(int)

    if 'Self_Employed' in df.columns:
        df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # ---------------------
    # 6) One-hot encode Property_Area (keeps deterministic columns)
    # ---------------------
    if 'Property_Area' in df.columns:
        prop_dummies = pd.get_dummies(df['Property_Area'], prefix='Property')
        # ensure consistent columns appear even if some categories are missing
        for col in ['Property_Rural','Property_Semiurban','Property_Urban']:
            if col not in prop_dummies.columns:
                prop_dummies[col] = 0
        # attach and drop original
        df = pd.concat([df.drop(columns=['Property_Area']), prop_dummies[['Property_Rural','Property_Semiurban','Property_Urban']]], axis=1)
        # cast to int
        df['Property_Rural'] = df['Property_Rural'].astype(int)
        df['Property_Semiurban'] = df['Property_Semiurban'].astype(int)
        df['Property_Urban'] = df['Property_Urban'].astype(int)

    # ---------------------
    # 7) Target mapping if present (optional)
    # ---------------------
    if 'Loan_Status' in df.columns:
        # Map 'Y' -> 1, 'N' -> 0. If you used a different mapping before, keep consistent.
        df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    # final: return cleaned df
    return df
