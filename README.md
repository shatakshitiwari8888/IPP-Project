# Credit Risk and Loan Default Prediction

**Course:** MBA Introduction to Python Programming  
**Methodology:** Multiple Logistic Regression & Econometric Credit Underwriting  
**Dataset:** LendingClub Historical Accepted Loans (2007–2018 Q4)  
**Target Variable:** Probability of Default ($PD \in [0, 1]$)  
**Decision Platform:** Interactive Streamlit Analytics Dashboard (`project.py`)

---

## 1. Project Overview & Methodology

This project models the **Probability of Default (PD)** for unsecured retail loan applicants using Multiple Logistic Regression on LendingClub's loan portfolio. The model processes borrower financial characteristics—Annual Income, Debt-to-Income (DTI), FICO credit score, Loan Amount, and an engineered Loan-to-Income (LTI) leverage ratio—to estimate the log-odds of loan delinquency.

To conform with institutional credit risk governance (Basel II/III Internal Ratings-Based standards):
1. **Look-Ahead Bias Elimination:** Active ongoing loans (`Current`, `In Grace Period`) are removed from training. Models are trained strictly on resolved loan contracts (`Fully Paid` vs. `Charged Off` / `Default`).
2. **Data Leakage Prevention:** Feature standardization (`StandardScaler`) is fitted strictly on the 80% training partition.
3. **Balanced Class Weighting:** `class_weight='balanced'` is utilized to heavily penalize False Negatives (missed defaults), overcoming the ~3:1 class imbalance.
4. **Calibrated Supervisory Grading:** Predicted default probabilities are converted into transparent, monotonically ordered **Credit Grades (A to F)**.

---

## 2. Directory Structure

```text
Credit_Risk_Project/
├── data/
│   ├── accepted_2007_to_2018Q4.csv.gz  # Compressed LendingClub loan dataset
│   └── accepted_2007_2018Q4.csv        # Uncompressed dataset
├── notebooks/
│   └── credit_risk_analysis.ipynb      # Step-by-step Jupyter Notebook analysis
├── models/
│   ├── credit_risk_pipeline.pkl        # Serialized preprocessing & logistic model
│   └── model_metadata.json             # Model coefficients, statistics, and cutoffs
├── outputs/
│   ├── coefficients_table.csv          # Standardized Betas and Odds Ratios
│   ├── grade_calibration_table.csv     # Monotonic grade validation table
│   └── test_set_predictions.csv       # Holdout evaluation predictions
├── project.py                          # Enterprise Streamlit Web Dashboard
├── requirements.txt                    # Python library dependencies
└── README.md                           # Documentation & execution guide
```

---

## 3. Local Installation & Setup (Windows Anaconda / macOS / Linux)

### Step 1: Open Terminal / Anaconda Prompt
```bash
# Clone or navigate to the project directory
cd Credit_Risk_Project
```

### Step 2: Create and Activate Virtual Environment
```bash
# Using Conda:
conda create -n credit_risk python=3.10 -y
conda activate credit_risk

# OR using standard Python venv:
python -m venv venv
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Launch the Streamlit Dashboard
```bash
streamlit run project.py
```
*The dashboard will automatically open in your default browser at `http://localhost:8501`.*

### Step 5: Launch the Jupyter Notebook
```bash
jupyter notebook notebooks/credit_risk_analysis.ipynb
```

---

## 4. Dashboard Architecture (`project.py`)

The Streamlit application features five institutional-grade modules:
1. **Portfolio Overview:** Executive KPI cards (analyzed loans, volume, settlement rate, observed default rate) and terminal outcome distribution charts.
2. **Risk Analysis:** Interactive Plotly bar charts and correlation heatmaps displaying default rates segmented by FICO score, DTI, and Loan Amount buckets with explicit sample sizes.
3. **Model Performance:** ROC-AUC score ($0.693$), Precision, Recall, F1 score, Confusion Matrix, and ROC discrimination curve.
4. **Borrower Risk Assessment:** Real-time loan decisioning screen. Calculates LTI dynamically, standardizes inputs, estimates PD, and assigns a supervisory Credit Grade (A to F).
5. **Project Insights:** Standardized Beta coefficients, Odds Ratios ($e^\beta$), plain-English interpretations, and model risk management (MRM) limitations.

---

## 5. Technical Specifications & Programmatic Concepts Demonstrated

- **Variables & Numeric Types:** Integers (FICO, counts), floats (PD, DTI, LTI), and booleans.
- **String Handling:** Stripping whitespace, standardizing categorical status labels, and formatting currency.
- **Conditionals (if-elif-else):** Multi-branch credit grading logic and boundary enforcement.
- **Loops & Comprehensions:** Iterating through borrower profiles, coefficient dictionaries, and threshold ranges.
- **Modular Functions:** Reusable preprocessing, prediction, and scoring methods with type hints.
- **Exception Handling (try-except):** Defensive file I/O and handling missing values.
- **Vectorized Math & NumPy:** Matrix multiplication, sigmoid activation, and quantile calculations.
- **Pandas DataFrames:** Complete-case filtering (.dropna()), groupbys, quantile clipping, and frequency aggregation.
- **Machine Learning Pipeline:** `StandardScaler` + `LogisticRegression(class_weight='balanced')`.

---

## 6. Technical Methodology & Risk Governance FAQ

### Q1: Why did you use `class_weight='balanced'` in Logistic Regression?
> **Answer:** In retail lending, default is an asymmetric minority event. An unweighted model achieves high accuracy simply by predicting that nobody defaults—which would lead to severe financial write-offs for a lending institution. Setting `class_weight='balanced'` automatically weights classes inversely proportional to their frequencies, forcing the algorithm to penalize False Negatives (missed defaults) and prioritize recall.

### Q2: Why are ongoing "Current" loans dropped from the dataset?
> **Answer:** To eliminate **Look-Ahead Bias**. A loan marked "Current" today has not reached terminal maturity; the borrower could default in subsequent quarters. Labeling active loans as safe ($0$) introduces false negatives. Basel II/III standards mandate that default models must be trained strictly on resolved historical contracts.

### Q3: Why must `StandardScaler` be fit strictly on the training partition?
> **Answer:** To prevent **Data Leakage**. In live production, future loan applicants do not yet exist; their empirical mean and standard deviation cannot be known. Fitting on the full dataset leaks test distribution parameters into training, producing unrealistically optimistic performance metrics.

### Q4: How do you interpret the Odds Ratio ($e^\beta$) on FICO score?
> **Answer:** For our standardized FICO score, Beta = $-0.448$, yielding an Odds Ratio of $e^{-0.448} \approx 0.639$. Holding all other financial variables constant, every 1-standard-deviation increase in a borrower's FICO score (~34 points) reduces the odds of default by **36.1%** ($1 - 0.639 = 0.361$).
