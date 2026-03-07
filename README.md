# Customer Churn Prediction

ML project that predicts which customers are likely to cancel their subscription. Uses Random Forest and has a Streamlit dashboard for easy predictions.

## What it does
- Predicts churn probability for each customer
- Groups customers into High/Medium/Low risk
- Shows key business metrics and charts
- Lets you input customer info and get instant predictions

## Quick Start

```bash
# clone and setup
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# train the model
python train_model.py

# launch dashboard
streamlit run dashboard.py
```

## Model Results

- Accuracy: ~74%
- ROC-AUC: ~71%
- Main churn drivers: monthly charges, tenure, contract type

## Files

- `train_model.py` - trains the model
- `dashboard.py` - streamlit web app
- `src/model.py` - ChurnModel class
- `src/preprocessing.py` - data cleaning
- `src/analysis.py` - business metrics
- `src/visualizations.py` - charts

## Tech Stack

- Python 3.8+
- scikit-learn
- pandas, numpy
- streamlit
- plotly

## Notes

The dataset is synthetic. To generate new data run `python generate_realistic_data.py`.
