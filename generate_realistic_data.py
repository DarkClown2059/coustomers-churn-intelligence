# generates synthetic customer data with realistic churn patterns
# typical telecom churn rate is around 15-25%

import pandas as pd
import numpy as np

np.random.seed(42)

NUM_CUSTOMERS = 2000
TARGET_CHURN_RATE = 0.18  # aiming for ~18%

print("generating customer data...")

# basic customer attributes
data = {
    'CustomerID': range(1, NUM_CUSTOMERS + 1),
    'Age': np.random.normal(42, 12, NUM_CUSTOMERS).clip(18, 80).astype(int),
    'Gender': np.random.choice(['Male', 'Female'], NUM_CUSTOMERS),
    'Tenure': np.random.exponential(24, NUM_CUSTOMERS).clip(0, 72).astype(int),
    'ContractType': np.random.choice(
        ['Month-to-Month', 'One-Year', 'Two-Year'], 
        NUM_CUSTOMERS, 
        p=[0.55, 0.25, 0.20]  # most folks are month-to-month
    ),
    'InternetService': np.random.choice(
        ['DSL', 'Fiber Optic', 'None'], 
        NUM_CUSTOMERS, 
        p=[0.35, 0.45, 0.20]
    ),
    'TechSupport': np.random.choice(['Yes', 'No'], NUM_CUSTOMERS, p=[0.40, 0.60]),
}

df = pd.DataFrame(data)

# monthly charges depend on what services they have
base_charge = 30
df['MonthlyCharges'] = base_charge + np.random.normal(0, 5, NUM_CUSTOMERS)
df.loc[df['InternetService'] == 'DSL', 'MonthlyCharges'] += 25
df.loc[df['InternetService'] == 'Fiber Optic', 'MonthlyCharges'] += 45
df.loc[df['TechSupport'] == 'Yes', 'MonthlyCharges'] += 15
df.loc[df['ContractType'] == 'One-Year', 'MonthlyCharges'] *= 0.95  # slight discount
df.loc[df['ContractType'] == 'Two-Year', 'MonthlyCharges'] *= 0.90
df['MonthlyCharges'] = df['MonthlyCharges'].clip(20, 150).round(2)

# total is just monthly * tenure
df['TotalCharges'] = (df['MonthlyCharges'] * df['Tenure']).round(2)

# now calculate churn probability based on various factors
churn_prob = np.zeros(NUM_CUSTOMERS) + 0.10  # start with 10% base rate

# things that make customers more likely to leave
churn_prob += (df['ContractType'] == 'Month-to-Month') * 0.15
churn_prob += (df['TechSupport'] == 'No') * 0.08
churn_prob += (df['InternetService'] == 'Fiber Optic') * 0.05  # fiber has more issues apparently
churn_prob += (df['MonthlyCharges'] > 80) * 0.07
churn_prob += (df['Tenure'] < 6) * 0.12  # new customers leave more often

# things that keep customers around
churn_prob -= (df['ContractType'] == 'Two-Year') * 0.08
churn_prob -= (df['TechSupport'] == 'Yes') * 0.05
churn_prob -= (df['Tenure'] > 36) * 0.10  # loyalty
churn_prob -= (df['Tenure'] > 48) * 0.05

# add noise so it's not deterministic
churn_prob += np.random.normal(0, 0.05, NUM_CUSTOMERS)
churn_prob = churn_prob.clip(0.02, 0.60)

# flip the coin for each customer
df['Churn'] = np.where(np.random.random(NUM_CUSTOMERS) < churn_prob, 'Yes', 'No')

# reorder columns to look nice
df = df[['CustomerID', 'Age', 'Gender', 'Tenure', 'MonthlyCharges', 
         'ContractType', 'InternetService', 'TotalCharges', 'TechSupport', 'Churn']]

df.to_csv('customer_churn_data.csv', index=False)

# print some stats
print(f"\ngenerated {NUM_CUSTOMERS} customers")
print(f"\nchurn breakdown:")
print(df['Churn'].value_counts())
print(f"\nchurn rate: {(df['Churn']=='Yes').mean()*100:.1f}%")

print(f"\nfeature analysis:")
for col in ['Tenure', 'MonthlyCharges']:
    print(f"\n{col}:")
    print(df.groupby('Churn')[col].agg(['mean', 'std']).round(2))

for col in ['TechSupport', 'ContractType', 'InternetService']:
    print(f"\n{col}:")
    print(pd.crosstab(df[col], df['Churn'], normalize='index').round(2))

print("\nsaved to customer_churn_data.csv")
print("now run: python train_model.py")
