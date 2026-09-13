import numpy as np, pandas as pd

rng = np.random.default_rng(20260912)
N = 1200

age = np.clip(rng.normal(52, 14, N), 20, 85).round().astype(int)
sex = rng.choice(['F', 'M'], N, p=[0.55, 0.45])
blood = rng.choice(['A', 'B', 'O', 'AB'], N, p=[.35, .25, .32, .08])
doctor = rng.choice(['A', 'B', 'C', 'D'], N, p=[.28, .27, .23, .22])

visits_12m = rng.poisson(2.6, N).clip(0, 9)
missing = rng.poisson(1.5 + (age - 50) * 0.035, N).clip(0, 8)
perio = np.clip(rng.normal(3.4 + (age - 50) * 0.02, 1.0, N), 1.5, 8).round(1)
no_show = np.clip(rng.beta(2, 12, N), 0, 0.6).round(2)
past_selfpay = rng.binomial(1, 0.28, N)
has_line = rng.binomial(1, 0.62, N)
income = rng.choice(['low', 'mid', 'high'], N, p=[.30, .48, .22])
days_since = rng.integers(20, 400, N)

inc_eff = pd.Series(income).map({'low': 0.0, 'mid': 0.35, 'high': 0.80}).to_numpy()
doc_eff = pd.Series(doctor).map({'A': 0.0, 'B': -0.28, 'C': 1.15, 'D': 0.05}).to_numpy()

z = (-3.05
     + 0.30 * missing
     + 0.62 * has_line
     + inc_eff
     + 0.55 * past_selfpay
     - 2.60 * no_show
     + 0.20 * perio
     + doc_eff
     - 0.0022 * (age - 55) ** 2
     + 0.10 * visits_12m
     + rng.normal(0, 0.20, N))

p = 1 / (1 + np.exp(-z))
y = rng.binomial(1, p)

consult = np.where(y == 1, rng.normal(34, 9, N), rng.normal(12, 6, N)).clip(0, 90).round().astype(int)

df = pd.DataFrame({
    'patient_id': [f'P{i:04d}' for i in range(1, N + 1)],
    'age': age, 'sex': sex, 'blood_type': blood, 'doctor': doctor,
    'visits_12m': visits_12m, 'missing_teeth': missing, 'perio_depth': perio,
    'no_show_rate': no_show, 'past_selfpay': past_selfpay, 'has_line': has_line,
    'income_area': income, 'days_since_visit': days_since,
    'consult_minutes': consult, 'selfpay_next90': y,
})

# --- 故意弄髒 ---
bad = rng.choice(N, 8, replace=False)
df.loc[bad, 'age'] = 999
miss = rng.choice(N, 55, replace=False)
df.loc[miss, 'perio_depth'] = np.nan
dup = df.sample(6, random_state=7)
df = pd.concat([df, dup], ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)

df.to_csv('dental_patients.csv', index=False, encoding='utf-8-sig')
print(df.shape, 'base rate =', round(df.selfpay_next90.mean(), 3))
print(df.groupby('doctor')['selfpay_next90'].mean().round(3).to_dict())
print(df.groupby('sex')['selfpay_next90'].mean().round(3).to_dict())
print(df.groupby('blood_type')['selfpay_next90'].mean().round(3).to_dict())
print(df.groupby('income_area')['selfpay_next90'].mean().round(3).to_dict())
