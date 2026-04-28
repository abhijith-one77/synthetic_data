import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(2026)

# INCREASING SIZE: 500 Cases and 500 Controls (1000 total participants)
n_cases = 500
n_controls = 500
total_n = n_cases + n_controls

def generate_participant_data(start_id, n_size, filename, is_case):
    ids = range(start_id, start_id + n_size)

    # Cases get a dementia diagnosis date, controls get None (healthy)
    if is_case:
        dementia_dates = ['2020-01-01' if x > 0.1 else None for x in np.random.rand(n_size)]
    else:
        dementia_dates = [None] * n_size

    df = pd.DataFrame({
        'Participant ID': ids,
        'Date of attending assessment centre | Instance 0': pd.to_datetime(['2010-01-01'] * n_size),
        'Date of all cause dementia report': dementia_dates,
        'Albumin | Instance 0': np.random.normal(45, 5, n_size),
        'Creatinine | Instance 0': np.random.normal(70, 15, n_size),
        'Glucose | Instance 0': np.random.normal(5, 1, n_size),
        'C-reactive protein | Instance 0': np.random.lognormal(0.5, 0.5, n_size),
        'Lymphocyte percentage | Instance 0': np.random.normal(30, 5, n_size),
        'Mean corpuscular volume | Instance 0': np.random.normal(90, 5, n_size),
        'Red blood cell (erythrocyte) distribution width | Instance 0': np.random.normal(13, 1, n_size),
        'Alkaline phosphatase | Instance 0': np.random.normal(70, 20, n_size),
        'White blood cell (leukocyte) count | Instance 0': np.random.normal(6, 2, n_size),
        'Age at recruitment': np.random.randint(50, 80, n_size),
        'Sex': np.random.choice(['Male', 'Female'], n_size)
    })

    # Inject 10% missing values (NaN) into the biomarker columns for MICE to impute
    biomarkers = [
        'Albumin | Instance 0', 'Creatinine | Instance 0', 'Glucose | Instance 0',
        'C-reactive protein | Instance 0', 'Lymphocyte percentage | Instance 0',
        'Mean corpuscular volume | Instance 0', 'Red blood cell (erythrocyte) distribution width | Instance 0',
        'Alkaline phosphatase | Instance 0', 'White blood cell (leukocyte) count | Instance 0'
    ]
    for col in biomarkers:
        mask = np.random.rand(n_size) < 0.10
        df.loc[mask, col] = np.nan

    df.to_csv(filename, index=False)
    return ids

# 1. Generate the main cases and controls
print("Generating 1000 participants...")
case_ids = generate_participant_data(1000, n_cases, "Participant_table.csv", is_case=True)
control_ids = generate_participant_data(1000 + n_cases, n_controls, "Participant_table_control.csv", is_case=False)

all_ids = list(case_ids) + list(control_ids)

# 2. Death Record_table.csv
pd.DataFrame({
    'Participant ID': all_ids,
    'Date of death': ['2021-05-01' if x > 0.85 else None for x in np.random.rand(total_n)]
}).dropna().to_csv("Death Record_table.csv", index=False)

# 3. apoe_results.raw
pd.DataFrame({
    'FID': all_ids,
    'IID': all_ids,
    'rs429358_C': np.random.choice([0, 1, 2], total_n),
    'rs7412_T': np.random.choice([0, 1, 2], total_n)
}).to_csv("apoe_results.raw", sep='\t', index=False)

# 4. covariates.csv
pd.DataFrame({
    'Participant ID': all_ids,
    'BMI': np.random.normal(25, 4, total_n)
}).to_csv("covariates.csv", index=False)

# 5. NEW: AD_Participant_table.csv (Alzheimer's Disease specific subset)
# Randomly assign 60% of our dementia cases to be specific Alzheimer's cases
ad_cases = np.random.choice(case_ids, size=int(n_cases * 0.60), replace=False)
pd.DataFrame({
    'Participant ID': ad_cases
}).to_csv("AD_Participant_table.csv", index=False)

print("All files successfully generated!")
