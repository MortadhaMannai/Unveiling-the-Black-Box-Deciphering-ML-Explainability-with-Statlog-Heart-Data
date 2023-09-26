from pathlib import Path

RANDOM_SEED = 42

DATA_BASE_PATH = Path('../data')
RAW_DATA_BASE_PATH = DATA_BASE_PATH / 'raw'
RAW_TRAINING_DATA = RAW_DATA_BASE_PATH / 'heart.dat'
PROCESSED_DATA_OUTPUT_PATH = DATA_BASE_PATH / 'processed'
SPLITS_BASE_PATH = PROCESSED_DATA_OUTPUT_PATH / 'splits'

column_definitions = {
    'age': int, 
    'sex': int,
    'chest_pain_type': int,
    'resting_blood_pressure': float,
    'serum_cholesterol_mg_per_dl': float,
    'fasting_blood_sugar_gt_120_mg_per_dl': int,
    'resting_ekg_results': int,
    'max_heart_rate_achieved': int,
    'exercise_induced_angina': int,
    'oldpeak_eq_st_depression': float,
    'slope_of_peak_exercise_st_segment': int,
    'num_major_vessels': int,
    'thal': int,
    'heart_disease': int
}

all_column_names = list(column_definitions.keys())

numerical_column_names = [
    'resting_blood_pressure',
    'serum_cholesterol_mg_per_dl', 'oldpeak_eq_st_depression', 'age',
    'max_heart_rate_achieved'
]

categorical_column_names = [
    'slope_of_peak_exercise_st_segment', 'thal',
    'chest_pain_type', 'resting_ekg_results', 'num_major_vessels'
]

binary_column_names = [
    'sex', 'exercise_induced_angina', 'fasting_blood_sugar_gt_120_mg_per_dl'
]
