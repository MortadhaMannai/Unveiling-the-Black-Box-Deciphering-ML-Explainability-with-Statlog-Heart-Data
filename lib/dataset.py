import pandas as pd
from sklearn.metrics import accuracy_score

from .definitions import RANDOM_SEED, PROCESSED_DATA_OUTPUT_PATH, SPLITS_BASE_PATH


def load_training_data():
    df = pd.read_csv(PROCESSED_DATA_OUTPUT_PATH / 'heart.csv')

    features = df.drop(['heart_disease'], axis=1)
    labels = df[['heart_disease']]

    return features, labels


def evaluate_estimator(estimator, train_values, train_labels, val_values, val_labels):
        train_labels_list = list(train_labels['heart_disease'].values)
        val_labels_list = list(val_labels['heart_disease'].values)
        
        estimator.fit(train_values, train_labels_list)
        
        predictions = estimator.predict(val_values)
        score = accuracy_score(val_labels_list, predictions)
        
        return score


def load_split(split_index):
    split_path = SPLITS_BASE_PATH / str(split_index + 1)

    split_files = ['train_values', 'train_labels', 'val_values', 'val_labels']

    return list(
        map(
            lambda filename: pd.read_csv(split_path / f'{filename}.csv'),
            split_files))


def evaluate_splits(fn_create_estimator, verbose=0, random_state=RANDOM_SEED, n_jobs=-1):
    scores = []
    
    for split_index in range(3):
        (train_values, train_labels, val_values, val_labels) = load_split(split_index)
        
        estimator = fn_create_estimator(verbose=verbose, random_state=random_state, n_jobs=n_jobs)
        split_score = evaluate_estimator(estimator, train_values, train_labels, val_values, val_labels)    
        scores.append(split_score)
    
    return scores


def train_on_splits(fn_create_estimator, verbose=0, random_state=RANDOM_SEED, n_jobs=-1):    
    results = []
    
    for split_index in range(3):
        (train_values, train_labels, val_values, val_labels) = load_split(split_index)
        
        estimator = fn_create_estimator(verbose=verbose, random_state=random_state, n_jobs=n_jobs)
        split_score = evaluate_estimator(estimator, train_values, train_labels, val_values, val_labels)
        
        results.append({
            'estimator': estimator,
            'score': split_score,
            'data': (train_values, train_labels, val_values, val_labels)
        })
    
    return results
