import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

def test_mean_target_encoding(train, test, target, categorical, alpha=5):
    # Calculate global mean on the train data
    global_mean = train[target].mean()
    # Group by the categorical feature and calculate its properties
    train_groups = train.groupby(categorical)
    category_sum = train_groups[target].sum()
    category_size = train_groups.size()
    # Calculate smoothed mean target statistics
    train_statistics = (category_sum + global_mean * alpha) / (category_size + alpha)
    # Apply statistics to the test data and fill new categories
    test_feature = test[categorical].map(train_statistics).fillna(global_mean)
    return test_feature.values


def train_mean_target_encoding(train, target, categorical, n_splits=5, shuffle=False, stratify=False, alpha=5, random_state=None):
    # Create 5-fold cross-validation
    kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle) if stratify else KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    train_feature = pd.Series(index=train.index)
    # For each fold split
    for train_index, test_index in kf.split(train, train[target]):
        cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
        # Calculate out-of-fold statistics and apply to cv_test
        cv_test_feature = test_mean_target_encoding(cv_train, cv_test, target, categorical, alpha)
        # Save new feature for this particular fold
        train_feature.iloc[test_index] = cv_test_feature
    return train_feature.values


def mean_target_encoding(train, test, target, categorical, n_splits=5, shuffle=False, stratify=False, alpha=5, random_state=None):
    # Get the train feature
    train_feature = train_mean_target_encoding(train, target, categorical, n_splits=n_splits, alpha=alpha, shuffle=shuffle, stratify=stratify, random_state=random_state)
    # Get the test feature
    test_feature = test_mean_target_encoding(train, test, target, categorical, alpha)
    # Return new features to add to the model
    return train_feature, test_feature