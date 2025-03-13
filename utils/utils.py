import numpy as np
import pandas as pd
import pickle

def load_dataset(dataset_id):
    # Loads training, test, and training labels for dataset_id (0, 1, or 2)
    Xtrk = pd.read_csv(f"./data/Xtr{dataset_id}.csv", index_col=0)
    Xtek = pd.read_csv(f"./data/Xte{dataset_id}.csv", index_col=0)
    Ytrk = pd.read_csv(f"./data/Ytr{dataset_id}.csv", index_col=0)
    return Xtrk, Xtek, Ytrk

def param_loader(k, m, idx, alpha=1):
    with open(f'saved_dictionnaries/kmers_dict_data{idx}_k{k}_m{m}.pkl', 'rb') as f:
        dict_kmers = pickle.load(f)
    with open(f'saved_dictionnaries/neighbors_dict_data{idx}_k{k}_m{m}.pkl', 'rb') as f:
        dict_neighbors = pickle.load(f)
    dict = {
        'k': k,
        'm': m,
        'kmers_dict': dict_kmers,
        'neighbors': dict_neighbors,
        'alpha': alpha
    }
    return dict

def train_test_split(X, y, test_size=0.25, random_state=None, stratify=None):
    """
    Splits X and y into train and test sets.
    
    Parameters:
      X: pandas DataFrame, Series, or numpy array.
      y: pandas Series, DataFrame, or numpy array.
      test_size: float between 0.0 and 1.0 representing the proportion of the dataset to include in the test split.
      random_state: int seed for reproducibility.
      stratify: array-like (same length as y) that defines group labels for stratified splitting.
    
    Returns:
      X_train, X_test, y_train, y_test.
    """
    n_samples = len(y)
    rng = np.random.RandomState(random_state)
    
    if stratify is None:
        # Simple random permutation split.
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        n_test = int(round(test_size * n_samples))
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    else:
        # Use stratified sampling to maintain class proportions.
        stratify = np.array(stratify)
        # Determine total test sample count
        n_test_total = int(round(test_size * n_samples))
        test_indices = []
        train_indices = []
        
        # Compute counts for each class.
        unique_classes, class_counts = np.unique(stratify, return_counts=True)
        # Allocate test samples per class proportionally.
        class_test_counts = {}
        for cls, count in zip(unique_classes, class_counts):
            # Initial allocation: round the proportion.
            count_test = int(round((count / n_samples) * n_test_total))
            # Ensure that if a class is present, at least one sample goes to test (if possible).
            if count > 0 and count_test == 0:
                count_test = 1
            # Ensure we don't assign all samples of a class to the test set.
            if count_test >= count:
                count_test = count - 1 if count > 1 else 1
            class_test_counts[cls] = count_test

        # Adjust total test count in case of rounding issues.
        total_allocated = sum(class_test_counts.values())
        diff = n_test_total - total_allocated
        unique_classes_list = list(unique_classes)
        idx = 0
        while diff != 0:
            cls = unique_classes_list[idx % len(unique_classes_list)]
            if diff > 0:
                # Add one if possible.
                if class_test_counts[cls] < np.sum(stratify == cls):
                    class_test_counts[cls] += 1
                    diff -= 1
            elif diff < 0:
                # Remove one if possible (but leave at least one sample for test if class has more than one sample).
                if class_test_counts[cls] > 1:
                    class_test_counts[cls] -= 1
                    diff += 1
            idx += 1

        # For each class, split its indices.
        for cls in unique_classes:
            cls_indices = np.where(stratify == cls)[0]
            rng.shuffle(cls_indices)
            n_test_cls = class_test_counts[cls]
            test_indices.extend(cls_indices[:n_test_cls])
            train_indices.extend(cls_indices[n_test_cls:])
        
        test_indices = np.array(test_indices)
        train_indices = np.array(train_indices)
        # Optional: shuffle the final train/test indices.
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)
    
    # Helper: index data preserving its type (pandas DataFrame/Series or numpy array).
    def _index_data(data, indices):
        return data.iloc[indices] if hasattr(data, "iloc") else data[indices]
    
    X_train = _index_data(X, train_indices)
    X_test  = _index_data(X, test_indices)
    y_train = _index_data(y, train_indices)
    y_test  = _index_data(y, test_indices)
    
    return X_train, X_test, y_train, y_test

def accuracy_score(y_true, y_pred):
    """
    Computes the accuracy classification score.
    
    Parameters:
      y_true: array-like of true labels.
      y_pred: array-like of predicted labels.
    
    Returns:
      Accuracy as a float.
    """
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    return np.mean(y_true == y_pred)