import numpy as np
import pandas as pd
from utils.utils import load_dataset, param_loader
from models.models import SVC, LogisticRegression
from kernels.kernels import SpectrumKernel, MismatchKernel, LocalAlignmentKernel, WeightedDegreeKernel
from utils.utils import train_test_split, accuracy_score

# Control variables
validation = True
submission = True

# Define the kernels to use for each dataset.
# For each dataset (key), the value is a list of kernel objects.
# If more than one kernel is provided, their resulting kernel matrices will be summed.
kernels = {
    0: [MismatchKernel(**param_loader(18, 3, 0, 0.9)), MismatchKernel(**param_loader(16, 2, 0, 0.9)), MismatchKernel(**param_loader(14, 3, 0, 0.9)), MismatchKernel(**param_loader(12, 2, 0, 0.9)), MismatchKernel(**param_loader(5, 1, 0, 0.9))],
    1: [MismatchKernel(**param_loader(18, 3, 1)), MismatchKernel(**param_loader(16, 2, 1)), MismatchKernel(**param_loader(14, 3, 1)), MismatchKernel(**param_loader(12, 2, 1)), MismatchKernel(**param_loader(5, 1, 1))],
    2: [MismatchKernel(**param_loader(14, 3, 2)), MismatchKernel(**param_loader(16, 2, 2)), MismatchKernel(**param_loader(12, 2, 2)), MismatchKernel(**param_loader(8, 1, 2)), MismatchKernel(**param_loader(5, 1, 2))]
}

# Define the classifier to use for each dataset.
classifiers = {
    0: LogisticRegression(max_iter=3000, C=1.0),
    1: SVC(kernel='precomputed'),
    2: LogisticRegression(max_iter=3000, C=1.0)
}

# Define the datasets to process.
indices = [0, 1, 2]

all_test_predictions = []
for dataset_id in indices:
    print(f"\nProcessing Dataset {dataset_id}...")
    Xtrk, Xtek, Ytrk = load_dataset(dataset_id)
    
    if validation:
        # Split training data into training and validation sets.
        X_train, X_val, y_train, y_val = train_test_split(
            Xtrk['seq'], Ytrk['Bound'], test_size=0.25, stratify=Ytrk['Bound']
        )
    else:
        X_train = Xtrk['seq']
        y_train = Ytrk['Bound']
    
    # Retrieve the list of kernels for this dataset.
    kernel_list = kernels[dataset_id]
    
    # Compute the training kernel matrix as the sum of the kernels (if more than one is provided).
    K_train = sum(kernel.compute_kernel(X_train, X_train) for kernel in kernel_list)
    
    if validation:
        K_val = sum(kernel.compute_kernel(X_val, X_train) for kernel in kernel_list)
    
    # Retrieve the classifier for this dataset.
    clf = classifiers[dataset_id]
    
    if validation:
        clf.fit(K_train, y_train)
        y_val_pred = clf.predict(K_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Validation Accuracy on Dataset {dataset_id} = {val_accuracy:.3f}")
    
    if submission:
        # Re-fit on full training data and predict on test data.
        K_tr_full = sum(kernel.compute_kernel(Xtrk['seq'], Xtrk['seq']) for kernel in kernel_list)
        K_test = sum(kernel.compute_kernel(Xtek['seq'], Xtrk['seq']) for kernel in kernel_list)
        
        clf.fit(K_tr_full, Ytrk['Bound'])
        preds = clf.predict(K_test)
        all_test_predictions.extend(preds)

# Save submission file if required.
if submission:
    submission_df = pd.DataFrame({
        "Id": np.arange(len(all_test_predictions)),
        "Bound": all_test_predictions
    })
    submission_df.to_csv("Yte.csv", index=False)
    print("\nSubmission file 'submission.csv' has been saved.")
