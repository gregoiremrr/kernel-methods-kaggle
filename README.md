# Kernel-Based Sequence Classification and Learnable Kernels

## Overview
This project provides a modular framework for sequence classification using kernel methods, with both fixed (e.g., Mismatch Kernel) and learnable kernel representations. The implementation supports traditional kernel-based classifiers (e.g., Logistic Regression, SVC) as well as deep learning models that learn sequence representations via architectures such as BiGRU, CNN, and Transformer-based models.

## Installation
Before running any of the provided scripts, install the required dependencies using:
```bash
pip install -r requirements.txt
```

## Project Structure
- **create_cache.py**: Precomputes and caches k-mer dictionaries and neighbor mappings used for mismatch kernel computations.
- **start.py**: Main pipeline for traditional kernel methods. It loads datasets, computes kernel matrices (by summing multiple kernels if needed), trains classifiers, and optionally generates a submission file.
- **start_learn.py**: Pipeline for learnable kernel models. It trains deep learning models (using BiGRU, CNN, or Transformer architectures) to obtain sequence representations, computes a precomputed kernel matrix from these representations, trains an SVC, and reports validation accuracy or produces a submission file.
- **utils/**, **models/**, **kernels/**: These directories contain the necessary utilities, classifier implementations, and kernel functions respectively.

## Preprocessing: Cache Generation for Mismatch Kernels
Before running either `start.py` or `start_learn.py`, ensure that the necessary cache files for the mismatch kernels are generated. Use the following command:
```bash
python create_cache.py --k <k> --m <m>
```
Replace `<k>` with the desired k-mer length and `<m>` with the allowed number of mismatches (with the constraint that `k > m`). For example:
```bash
python create_cache.py --k 16 --m 2
```

## Usage Instructions

### Running the Traditional Kernel Pipeline (`start.py`)
1. **Generate Cache Files:**  
   Run `create_cache.py` as described above.
2. **Execute the Pipeline:**  
   Run the main script:
```bash
python start.py
```
This script:
- Loads the datasets.
- Computes kernel matrices using a sum of mismatch kernels (or other kernels if configured).
- Splits data for validation (if enabled) and reports accuracy.
- Trains the classifier on the full dataset and produces a submission file (`Yte.csv`) if submission mode is enabled.

### Running the Learnable Kernel Pipeline (`start_learn.py`)
1. **Generate Cache Files:**  
As with the traditional pipeline, run `create_cache.py` to generate the necessary cache files.
2. **Execute the Pipeline:**  
Run the learnable kernel script:
```bash
python start_learn.py
```
This script:
- Loads the datasets and splits them appropriately.
- Creates PyTorch datasets and dataloaders.
- Trains a learnable kernel model (select from BiGRU, CNN, or Transformer representations).
- Computes sequence features via the trained representation model.
- Constructs a precomputed kernel matrix and trains an SVC on these features.
- Reports validation accuracy and generates a submission file (`Yte.csv`) when in submission mode.

## Customization and Experimentation
- **Kernel Parameters:** Adjust k-mer lengths, mismatch tolerance, and other hyperparameters via the configuration in the scripts or through the utility functions.
- **Model Selection:** In `start_learn.py`, uncomment the desired learnable kernel model (e.g., `LearnableKernelModel1`, `LearnableKernelModel2`, or `LearnableKernelModel3`) to experiment with different architectures.
- **Classifier Options:** Modify classifier settings such as regularization parameters and maximum iterations directly in the scripts.
- **Dataset Processing:** The scripts are configured to handle multiple datasets. Modify the dataset indices and split ratios as needed.

## Conclusion
This framework offers a robust and extensible approach to sequence classification using both traditional kernel methods and learnable representations. Its modular design facilitates academic research and rapid experimentation with various kernel functions and deep learning architectures.
