# Does Domain-Specific Pretraining Improve Patent Reference Extraction?

This repository contains the implementation and experimental configurations for comparing BERT-base and SciBERT on patent reference extraction.

The study investigates whether domain-specific pretraining improves token-level reference extraction performance and whether the performance difference between the two models varies between B-labels and I-labels.

## 1. Research Overview

This project compares two pretrained language models:

- **BERT-base:** `bert-base-cased`
- **SciBERT:** `allenai/scibert_scivocab_cased`

The experiments use a BIO-labeled patent dataset and evaluate the models using Leave-One-Out Cross-Validation (LOO-CV).

The main research questions are:

- **RQ1:** Does SciBERT outperform BERT-base in patent reference extraction?
- **RQ2:** Does the performance difference between SciBERT and BERT-base vary between B-labels and I-labels?

## 2. Repository Structure

```text
patent-citation-extraction/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── run_ner.py
├── bert.py
├── ann2bio.py
│
 |
│  
│   
│
└── results/
    ├── OUTPUT_DIR_BERT_BASED/
    │   ├── seed_42/
    │   ├── seed_123/
    │   └── seed_2026/
    │
    └── OUTPUT_DIR_SCIBERT/
        ├── seed_42/
        ├── seed_123/
        └── seed_2026/
```

**Note:** The `data/` directory is not included in this repository. Users must obtain and prepare the original dataset separately.

## 3. Dataset

This study uses the patent BIO dataset associated with the following work:

**Original Paper**

*Improving Reference Mining in Patents with BERT.*

**Original Repository**

https://github.com/kaesve/patent-citation-extraction

The dataset consists of 22 manually annotated patent documents in BIO format.

The dataset files are not redistributed in this repository. Users should obtain the data from the original source and comply with its applicable licensing and usage conditions.

Before running the experiments, place the prepared BIO files in the directory expected by the data-loading code.

The original dataset contains the following BIO label distribution:

| Label | Number of tokens | Percentage |
|---|---:|---:|
| B | 2,318 | 0.37% |
| I | 32,359 | 5.23% |
| O | 583,973 | 94.40% |
| **Total** | **618,650** | **100%** |

## 4. Environment

The experiments were conducted using the following software environment.

| Component | Version |
|---|---|
| Python | 3.13.16 |
| PyTorch | 2.14.1+cu130 |
| Transformers | 5.22.1 |
| NumPy | 2.4.6 |
| Scikit-learn | 1.6.1 |
| NLTK | 3.9.1 |
| tqdm | 4.67.3 |
| joblib | 1.6.0 |
| sklearn-crfsuite | 0.5.0 |
| python-crfsuite | 0.9.12 |

### Hardware

The experiments were conducted using Google Colab GPUs.

| Random Seed | GPU |
|---|---|
| 42 | NVIDIA Tesla T4 |
| 123 | NVIDIA L4 |
| 2026 | NVIDIA L4 |

GPU differences are documented for reproducibility.

## 5. Installation

### 5.1. Clone the repository

```bash
git clone [YOUR_REPOSITORY_URL]
cd patent-citation-extraction
```

### 5.2. Install dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

PyTorch should be installed using the official installation instructions to obtain the appropriate CUDA build for the available GPU.

Official installation guide:

https://pytorch.org/get-started/locally/

The reported experimental environment used PyTorch 2.14.1 with CUDA 13.0.

### 5.3. Verify the environment

```python
import torch
import transformers
import sklearn
import nltk
import numpy
import tqdm

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

## 6. Experimental Configuration

The main experimental settings are summarized below.

| Parameter | Value |
|---|---|
| Evaluation strategy | Leave-One-Out Cross-Validation |
| Number of patents | 22 |
| Models | BERT-base, SciBERT |
| Random seeds | 42, 123, 2026 |
| Maximum sequence length | 64 |
| Training batch size | 32 |
| Evaluation batch size | 8 |
| Learning rate | 5e-5 |
| Number of epochs | 3 |
| Warmup proportion | 0.1 |

Both models were evaluated using the same data splits, random seeds, and training configurations.

The pretrained model identifiers are:

```python
MODEL_IDS = {
    "BERT-base": "bert-base-cased",
    "SciBERT": "allenai/scibert_scivocab_cased"
}
```

## 7. Running the Experiments

The experiments compare BERT-base and SciBERT using the same Leave-One-Out evaluation procedure.

The following steps describe the intended execution workflow:

1. Prepare the 22 BIO files in the expected data directory.
2. Install the required dependencies.
3. Download the pretrained model checkpoints from Hugging Face.
4. Run the Leave-One-Out experiments for each model.
5. Repeat the experiments using random seeds 42, 123, and 2026.
6. Collect and compare the evaluation results.

**Execution commands**

The exact command-line arguments should match the argument parser implemented in `run_ner.py`.

- BERT-base: `python run_ner.py --data_dir data/bio --bert_model "google-bert/bert-base-cased"  --output_dir out_leave_one_out --do_leave_one_out --max_seq_length 64 --train_batch_size 32 --eval_batch_size 8 --learning_rate 5e-5 --num_train_epochs 3 --warmup_proportion=0.1 --gradient_accumulation_steps 1'

 - SciBERT: `python run_ner.py --data_dir data/bio --bert_model "allenai/scibert_scivocab_cased"  --output_dir out_leave_one_out --do_leave_one_out --max_seq_length 64 --train_batch_size 32 --eval_batch_size 8 --learning_rate 5e-5 --num_train_epochs 3 --warmup_proportion=0.1 --gradient_accumulation_steps 1'
 
For the exact available arguments, run:

```bash
python run_ner.py --help
```

## 8. Evaluation

The models are evaluated using token-level F1 scores.

The evaluation focuses on:

- B-label F1
- I-label F1
- Micro F1

Results are reported for each random seed and summarized using the mean and standard deviation.

The study also compares the performance differences between BERT-base and SciBERT across the B- and I-labels.

## 9. Statistical Analysis and Error Analysis

The analysis scripts are provided in the `analysis/` directory.

The scripts reproduce the statistical tests and error analyses reported in the paper.

### Statistical Tests

- Paired cluster-level permutation tests were used to compare BERT-base and SciBERT.
- Patent-level results were used as the permutation units.
- Three random-seed results were considered in the analysis.
- The permutation procedure used 200,000 permutations.
- Holm correction was applied to the designated family of statistical tests.

### Error Analysis

The analysis scripts also support:

- Patent-level F1-score comparison.
- Label-wise error rate calculation for B, I, and O labels.
- Comparison of prediction errors between BERT-base and SciBERT.
- Generation of result tables reported in the paper.

### Reproducing the Analysis
python analysis/04_patent_level_comparison_and_permutation.py \
  --bio path/to/bio.zip \
  --bert path/to/bert_outputs.zip \
  --scibert path/to/scibert_outputs.zip \
  --out analysis_results
The input prediction files and evaluation outputs must be generated using the experimental procedure described above.

All statistical results should be regenerated from the provided prediction and evaluation files rather than manually entered values.

## 10. Reproducibility Notes

To support reproducibility:

- Both models use the same dataset and Leave-One-Out splits.
- The same random seeds are used for both models.
- Training and evaluation hyperparameters are kept consistent.
- GPU hardware differences are documented.
- The original dataset is referenced but not redistributed.

The repository provides the source code and experimental configurations. Users must independently obtain the original dataset and prepare the required input files before executing the experiments.

## 11. Acknowledgements and Attribution

This project is based on the original implementation associated with:

*Improving Reference Mining in Patents with BERT.*

Original repository:

https://github.com/kaesve/patent-citation-extraction

The original repository includes an AGPL-3.0 license. Original copyright and license notices are retained in the relevant source files.

The `run_ner.py` file has been modified to support the experimental setup used in this study, including Leave-One-Out cross-validation and multiple random seed experiments.

The original patent BIO dataset is not redistributed in this repository.

## 12. License

This repository retains the applicable GNU Affero General Public License v3.0 (AGPL-3.0) terms for the original code and modifications that are distributed as part of the covered program.

See the `LICENSE` file for the full license text.

Please note that the dataset may be subject to separate terms and conditions.