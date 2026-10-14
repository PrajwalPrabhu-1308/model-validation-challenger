# Independent Model Validation & Effective Challenge Framework

**Champion–Challenger Modeling | Model Risk Management | Fairness | Drift Detection | SHAP | LLM-as-Judge**

## Overview

This project implements an independent model validation framework for a synthetic customer-response scoring problem with credit-adjacent characteristics.

The framework evaluates whether a higher-performing predictive model is also robust, fair across demographic segments, and stable under temporal distribution shifts.

Rather than relying on a single performance metric, it combines automated challenger tests, explainability, and an LLM-based reporting layer to identify potential model risks and support human review.

The project demonstrates a model-risk validation workflow inspired by the principle of **effective challenge**: independently scrutinizing a model's assumptions, behavior, limitations, and potential failure modes before deployment.

**The current implementation is a research and portfolio prototype using synthetic data. It is not a validated banking model or a production model-risk management system.**

## Key Objectives

- Benchmark a stacked champion model against a simpler legacy-style baseline.
- Build an independent challenger framework that can evaluate fitted classification models through their prediction interfaces.
- Detect sensitivity to small input perturbations.
- Identify demographic disparities, including potential proxy discrimination.
- Monitor feature and prediction-score distribution drift.
- Use SHAP to investigate the features associated with observed disparities.
- Evaluate a local large language model as a validation-reporting and escalation-triage assistant.
- Generate an aggregate model validation report for human review.

## Architecture

The notebook implements six main components.

### 1. Synthetic customer-response dataset

A synthetic dataset is generated to represent customer characteristics, behavioral variables, and a binary response outcome.

The dataset is deliberately constructed to contain three test conditions:

- Predictive signal that allows meaningful model comparison.
- Potential proxy discrimination through behavioral features correlated with age.
- Temporal distribution shift in a recent data vintage.

The dataset includes numerical features, categorical features, an outcome variable, age-group labels for fairness testing, and vintage labels for drift evaluation.

No real customer or banking data is required.

### 2. Champion–Challenger model comparison

Two predictive models are trained and evaluated.

| Model | Description |
|---|---|
| Baseline | A shallow XGBoost classifier using a reduced feature set, representing a legacy-style model. |
| Champion | A stacked classifier combining Logistic Regression, Random Forest, and XGBoost, with Logistic Regression as the meta-learner. |

The models use preprocessing pipelines with numerical standardization and categorical one-hot encoding.

The notebook compares their ROC-AUC scores on an in-time holdout dataset.

The champion's additional feature and model complexity is intended to improve predictive performance. The validation framework then tests whether the resulting model behavior introduces other risks.

### 3. Independent Challenger Framework

The standalone `ModelChallenger` class evaluates the fitted champion through its `predict_proba()` interface.

It does not depend on the champion's internal model architecture or training pipeline. This separates the validation logic from model development and makes the framework reusable with other compatible classifiers.

The challenger runs three families of tests.

**Robustness testing**

- Perturbs numerical input features with Gaussian noise.
- Compares predictions before and after perturbation.
- Measures changes in predicted probabilities.
- Flags excessive score sensitivity against a configurable threshold.

**Fairness testing**

- Evaluates outcomes across demographic segments.
- Calculates demographic-parity differences.
- Calculates disparate-impact ratios.
- Flags potential disparities against configurable thresholds.

Age group is withheld from the model's direct feature set but retained for downstream fairness analysis. This allows the experiment to examine whether correlated behavioral features may still produce disparate outcomes.

**Distribution drift testing**

- Calculates Population Stability Index (PSI) for numerical features.
- Uses a recent, held-out data vintage to represent temporal distribution shift.
- Applies a two-sample Kolmogorov–Smirnov test to model-score distributions.
- Flags shifts that exceed configured thresholds.

The outputs are structured test results and candidate risk flags for further investigation.

### 4. SHAP explainability

The notebook uses SHAP to investigate the feature contributions associated with a detected fairness concern.

A separate XGBoost model is trained on the same training split and preprocessed feature representation to provide a tractable TreeExplainer analysis.

The analysis includes:

- Global feature importance based on mean absolute SHAP values.
- Comparison of mean SHAP values for the age 66–80 segment against the remaining population.
- Identification of features with the largest differences between these groups.
- A global SHAP feature-importance visualization.

This analysis illustrates how facially neutral variables, such as transaction frequency or contact recency, can contribute to different model behavior across demographic groups.

**Interpretation caveat:** SHAP describes feature contributions to the explainer model's predictions. Because the explainer is a separate XGBoost model rather than the full stacked champion, its explanations are diagnostic evidence, not an exact decomposition of the champion's predictions or proof of causal discrimination.

### 5. LLM-as-Judge reporting layer

The notebook integrates a locally executed Qwen2.5-1.5B-Instruct model to convert challenger findings into concise natural-language summaries and escalation recommendations.

The reporting layer:

1. Converts a validation scenario into a structured prompt.
2. Requests an `ESCALATE` or `NO_ACTION` verdict and a short explanation.
3. Parses the generated response as JSON.
4. Compares the verdict with a predefined human-review label.
5. Aggregates the results into a validation report.

The judge uses greedy decoding for deterministic generation and defensive output parsing.

The model is loaded locally through Hugging Face Transformers. On a compatible GPU, the notebook uses 4-bit NF4 quantization through `bitsandbytes`.

Ten predefined validation scenarios cover fairness, drift, and robustness findings, including ambiguous cases intended to test contextual judgment beyond simple threshold comparisons.

The evaluation reports agreement with the scenario labels. This is a small prototype evaluation, not evidence of general LLM-judge reliability.

### 6. Aggregate validation report

The final reporting function consolidates:

- Champion and baseline ROC-AUC.
- Automated challenger findings.
- Robustness, fairness, and drift flags.
- LLM verdicts and agreement with human-review labels.
- Recommended areas for further investigation.

The report supports a human-led review process. It does not automatically approve a model for deployment.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Predictive modeling | scikit-learn, XGBoost |
| Data processing | NumPy, pandas |
| Statistical testing | SciPy |
| Explainability | SHAP |
| Local LLM inference | Hugging Face Transformers |
| Quantization | bitsandbytes, NF4 |
| Deep learning backend | PyTorch |
| Visualization | Matplotlib |
| Execution environment | Jupyter Notebook / Google Colab |

## Repository Structure

The notebook is located directly in the repository root.

```text
model-validation-challenger/
├── README.md
└── model_validation_challenger_prototype.ipynb
```

Additional files, such as a dependency manifest, test suite, configuration files, and exported validation reports, can be added as the prototype develops.

## Getting Started

### Prerequisites

- Python 3.10 or 3.11 recommended.
- Jupyter Notebook or Google Colab.
- Internet access to download Python packages and pretrained model weights.
- A CUDA-enabled GPU recommended for quantized LLM inference.

The notebook can also run on CPU, although LLM inference will be slower.

### Recommended environment: Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `model_validation_challenger_prototype.ipynb` from your repository.
3. Select **Runtime → Change runtime type → T4 GPU**, if available.
4. Run the notebook cells sequentially.
5. Review the champion-versus-baseline comparison, challenger findings, SHAP outputs, LLM verdicts, and final validation report.

The notebook generates synthetic data, so no external dataset or API key is required.

### Local execution

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies used by the notebook:

```bash
python -m pip install --upgrade pip
pip install numpy pandas scipy matplotlib scikit-learn xgboost shap
pip install torch transformers accelerate bitsandbytes
pip install jupyter
```

For local GPU execution, ensure that the installed PyTorch, CUDA, and `bitsandbytes` versions are compatible with your hardware and operating system.

Launch Jupyter:

```bash
jupyter notebook
```

Open `model_validation_challenger_prototype.ipynb` from the repository root and execute the cells in sequence.

## Experimental Design

### Training and evaluation

The notebook separates the most recent synthetic data vintage from the in-time population used for model training and holdout evaluation.

The in-time population is divided into training and holdout subsets using a stratified split. The recent vintage is reserved for temporal drift analysis.

### Validation thresholds

The challenger uses configurable thresholds, including:

- Robustness: maximum mean absolute score change.
- Fairness: demographic-parity difference and disparate-impact ratio.
- Feature drift: PSI.
- Prediction-score drift: Kolmogorov–Smirnov test p-value.

These thresholds are configurable screening criteria. They are not universal regulatory standards, and a threshold breach does not independently establish model failure or legal non-compliance.

### LLM judge evaluation

The judge is evaluated against ten predefined human-review labels. The resulting agreement rate is descriptive and sensitive to scenario selection, label quality, and the composition of the evaluation set.

A larger, independently constructed scenario set and review by multiple qualified human validators would be required to establish reliability.

## Limitations

- **Synthetic data:** Results demonstrate behavior under constructed conditions rather than real customer populations.
- **Limited fairness metrics:** Demographic parity and disparate impact do not cover all relevant definitions of fairness.
- **Proxy-discrimination interpretation:** Correlations and SHAP contributions identify potential mechanisms but do not establish causality.
- **Separate SHAP explainer:** The explained XGBoost model is not the full stacked champion.
- **Limited drift testing:** PSI and score-distribution tests detect distributional changes but do not establish why they occurred or whether predictive performance deteriorated.
- **Robustness scope:** Gaussian perturbations test one specific form of input sensitivity.
- **Small LLM evaluation set:** Ten scenarios are insufficient to establish general judge accuracy or reliability.
- **LLM output risks:** The judge may produce incorrect summaries, inconsistent decisions, or parsing failures. Human review remains necessary.
- **Threshold calibration:** Results depend on selected thresholds, data distributions, and evaluation design.
- **Production readiness:** The notebook is not integrated with model registries, monitoring infrastructure, access controls, audit storage, or deployment approval workflows.

## Future Work

- Evaluate the framework on appropriately governed, independently labeled real-world data.
- Add equalized-odds, calibration-by-group, and intersectional fairness analyses.
- Validate statistical thresholds against business risk and applicable regulatory expectations.
- Add repeated temporal backtesting and performance monitoring.
- Test the challenger against multiple model architectures and model versions.
- Improve SHAP analysis for the complete champion pipeline.
- Expand the LLM evaluation set and assess inter-rater agreement across human reviewers.
- Track prompts, model versions, thresholds, and validation results for auditability.
- Add automated tests and integrate validation into continuous integration workflows.
- Persist validation findings and escalation decisions in a governed model-risk repository.

## Project Status

**Status: Research and portfolio prototype.**

The project demonstrates how independent model validation can combine predictive benchmarking, automated risk tests, explainability, and LLM-assisted reporting in one workflow.

Its central premise is that predictive performance alone is insufficient for model approval. A model should also be challenged for robustness, fairness, and distributional stability, with findings investigated by qualified human reviewers.

## References and Further Reading

- [SR 11-7: Guidance on Model Risk Management](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [XGBoost documentation](https://xgboost.readthedocs.io/)
- [SHAP documentation](https://shap.readthedocs.io/)
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/)
- [Qwen2.5-1.5B-Instruct model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

