# EDM-ARS: Educational Data Mining Automated Research System
## Architecture Specification & Pilot Implementation Guide

---

## 1. System Overview

EDM-ARS is a domain-specific multi-agent system that automates the end-to-end workflow of prediction-focused educational data mining research. The system consists of five specialized agents coordinated by a central orchestrator, operating over a structured data registry.

```
┌──────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                            │
│            (State Machine + Message Router)                   │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐  │
│  │  Problem  │──▶│   Data   │──▶│ Analyst │──▶│  Writer  │  │
│  │Formulator│   │ Engineer │   │         │   │          │  │
│  └──────────┘   └──────────┘   └─────────┘   └──────────┘  │
│       │              │              │              │         │
│       └──────────────┴──────────────┴──────────────┘         │
│                          ▲                                   │
│                     ┌────┴────┐                               │
│                     │ Critic  │  (reviews all agent outputs)  │
│                     └─────────┘                               │
│                          ▲                                   │
│                  ┌───────┴────────┐                           │
│                  │  Data Registry │                           │
│                  └────────────────┘                           │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Data Registry Schema

The data registry encodes domain knowledge about educational datasets so that agents can reason about variables substantively rather than treating them as abstract column names.

### 2.1 Registry File Structure

```
data_registry/
├── datasets/
│   ├── hsls09.yaml
│   ├── assistments.yaml
│   └── pisa2022.yaml
├── task_templates/
│   ├── prediction.yaml
│   ├── causal_inference.yaml
│   └── fairness_audit.yaml
└── evaluation_rubrics/
    ├── methodological_checklist.yaml
    └── edm_review_criteria.yaml
```

### 2.2 Dataset Registry: HSLS:09 Example

```yaml
# data_registry/datasets/hsls09.yaml

dataset:
  name: "HSLS:09"
  full_name: "High School Longitudinal Study of 2009"
  source: "NCES / IES"
  documentation_url: "https://nces.ed.gov/surveys/hsls09/"
  data_structure: "multilevel"
  levels:
    - name: "student"
      id_variable: "STU_ID"
      n_approx: 23503
    - name: "school"
      id_variable: "SCH_ID"
      n_approx: 944
  waves:
    - name: "base_year"
      year: 2009
      grade: 9
      prefix: "X1"
    - name: "first_follow_up"
      year: 2012
      grade: 11
      prefix: "X2"
    - name: "second_follow_up"
      year: 2013
      grade: 12_or_beyond
      prefix: "X3"
    - name: "update_panel"
      year: 2016
      status: "postsecondary"
      prefix: "X4"
  sampling:
    design: "two-stage stratified"
    weights:
      base_year: "W1STUDENT"
      first_follow_up: "W2W1STU"
      panel: "W4W1W2W3STU"
    strata_var: "W1STRATUM"
    psu_var: "W1SCHOOL"
    note: >
      Survey weights should be used for population-level inference.
      For prediction tasks, unweighted analysis is acceptable but
      should be acknowledged as a limitation.

  # ──────────────────────────────────────────
  # VARIABLE CATALOG
  # ──────────────────────────────────────────
  variables:
    # --- COMMON OUTCOME VARIABLES ---
    outcomes:
      - name: "X3TGPAENG"
        label: "GPA in English courses (transcript)"
        type: "continuous"
        range: [0.0, 4.0]
        wave: "second_follow_up"
        missingness_pct: 18.2
        notes: "Transcript-based, more reliable than self-report"

      - name: "X3TGPAMAT"
        label: "GPA in math courses (transcript)"
        type: "continuous"
        range: [0.0, 4.0]
        wave: "second_follow_up"
        missingness_pct: 17.5

      - name: "X2TXMTSC"
        label: "Math theta score (IRT-scaled)"
        type: "continuous"
        range: [-5.0, 5.0]
        wave: "first_follow_up"
        missingness_pct: 24.3
        notes: "IRT-scaled; suitable for continuous outcome modeling"

      - name: "X4EVRATNDCLG"
        label: "Ever attended college (postsecondary)"
        type: "binary"
        values: {1: "Yes", 0: "No"}
        wave: "update_panel"
        missingness_pct: 22.8

      - name: "dropout_derived"
        label: "High school dropout indicator (derived)"
        type: "binary"
        derivation: >
          Constructed from X3HSCOMPSTAT. Code 1 if student did
          not complete high school by expected timeline.
        notes: "Must be explicitly constructed; not a raw variable"

    # --- KEY PREDICTOR VARIABLES ---
    predictors:
      demographic:
        - name: "X1SEX"
          label: "Student sex"
          type: "categorical"
          values: {1: "Male", 2: "Female"}
          protected_attribute: true
          fairness_note: "Commonly used as protected attribute in fairness analyses"

        - name: "X1RACE"
          label: "Student race/ethnicity"
          type: "categorical"
          values:
            0: "Survey component not applicable"
            1: "American Indian/Alaska Native"
            2: "Asian"
            3: "Black/African American"
            4: "Hispanic (no race specified)"
            5: "Hispanic (race specified)"
            6: "More than one race"
            7: "Native Hawaiian/Pacific Islander"
            8: "White"
          protected_attribute: true
          common_recoding: >
            Often collapsed into: White, Black, Hispanic, Asian, Other.
            Codes 4 and 5 typically merged into Hispanic.

        - name: "X1SES_U"
          label: "Socioeconomic status composite (continuous, unstandardized)"
          type: "continuous"
          components: ["parent education", "parent occupation", "family income"]
          wave: "base_year"
          notes: "Composite of parent education, occupation prestige, family income"

        - name: "X1POVERTY185"
          label: "Poverty indicator (185% threshold)"
          type: "binary"
          values: {0: "Above 185% FPL", 1: "At or below 185% FPL"}

      academic:
        - name: "X1TXMTSC"
          label: "Math theta score (base year, IRT-scaled)"
          type: "continuous"
          wave: "base_year"
          notes: "Base-year math proficiency; strong predictor of later outcomes"

        - name: "X1MTHID"
          label: "Math identity composite"
          type: "continuous"
          range: [-3.0, 3.0]
          components: ["see self as math person", "others see me as math person"]

        - name: "X1MTHEFF"
          label: "Math self-efficacy composite"
          type: "continuous"
          range: [-3.0, 3.0]

        - name: "X1MTHINT"
          label: "Math interest composite"
          type: "continuous"
          range: [-3.0, 3.0]

        - name: "X1MTHUTI"
          label: "Math utility value composite"
          type: "continuous"
          range: [-3.0, 3.0]

      course_taking:
        - name: "S3CLASSES"
          label: "Math course sequence (transcript-based)"
          type: "categorical_ordered"
          values:
            1: "No math / below Algebra I"
            2: "Algebra I"
            3: "Geometry"
            4: "Algebra II"
            5: "Trigonometry / Pre-Calculus"
            6: "Calculus"
            7: "AP Calculus"
          wave: "second_follow_up"
          notes: >
            Highest math course taken. Critical variable for OTR
            and course recommendation research. Reflects cumulative
            course pathway through high school.

      school_level:
        - name: "X1LOCALE"
          label: "School locale (urbanicity)"
          type: "categorical"
          values: {1: "City", 2: "Suburb", 3: "Town", 4: "Rural"}
          level: "school"

        - name: "X1REGION"
          label: "Census region"
          type: "categorical"
          values: {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
          level: "school"

        - name: "X1CONTROL"
          label: "School control"
          type: "categorical"
          values: {1: "Public", 2: "Catholic", 3: "Other private"}
          level: "school"

        - name: "C1FENRL"
          label: "Percent free/reduced lunch eligible"
          type: "continuous"
          range: [0, 100]
          level: "school"
          notes: "Proxy for school-level SES composition"

  # ──────────────────────────────────────────
  # COMMON DATA ISSUES & HANDLING GUIDELINES
  # ──────────────────────────────────────────
  data_issues:
    missingness:
      mechanism: "Likely MAR for most variables; MNAR plausible for dropout-related"
      recommended_handling:
        - method: "multiple_imputation"
          package: "mice (R) or sklearn.impute (Python)"
          notes: "Preferred for inference-oriented studies"
        - method: "mean_mode_imputation"
          notes: "Acceptable for pure prediction if missingness < 10%"
        - method: "listwise_deletion"
          notes: "Only if missingness < 5% and MCAR is plausible"
      variables_with_high_missingness:
        - "X2TXMTSC (24.3%): due to assessment nonresponse"
        - "X4EVRATNDCLG (22.8%): panel attrition"

    common_pitfalls:
      - issue: "Using follow-up variables to predict base-year outcomes (temporal leakage)"
        check: "Ensure all predictors precede the outcome temporally"
      - issue: "Ignoring survey design in standard errors"
        check: "Use survey-weighted estimation or acknowledge as limitation"
      - issue: "Treating ordered categories as continuous without justification"
        check: "Document scaling assumptions for Likert-type composites"
      - issue: "Including post-treatment variables in causal models"
        check: "If X1 variables predict X3 outcomes, do not include X2 mediators"

  # ──────────────────────────────────────────
  # CANONICAL RESEARCH QUESTIONS
  # ──────────────────────────────────────────
  canonical_questions:
    prediction:
      - "Predict 11th-grade math achievement (X2TXMTSC) from 9th-grade covariates"
      - "Predict high school dropout from base-year academic and demographic factors"
      - "Predict college enrollment (X4EVRATNDCLG) from high school transcript data"
      - "Predict math GPA (X3TGPAMAT) using course-taking sequences and attitudes"

    fairness:
      - "Audit racial disparities in dropout prediction model (X1RACE as protected)"
      - "Evaluate gender fairness of math achievement predictions (X1SEX as protected)"
      - "Assess SES-based fairness in college enrollment prediction"

    causal_inference:
      - "Estimate effect of advanced math course-taking on college enrollment"
      - "Identify optimal math course sequence for maximizing achievement (OTR)"
      - "Estimate heterogeneous treatment effects of school type on outcomes"
```

### 2.3 Task Template: Prediction

```yaml
# data_registry/task_templates/prediction.yaml

task:
  name: "prediction"
  description: >
    Supervised learning task: predict an educational outcome from
    a set of antecedent variables using standard ML methods.

  required_specifications:
    - outcome_variable: "Single target variable from dataset outcomes catalog"
    - predictor_set: "List of predictor variables, all temporally preceding outcome"
    - target_population: "Full sample or defined subgroup"
    - evaluation_strategy: "Cross-validation scheme"

  standard_workflow:
    steps:
      - name: "data_preparation"
        substeps:
          - "Load dataset and select relevant variables"
          - "Handle missing data (document method and justification)"
          - "Encode categorical variables (one-hot or ordinal as appropriate)"
          - "Check class balance for classification outcomes"
          - "Split into train/test (stratified if classification)"
        checks:
          - "No temporal leakage: all predictors precede outcome"
          - "No target leakage: no variables derived from outcome"
          - "Missing data mechanism documented"

      - name: "model_training"
        substeps:
          - "Fit baseline model (logistic/linear regression)"
          - "Fit tree-based model (random forest or gradient boosting)"
          - "Fit ElasticNet (regression) or SGDClassifier(elasticnet) (classification); tune via GridSearchCV(cv=5)"
          - "Fit MLP (MLPClassifier or MLPRegressor) with max_iter=500, early_stopping=True; tune via GridSearchCV(cv=5)"
          - "Build StackingEnsemble from 5 tuned base models with RidgeCV/LogisticRegressionCV meta-learner"
          - "Tune hyperparameters for RF, XGBoost, ElasticNet, MLP via 5-fold inner CV on training data only"
        checks:
          - "At least 5 individual model families compared (LR, RF, XGBoost, ElasticNet, MLP); StackingEnsemble also present"
          - "Hyperparameter tuning uses inner CV (not test set)"
          - "Training convergence verified"

      - name: "evaluation"
        substeps:
          - "Report metrics on held-out test set only"
          - "Classification: AUC-ROC, accuracy, precision, recall, F1"
          - "Regression: RMSE, MAE, R-squared"
          - "Calibration plot for classification models"
          - "Compute confidence intervals via bootstrap"
        checks:
          - "Metrics computed on test set, not training set"
          - "Multiple metrics reported (not just accuracy)"
          - "Confidence intervals or standard errors provided"

      - name: "interpretation"
        substeps:
          - "Feature importance (permutation or SHAP)"
          - "Partial dependence plots for top 5 predictors"
          - "Subgroup performance analysis (by race, gender, SES)"
          - "Substantive interpretation: do important features make educational sense?"
        checks:
          - "Feature importance method is model-agnostic (SHAP preferred)"
          - "Subgroup disparities flagged if performance varies > 5% across groups"
          - "Educational interpretation connects statistical findings to domain knowledge"

  output_specification:
    tables:
      - "Table 1: Descriptive statistics (mean, SD, missingness for all variables)"
      - "Table 2: Model comparison (metrics across all models)"
      - "Table 3: Feature importance (top 15 predictors with SHAP values)"
    figures:
      - "Figure 1: ROC curves (overlay all models)"
      - "Figure 2: SHAP summary plot"
      - "Figure 3: Subgroup performance comparison"
    sections:
      - "Introduction (problem statement, literature context, research question)"
      - "Methods (data, variables, models, evaluation strategy)"
      - "Results (model performance, feature importance, subgroup analysis)"
      - "Discussion (interpretation, limitations, implications for practice)"
```

---

## 3. Agent System Prompts

### 3.1 Agent 1: Problem Formulator

```yaml
agent_name: "ProblemFormulator"
model: "claude-sonnet-4-20250514"  # or claude-opus-4-20250514 for higher quality
temperature: 0.7  # higher for creative ideation

system_prompt: |
  You are the Problem Formulator agent in an automated Educational Data Mining
  (EDM) research system. Your role is to generate well-specified, novel, and
  educationally meaningful research questions for prediction tasks.

  ## Your Inputs
  You will receive:
  1. A DATASET REGISTRY (YAML) describing available variables, their types,
     missingness, and substantive meaning.
  2. A TASK TEMPLATE describing the required workflow for prediction studies.
  3. Optionally, a SET OF SEED PAPERS (abstracts or summaries) representing
     recent EDM prediction research.
  4. Optionally, a USER PROMPT specifying a research direction or constraint.

  ## Your Output
  Produce a RESEARCH SPECIFICATION in the following JSON format:

  ```json
  {
    "research_question": "Clear, specific research question in one sentence",
    "motivation": "2-3 sentences on why this question matters educationally",
    "outcome_variable": "Exact variable name from the dataset registry",
    "outcome_type": "binary | continuous | multiclass",
    "predictor_set": [
      {
        "variable": "exact variable name",
        "rationale": "why this predictor is theoretically relevant"
      }
    ],
    "target_population": "full sample | description of subgroup",
    "subgroup_analyses": ["list of subgroup dimensions"],
    "expected_contribution": "What this study adds beyond existing literature",
    "potential_limitations": ["anticipated limitation 1", "limitation 2"],
    "novelty_score_self_assessment": 1-5
  }
  ```

  ## Rules
  1. TEMPORAL ORDERING: Every predictor must temporally precede the outcome.
     Check wave prefixes (X1 = base year 2009, X2 = 2012, X3 = 2013, X4 = 2016).
  2. SUBSTANTIVE GROUNDING: Every predictor must have a plausible educational
     rationale. Do not include variables just because they are available.
  3. NOVELTY: Avoid trivially replicating well-known findings (e.g., "SES
     predicts achievement"). Aim for at least one of:
     - Novel predictor combinations (e.g., interaction of math identity × SES)
     - Underexplored outcome variables
     - Underexplored subpopulations
     - Methodological angle (e.g., comparing interpretable vs. black-box models)
  4. FEASIBILITY: Check missingness percentages. If combined missingness across
     predictors would reduce the analytic sample below 10,000, flag this.
  5. PROTECTED ATTRIBUTES: If any predictor is marked protected_attribute: true,
     note this and suggest fairness-aware evaluation in subgroup_analyses.

  ## Quality Criteria
  A good research question:
  - Is answerable with the available data
  - Has clear practical implications for educators or policymakers
  - Goes beyond "which algorithm performs best" to ask substantive questions
  - Specifies the population and context clearly
```

### 3.2 Agent 2: Data Engineer

```yaml
agent_name: "DataEngineer"
model: "claude-sonnet-4-20250514"
temperature: 0.0  # deterministic for code generation

system_prompt: |
  You are the Data Engineer agent in an automated EDM research system.
  Your role is to produce correct, reproducible data preparation code.

  ## Your Inputs
  1. A RESEARCH SPECIFICATION (JSON) from the Problem Formulator agent.
  2. The DATASET REGISTRY (YAML) for the target dataset.
  3. The raw data file path.

  ## Your Output
  Produce a complete, executable Python script that:
  1. Loads the dataset
  2. Selects the specified outcome and predictor variables
  3. Handles missing data according to the strategy below
  4. Encodes categorical variables appropriately
  5. Creates train/test splits
  6. Saves the processed data as structured outputs
  7. Generates a DATA REPORT (JSON) summarizing what was done

  ## Code Requirements
  - Use pandas, numpy, scikit-learn only (no exotic dependencies)
  - Every operation must be logged with print statements
  - All random operations must use a fixed seed (random_state=42)
  - Output files: train_X.csv, train_y.csv, test_X.csv, test_y.csv, data_report.json

  ## Missing Data Protocol
  Follow this decision tree:
  1. If variable missingness < 5%: median imputation (continuous) or mode (categorical)
  2. If variable missingness 5-20%: multiple imputation (IterativeImputer) with 5 iterations
  3. If variable missingness > 20%: FLAG in data_report but still include with imputation;
     note as limitation
  4. If combined complete-case sample < 60% of original: ABORT and return error message
     explaining which variables cause excessive missingness

  ## Validation Checks (MANDATORY)
  Before saving outputs, verify and report:
  - [ ] No NaN values remain in processed data
  - [ ] Outcome variable has expected type and range
  - [ ] No constant (zero-variance) predictors remain
  - [ ] Train/test split is stratified (for classification)
  - [ ] Class balance: report majority/minority ratio
  - [ ] Feature count: report number of columns after encoding
  - [ ] Sample sizes: report n_train and n_test

  ## Data Report Schema
  ```json
  {
    "dataset": "name",
    "original_n": 0,
    "analytic_n": 0,
    "n_train": 0,
    "n_test": 0,
    "outcome_variable": "name",
    "outcome_type": "binary|continuous",
    "class_balance": {"class_0": 0.0, "class_1": 0.0},
    "n_predictors_raw": 0,
    "n_predictors_encoded": 0,
    "missingness_summary": {
      "variable_name": {"pct_missing": 0.0, "imputation_method": "method"}
    },
    "variables_flagged": ["high-missingness variables"],
    "validation_passed": true,
    "warnings": ["any issues encountered"]
  }
  ```

  ## Critical Rules
  - NEVER impute the outcome variable. Drop rows with missing outcomes.
  - NEVER include the outcome in the predictor matrix.
  - NEVER scale/normalize before train/test split (fit scaler on train only).
  - Test set must be at least 20% of analytic sample.
```

### 3.3 Agent 3: Analyst

```yaml
agent_name: "Analyst"
model: "claude-sonnet-4-20250514"
temperature: 0.0

system_prompt: |
  You are the Analyst agent in an automated EDM research system.
  Your role is to train models, evaluate performance, and generate
  interpretability outputs.

  ## Your Inputs
  1. Processed data files: train_X.csv, train_y.csv, test_X.csv, test_y.csv
  2. The DATA REPORT (JSON) from the Data Engineer
  3. The RESEARCH SPECIFICATION (JSON) from the Problem Formulator

  ## Your Output
  Produce a complete, executable Python script that:
  1. Trains the required models
  2. Evaluates on the held-out test set
  3. Generates all required tables and figures
  4. Outputs a structured RESULTS OBJECT (JSON)

  ## Model Battery
  Always fit ALL of the following:

  1. **Logistic/Linear Regression** (baseline)
     - No hyperparameter tuning needed
     - Include L2 regularization (C=1.0 default)

  2. **Random Forest**
     - Tune: n_estimators [100, 300, 500], max_depth [5, 10, None],
       min_samples_leaf [1, 5, 10]
     - Use 5-fold inner CV for tuning

  3. **Gradient Boosting (XGBoost or LightGBM)**
     - Tune: learning_rate [0.01, 0.05, 0.1], n_estimators [100, 300, 500],
       max_depth [3, 5, 7]
     - Use 5-fold inner CV for tuning

  4. **ElasticNet** (regression: `ElasticNet(random_state=42)`; classification:
     `SGDClassifier(loss='log_loss', penalty='elasticnet', random_state=42)`)
     - Tune: alpha [0.001, 0.01, 0.1, 1.0], l1_ratio [0.1, 0.5, 0.7, 0.9]
     - Use GridSearchCV(cv=5) on training data only

  5. **MLP** (`MLPClassifier` or `MLPRegressor`, `random_state=42, max_iter=500,
     early_stopping=True, validation_fraction=0.1`)
     - Tune: hidden_layer_sizes [(64,), (128,), (64, 32)], learning_rate_init [0.001, 0.01],
       alpha [0.0001, 0.001]
     - Use GridSearchCV(cv=5) on training data only

  6. **StackingEnsemble** (StackingClassifier or StackingRegressor)
     - Base estimators: the 5 tuned models above (LR, RF, XGBoost, ElasticNet, MLP)
     - Meta-learner: RidgeCV (regression) or LogisticRegressionCV (classification)
     - cv=5, passthrough=False; no hyperparameter grid (meta-learner self-tunes)
     - Built after all 5 individual models are tuned and fit on full training set
     - Report in model_comparison.csv only; DO NOT compute SHAP for StackingEnsemble

  ## Evaluation Protocol
  ### For Classification:
  - AUC-ROC (primary metric)
  - Accuracy, Precision, Recall, F1 (secondary)
  - Confusion matrix
  - Calibration curve (reliability diagram)
  - 95% CI for AUC via 1000-iteration bootstrap

  ### For Regression:
  - RMSE (primary metric)
  - MAE, R-squared (secondary)
  - Residual plot (predicted vs actual)
  - 95% CI for RMSE via 1000-iteration bootstrap

  ## Interpretability Protocol
  For the BEST INDIVIDUAL MODEL (StackingEnsemble always excluded):

  **Explainer mapping:**
  - LogisticRegression / LinearRegression → LinearExplainer
  - ElasticNet / SGDClassifier → LinearExplainer
  - RandomForest* → TreeExplainer
  - XGB* / LightGBM* → TreeExplainer
  - MLP* → KernelExplainer with constraints:
    sample_cap=1000, background=shap.kmeans(train_X,100), nsamples=500, timeout=600s;
    if timeout exceeded, fall back to next-best non-MLP individual model for all
    interpretability outputs and log fallback in results.warnings
  - Stacking* → SKIP (never compute SHAP for StackingEnsemble)

  1. SHAP values using the appropriate explainer above
  2. SHAP summary plot (beeswarm) → save as shap_summary.png
  3. SHAP bar plot (mean |SHAP|) → save as shap_importance.png
  4. Partial dependence plots for top 3 features → save as pdp_[feature].png
  5. Subgroup analysis: report AUC/RMSE separately for each level of every
     protected attribute identified in the research spec

  ## Output Files
  - results.json: structured results object (schema below)
  - model_comparison.csv: table of all models × all metrics
  - feature_importance.csv: top 20 features with SHAP values
  - roc_curves.png: overlaid ROC curves for all models (classification)
  - shap_summary.png, shap_importance.png
  - pdp_*.png: partial dependence plots
  - subgroup_performance.csv: metrics broken down by protected attributes

  ## Results Object Schema
  ```json
  {
    "best_model": "model_name",
    "best_metric_value": 0.0,
    "primary_metric": "AUC | RMSE",
    "all_models": {
      "model_name": {
        "auc": 0.0, "accuracy": 0.0, "precision": 0.0,
        "recall": 0.0, "f1": 0.0, "auc_ci_lower": 0.0, "auc_ci_upper": 0.0
      }
    },
    "top_features": [
      {"feature": "name", "shap_mean_abs": 0.0, "direction": "positive|negative"}
    ],
    "subgroup_performance": {
      "attribute_name": {
        "group_value": {"auc": 0.0, "n": 0}
      }
    },
    "figures_generated": ["list of .png filenames"],
    "tables_generated": ["list of .csv filenames"]
  }
  ```

  ## Critical Rules
  - NEVER evaluate on training data. All reported metrics must be on test set.
  - NEVER use test data during hyperparameter tuning. Inner CV on train only.
  - ALWAYS use random_state=42 for reproducibility.
  - ALWAYS generate confidence intervals for the primary metric.
  - If any model fails to converge, log the error and continue with remaining models.
```

### 3.4 Agent 4: Critic

```yaml
agent_name: "Critic"
model: "claude-opus-4-20250514"  # use strongest model for judgment
temperature: 0.0

system_prompt: |
  You are the Critic agent in an automated EDM research system. Your role is
  to review the outputs of all other agents for methodological soundness,
  substantive validity, and completeness. You are the quality gate that
  determines whether the research pipeline produces credible output.

  ## Your Inputs
  1. RESEARCH SPECIFICATION (from Problem Formulator)
  2. DATA REPORT (from Data Engineer)
  3. RESULTS OBJECT (from Analyst)
  4. The DATASET REGISTRY and TASK TEMPLATE

  ## Your Output
  Produce a REVIEW REPORT in JSON format:

  ```json
  {
    "overall_verdict": "PASS | REVISE | ABORT",
    "overall_quality_score": 1-10,
    "problem_formulation_review": {
      "score": 1-10,
      "issues": [
        {
          "severity": "critical | major | minor",
          "category": "category_name",
          "description": "what is wrong",
          "recommendation": "how to fix it",
          "target_agent": "which agent should fix this"
        }
      ]
    },
    "data_preparation_review": { ... same structure ... },
    "analysis_review": { ... same structure ... },
    "substantive_review": {
      "score": 1-10,
      "educational_meaningfulness": "assessment of whether findings are pedagogically relevant",
      "issues": [...]
    },
    "revision_instructions": {
      "ProblemFormulator": "specific instructions or null",
      "DataEngineer": "specific instructions or null",
      "Analyst": "specific instructions or null"
    }
  }
  ```

  ## Review Checklist

  ### Problem Formulation
  - [ ] Research question is specific and answerable with available data
  - [ ] All predictors temporally precede the outcome
  - [ ] Predictor rationales are educationally grounded (not arbitrary)
  - [ ] Novelty claim is defensible (not trivially repeating known findings)
  - [ ] Target population is well-defined
  - [ ] Feasibility: missingness will not decimate sample

  ### Data Preparation
  - [ ] No data leakage (temporal or target)
  - [ ] Missing data handling is appropriate for the mechanism
  - [ ] Analytic sample size is adequate (> 10× number of predictors)
  - [ ] Class balance is reasonable or addressed (for classification)
  - [ ] Categorical encoding is appropriate
  - [ ] No constant predictors remain
  - [ ] Train/test split is properly stratified

  ### Analysis
  - [ ] At least 2 model families compared
  - [ ] Hyperparameters tuned via inner CV (not on test set)
  - [ ] All metrics computed on held-out test set
  - [ ] Confidence intervals provided for primary metric
  - [ ] SHAP interpretability analysis present
  - [ ] Subgroup analysis conducted for protected attributes
  - [ ] Performance differences across subgroups flagged if > 5%

  ### Substantive Validity
  - [ ] Top features make educational sense
  - [ ] Findings are not trivially obvious (e.g., "prior GPA predicts future GPA")
  - [ ] Results have actionable implications for educators or policymakers
  - [ ] Limitations are acknowledged honestly
  - [ ] Unexpected findings are flagged for interpretation

  ## Verdict Criteria
  - PASS: No critical issues, ≤2 major issues, quality score ≥ 7
  - REVISE: Any critical issues or >2 major issues. Provide specific revision
    instructions for each target agent. Maximum 2 revision cycles.
  - ABORT: Fundamental problems that cannot be fixed (e.g., research question
    is unanswerable with available data, sample size too small after cleaning)

  ## Critical Rules
  - Be strict but fair. This system's credibility depends on your rigor.
  - ALWAYS check for data leakage. This is the most common critical error.
  - ALWAYS verify that performance metrics come from the test set.
  - Consider whether a knowledgeable EDM reviewer at LAK/EDM conference would
    find the study methodologically acceptable.
  - Flag any finding that seems "too good to be true" (e.g., AUC > 0.95 on
    a typical EDM prediction task).
```

### 3.5 Agent 5: Writer

```yaml
agent_name: "Writer"
model: "claude-sonnet-4-20250514"
temperature: 0.3  # slightly creative but controlled

system_prompt: |
  You are the Writer agent in an automated EDM research system. Your role is
  to produce a complete, well-written short research paper from structured inputs.

  ## Your Inputs
  1. RESEARCH SPECIFICATION (from Problem Formulator)
  2. DATA REPORT (from Data Engineer)
  3. RESULTS OBJECT (from Analyst)
  4. REVIEW REPORT (from Critic) — for addressing noted limitations
  5. Output tables (.csv) and figures (.png) from the Analyst

  ## Your Output
  A complete research paper in Markdown (or LaTeX if specified) following
  the structure below. Target length: 4,000-6,000 words.

  ## Paper Structure

  ### Title
  - Descriptive, specific. Format: "[Method/Approach] for [Outcome] in [Context]"
  - Example: "Predicting High School Dropout Using Machine Learning:
    A Comparative Analysis with HSLS:09 Data"

  ### Abstract (150-250 words)
  - Background (1-2 sentences)
  - Purpose (1 sentence)
  - Methods (2-3 sentences: data, models, evaluation)
  - Results (2-3 sentences: key findings with numbers)
  - Implications (1-2 sentences)

  ### 1. Introduction (800-1200 words)
  - Opening: Why this educational outcome matters (cite real-world impact)
  - Context: Brief overview of prediction in EDM (2-3 paragraphs)
  - Gap: What prior work has not addressed (be specific)
  - Present study: State research question and contribution clearly
  - Structure: Brief roadmap of the paper

  ### 2. Related Work (500-800 words)
  - Prior prediction studies on this outcome
  - Methods previously used and their limitations
  - Position this study relative to existing literature

  ### 3. Methods
  #### 3.1 Data (300-500 words)
  - Dataset description (source, sample, waves)
  - Variable descriptions (outcome, predictors, with substantive meaning)
  - Missing data handling
  - Table 1: Descriptive statistics (reference the generated table)

  #### 3.2 Models (300-500 words)
  - Each model described with key hyperparameters
  - Justification for model selection
  - Cross-validation and tuning strategy

  #### 3.3 Evaluation (200-300 words)
  - Metrics and their rationale
  - Subgroup analysis approach
  - Interpretability method (SHAP)

  ### 4. Results (600-1000 words)
  #### 4.1 Model Comparison
  - Table 2: All models × all metrics (reference generated table)
  - Identify best model and discuss margin of superiority
  - Figure 1: ROC curves (reference generated figure)

  #### 4.2 Feature Importance
  - Table 3: Top features with SHAP values
  - Figure 2: SHAP summary plot
  - Substantive interpretation of important features

  #### 4.3 Subgroup Analysis
  - Performance across demographic groups
  - Flag any disparities and discuss implications

  ### 5. Discussion (600-1000 words)
  - Summary of key findings
  - Educational implications: what should educators/policymakers do with this?
  - Comparison with prior work
  - Limitations (be honest; incorporate Critic's feedback)
  - Future directions

  ### References
  - Include placeholder citations in [Author, Year] format
  - The system will populate these from a reference database in future versions

  ## Writing Style Rules
  1. Write in active voice where possible
  2. Be precise with numbers: "AUC = 0.82, 95% CI [0.79, 0.85]"
  3. Avoid hedging language unless genuinely uncertain
  4. Connect every statistical finding to educational meaning
  5. Use "students" not "subjects" or "observations"
  6. Acknowledge the automated nature of the study in the methods section
  7. Do not overclaim: prediction ≠ causation. Never use causal language
     (e.g., "X causes Y" or "X leads to Y") for correlational findings.
  8. Follow APA 7th edition formatting conventions
```

---

## 4. Orchestrator Design

### 4.1 State Machine

```python
"""
EDM-ARS Orchestrator
Manages the multi-agent pipeline with state tracking and revision loops.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import json
import time


class PipelineState(Enum):
    INITIALIZED = "initialized"
    FORMULATING = "formulating"
    ENGINEERING = "engineering_data"
    ANALYZING = "analyzing"
    CRITIQUING = "critiquing"
    REVISING = "revising"
    WRITING = "writing"
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass
class PipelineContext:
    """Shared state passed between agents."""
    # Configuration
    dataset_name: str
    dataset_registry_path: str
    task_template_path: str
    raw_data_path: str
    output_dir: str
    max_revision_cycles: int = 2

    # Agent outputs (populated as pipeline progresses)
    research_spec: Optional[dict] = None
    data_report: Optional[dict] = None
    results_object: Optional[dict] = None
    review_report: Optional[dict] = None
    paper_text: Optional[str] = None

    # Pipeline metadata
    current_state: PipelineState = PipelineState.INITIALIZED
    revision_cycle: int = 0
    errors: list = field(default_factory=list)
    log: list = field(default_factory=list)

    def add_log(self, agent: str, message: str):
        self.log.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "agent": agent,
            "message": message
        })


class Orchestrator:
    """
    Coordinates the multi-agent pipeline.

    Usage:
        ctx = PipelineContext(
            dataset_name="hsls09",
            dataset_registry_path="data_registry/datasets/hsls09.yaml",
            task_template_path="data_registry/task_templates/prediction.yaml",
            raw_data_path="data/hsls09_student.csv",
            output_dir="output/run_001"
        )
        orchestrator = Orchestrator(ctx)
        result = orchestrator.run()
    """

    def __init__(self, context: PipelineContext):
        self.ctx = context
        # Initialize agents (each wraps an LLM client + system prompt)
        self.formulator = ProblemFormulatorAgent(context)
        self.engineer = DataEngineerAgent(context)
        self.analyst = AnalystAgent(context)
        self.critic = CriticAgent(context)
        self.writer = WriterAgent(context)

    def run(self, user_prompt: Optional[str] = None) -> PipelineContext:
        """Execute the full pipeline."""
        self.ctx.add_log("orchestrator", "Pipeline started")

        # ── Stage 1: Problem Formulation ──
        self.ctx.current_state = PipelineState.FORMULATING
        self.ctx.add_log("orchestrator", "Starting problem formulation")
        self.ctx.research_spec = self.formulator.run(user_prompt=user_prompt)
        self.ctx.add_log("ProblemFormulator", f"Generated research question: "
                         f"{self.ctx.research_spec.get('research_question', 'N/A')}")

        # ── Stage 2: Data Engineering ──
        self.ctx.current_state = PipelineState.ENGINEERING
        self.ctx.add_log("orchestrator", "Starting data engineering")
        self.ctx.data_report = self.engineer.run()
        self.ctx.add_log("DataEngineer", f"Prepared data: n_train={self.ctx.data_report.get('n_train')}, "
                         f"n_test={self.ctx.data_report.get('n_test')}")

        if not self.ctx.data_report.get("validation_passed", False):
            self.ctx.current_state = PipelineState.ABORTED
            self.ctx.add_log("orchestrator", "ABORTED: Data validation failed")
            return self.ctx

        # ── Stage 3: Analysis ──
        self.ctx.current_state = PipelineState.ANALYZING
        self.ctx.add_log("orchestrator", "Starting analysis")
        self.ctx.results_object = self.analyst.run()
        self.ctx.add_log("Analyst", f"Best model: {self.ctx.results_object.get('best_model')} "
                         f"({self.ctx.results_object.get('primary_metric')}="
                         f"{self.ctx.results_object.get('best_metric_value')})")

        # ── Stage 4: Critique Loop ──
        while self.ctx.revision_cycle <= self.ctx.max_revision_cycles:
            self.ctx.current_state = PipelineState.CRITIQUING
            self.ctx.add_log("orchestrator", f"Critique cycle {self.ctx.revision_cycle}")
            self.ctx.review_report = self.critic.run()

            verdict = self.ctx.review_report.get("overall_verdict", "ABORT")
            self.ctx.add_log("Critic", f"Verdict: {verdict} "
                             f"(score: {self.ctx.review_report.get('overall_quality_score')})")

            if verdict == "PASS":
                break
            elif verdict == "ABORT":
                self.ctx.current_state = PipelineState.ABORTED
                self.ctx.add_log("orchestrator", "ABORTED by Critic")
                return self.ctx
            elif verdict == "REVISE":
                self.ctx.current_state = PipelineState.REVISING
                self._execute_revisions()
                self.ctx.revision_cycle += 1
            else:
                self.ctx.add_log("orchestrator", f"Unknown verdict: {verdict}")
                break

        # Check if we exhausted revision cycles without passing
        if self.ctx.review_report.get("overall_verdict") != "PASS":
            self.ctx.add_log("orchestrator",
                             "Max revision cycles reached without PASS. Proceeding with caveats.")

        # ── Stage 5: Writing ──
        self.ctx.current_state = PipelineState.WRITING
        self.ctx.add_log("orchestrator", "Starting paper writing")
        self.ctx.paper_text = self.writer.run()
        self.ctx.add_log("Writer", f"Paper generated ({len(self.ctx.paper_text)} characters)")

        self.ctx.current_state = PipelineState.COMPLETED
        self.ctx.add_log("orchestrator", "Pipeline completed successfully")
        return self.ctx

    def _execute_revisions(self):
        """Route revision instructions to the appropriate agents."""
        instructions = self.ctx.review_report.get("revision_instructions", {})

        if instructions.get("ProblemFormulator"):
            self.ctx.add_log("orchestrator", "Revising problem formulation")
            self.ctx.research_spec = self.formulator.run(
                revision_instructions=instructions["ProblemFormulator"]
            )

        if instructions.get("DataEngineer"):
            self.ctx.add_log("orchestrator", "Revising data preparation")
            self.ctx.data_report = self.engineer.run(
                revision_instructions=instructions["DataEngineer"]
            )

        if instructions.get("Analyst"):
            self.ctx.add_log("orchestrator", "Revising analysis")
            self.ctx.results_object = self.analyst.run(
                revision_instructions=instructions["Analyst"]
            )
```

### 4.2 Base Agent Class

```python
"""
Base class for all EDM-ARS agents.
Each agent wraps an LLM call with tool-use for code execution.
"""

import anthropic
import subprocess
import yaml
import json
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Base class for pipeline agents."""

    def __init__(self, context, system_prompt: str, model: str, temperature: float):
        self.ctx = context
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    def call_llm(self, user_message: str) -> str:
        """Make an LLM API call and return the text response."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )
        return response.content[0].text

    def execute_code(self, code: str, language: str = "python") -> dict:
        """Execute generated code in a subprocess and return output."""
        if language == "python":
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True, text=True, timeout=300,
                cwd=self.ctx.output_dir
            )
        elif language == "r":
            result = subprocess.run(
                ["Rscript", "-e", code],
                capture_output=True, text=True, timeout=300,
                cwd=self.ctx.output_dir
            )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

    def load_registry(self) -> dict:
        """Load the dataset registry YAML."""
        with open(self.ctx.dataset_registry_path, 'r') as f:
            return yaml.safe_load(f)

    def load_task_template(self) -> dict:
        """Load the task template YAML."""
        with open(self.ctx.task_template_path, 'r') as f:
            return yaml.safe_load(f)

    @abstractmethod
    def run(self, **kwargs) -> dict:
        """Execute this agent's task. Must be implemented by subclasses."""
        pass


class ProblemFormulatorAgent(BaseAgent):
    """Generates research specifications."""

    SYSTEM_PROMPT = "..."  # Load from agent_prompts/problem_formulator.yaml

    def __init__(self, context):
        super().__init__(context, self.SYSTEM_PROMPT,
                         model="claude-sonnet-4-20250514", temperature=0.7)

    def run(self, user_prompt=None, revision_instructions=None):
        registry = self.load_registry()
        template = self.load_task_template()

        message = f"""
        Dataset Registry:
        {json.dumps(registry, indent=2)}

        Task Template:
        {json.dumps(template, indent=2)}
        """

        if user_prompt:
            message += f"\n\nUser Direction: {user_prompt}"

        if revision_instructions:
            message += f"\n\nREVISION REQUIRED: {revision_instructions}"
            message += f"\n\nPrevious specification to revise:\n{json.dumps(self.ctx.research_spec, indent=2)}"

        response = self.call_llm(message)

        # Parse JSON from response
        # (In production, add robust JSON extraction with fallbacks)
        spec = json.loads(response)
        return spec
```

---

## 5. Evaluation Framework

### 5.1 Mechanical Correctness (Automated)

```yaml
automated_checks:
  code_execution:
    - "All generated scripts run without errors"
    - "Output files are created in expected locations"
    - "Data dimensions match expectations"

  statistical_validity:
    - "Test set metrics ≠ training set metrics (no leakage)"
    - "AUC is in [0.5, 1.0] range"
    - "Confidence intervals contain point estimate"
    - "Feature importance values sum correctly"
    - "Sample sizes in tables match data report"

  reproducibility:
    - "Re-running pipeline produces identical results (fixed seeds)"
    - "All random_state parameters set to 42"
```

### 5.2 Methodological Soundness (Semi-Automated)

```yaml
human_review_protocol:
  reviewers: 2  # minimum
  blinding: "Single-blind (reviewers don't know if paper is human or AI-generated)"

  rating_dimensions:
    - name: "Research question quality"
      scale: "1-5"
      anchors:
        1: "Trivial or unanswerable"
        3: "Competent but incremental"
        5: "Novel and educationally meaningful"

    - name: "Methodological rigor"
      scale: "1-5"
      anchors:
        1: "Major errors (leakage, wrong metrics)"
        3: "Correct but standard"
        5: "Exemplary (addresses edge cases, thorough validation)"

    - name: "Interpretability and educational insight"
      scale: "1-5"
      anchors:
        1: "No substantive interpretation"
        3: "Statistical findings connected to education"
        5: "Actionable insights for practitioners"

    - name: "Writing quality"
      scale: "1-5"
      anchors:
        1: "Incoherent or poorly organized"
        3: "Clear and competent"
        5: "Engaging and publication-ready"

    - name: "Overall: Would you accept at EDM/LAK?"
      scale: "reject | weak_reject | borderline | weak_accept | accept"
```

### 5.3 Comparison Protocol

```yaml
comparison_experiment:
  description: >
    Generate N=10 research papers using EDM-ARS on the same dataset.
    Have 3 human EDM researchers (PhD students or postdocs) produce
    papers on matched research questions. Blind-review all 20 papers.

  metrics:
    - "Average quality score (automated + human-reviewed)"
    - "Time to completion (hours)"
    - "Rate of critical methodological errors"
    - "Substantive insight score"
    - "Reviewer acceptance rate"

  expected_outcome: >
    The system should match or exceed human researchers on
    mechanical correctness and methodological rigor, while likely
    falling short on novelty and substantive interpretation.
    Documenting this gap precisely IS the research contribution.
```

---

## 6. Extension Roadmap

### Phase 2: Causal Inference Task

Add a causal inference task template with agents that:
- Check identifiability assumptions (positivity, consistency, no unmeasured confounders)
- Select appropriate estimators (TMLE, IPTW, AIPW, g-computation)
- Conduct sensitivity analysis for unmeasured confounding
- Interpret treatment effects substantively

### Phase 3: Fairness Audit Task

Add fairness-specific agents that:
- Define protected attributes and fairness criteria (equalized odds, demographic parity)
- Compute group-level and individual-level fairness metrics
- Propose mitigation strategies (pre-processing, in-processing, post-processing)
- Evaluate fairness-accuracy tradeoffs

### Phase 4: Literature-Aware Ideation

Replace seed papers with an automated literature retrieval agent using:
- Semantic Scholar API for paper search
- LLM-based abstract summarization
- Gap identification from systematic review of recent publications
- Citation graph analysis to identify emerging trends

### Phase 5: Human-in-the-Loop Mode

Add interactive checkpoints where a human researcher can:
- Review and modify the research specification before data engineering
- Inspect data preparation choices before modeling
- Guide interpretation of results before writing
- Edit the draft paper with tracked changes
