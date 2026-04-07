# UC1 — Distribution Analysis
### Diabetes 130-US Hospitals Dataset: Understanding the Data Before Federated Experiments

---

## Why This Analysis Exists

Before running any federated learning experiment, I need to deeply understand the structure of the data I am working with. This is not just good practice — it is essential for interpreting the results that come later. When FedAvg underperforms at a given heterogeneity level, or when FedGen's generator fails to close the gap, the explanation will almost always live in the data: which features drive the target, how they are distributed, and how those distributions change when the data is split across simulated hospitals.

This document walks through every finding from the distribution analysis notebook in the order it was computed, explaining what each result means clinically, statistically, and in the context of the federated learning problem.

---

## 1. The Dataset

The dataset used throughout this thesis is the **Diabetes 130-US Hospitals for Years 1999–2008** dataset, introduced by Strack et al. (2014). It contains clinical records from 130 US hospitals over a ten-year period, covering diabetic patients who had a hospital stay lasting between one and fourteen days. Each row represents a single encounter, and the task is to predict whether that patient will be readmitted to any hospital within 30 days of discharge.

After the full preprocessing pipeline described below, the working dataset has the following properties:

```
Total encounters:   97,825
Total features:     99
Positive rate:      11.46%   (readmitted within 30 days)
Negative rate:      88.54%   (not readmitted within 30 days)
Class imbalance:    7.73 : 1
```

The dataset starts with 101,766 raw rows. The difference comes from two sources. First, 2,423 encounters belonging to patients who died during the encounter are removed, because a deceased patient cannot be readmitted — including them would introduce impossible-to-predict positive cases. Second, 1,518 rows (1.53%) are dropped after the main cleaning step because they contain admission type or source codes that do not appear in the standard ICD mapping dictionaries used by Strack et al. (2014). These unmapped codes represent less than 2% of the data, and imputing them as "unknown" would mean fabricating a clinical category that the original authors did not define. Dropping them is the safer and more honest choice.

---

## 2. Feature Description and Preprocessing

The raw dataset has 55 features before any engineering. They fall into six categories. Below I describe each group, its preprocessing, and the reasoning behind every decision.

### 2.1 Columns Dropped Entirely

Three columns are dropped before any analysis:

- **`weight`**: missing in 97% of encounters (Strack et al., 2014, Table 1). There is no meaningful imputation strategy for a column that is almost always absent.
- **`payer_code`**: missing in 52% of encounters and carries no clinical signal relevant to readmission prediction.
- **`encounter_id`**: a pure administrative identifier with no predictive value.

### 2.2 Demographic Features

**`age`** is provided as a categorical decade bracket (`[0-10)`, `[10-20)`, ..., `[90-100)`). I map it to ordinal integers 0–9 following Strack et al. (2014, Table 1). The ordinal encoding is appropriate here because age brackets have a natural order and the spacing between them is uniform.

**`gender`** is binary in the raw data (Male/Female). It is one-hot encoded. A small number of rows with invalid gender codes are caught by the `dropna` step.

**`race`** has approximately 2% missing values (Strack et al., 2014, Table 1). Rather than dropping these rows or imputing with the mode, I fill missing values with `'Unknown'` and then one-hot encode, giving `Unknown` its own dummy column. The rationale is that missingness in a race field often reflects a documentation pattern rather than a random omission — it may carry its own signal about the encounter context.

### 2.3 Administrative Features

These features describe the administrative circumstances of the encounter rather than the patient's clinical state. They all have high cardinality in the raw data.

**`admission_type_id`** has 8 possible integer codes. I group them into 5 clinically meaningful categories: `emergency`, `urgent`, `elective`, `newborn`, and `unknown` (Strack et al., 2014, Table 1). Codes 5, 6, and 8 are all described as "Not Available" or "Not Mapped" in the original IDS mapping file and are grouped as `unknown`.

**`admission_source_id`** has 21 possible codes, grouped into: `referral`, `transfer`, `emergency`, `newborn`, `unknown`, and `other`. This follows the same IDS mapping approach (Strack et al., 2014).

**`discharge_disposition_id`** has 29 possible codes, which I group into: `home`, `transfer`, `expired`, `hospice`, `ama` (against medical advice), `inpatient`, and `unknown`. The grouping is clinically motivated — for example, a patient discharged to a transfer facility has a very different readmission risk profile than one sent home.

All three are one-hot encoded after grouping.

### 2.4 Diagnosis Codes

**`diag_1`**, **`diag_2`**, and **`diag_3`** record the primary and two secondary ICD-9 diagnosis codes for each encounter. In the raw data these are free-form strings with thousands of possible values. I group them into 8 clinically meaningful categories following Strack et al. (2014, Table 2):

| Category | ICD-9 Range |
|---|---|
| `circulatory` | 390–459, 785 |
| `respiratory` | 460–519, 786 |
| `digestive` | 520–579, 787 |
| `diabetes` | 250–250.99 |
| `injury` | 800–999 |
| `musculoskeletal` | 710–739 |
| `genitourinary` | 580–629, 788 |
| `neoplasms` | 140–239 |
| `other` | everything else, V-codes, E-codes |

Codes starting with V or E (supplementary codes) and any non-numeric strings are mapped to `other`. Missing values are handled inside the mapping function itself — a missing diagnosis code returns `other` rather than triggering a row drop. All three diagnosis columns are one-hot encoded after grouping.

### 2.5 Laboratory Test Features

**`A1Cresult`** records the result of an HbA1c blood test, with possible values `>7`, `>8`, `Norm` (normal), and `None` (test not ordered). In the raw CSV, `?` indicates the test was not ordered — this is loaded as `NaN` and filled with `'none'` to match the existing valid category. This is a critical imputation decision: Strack et al. (2014) show that *ordering* the HbA1c test at all is associated with significantly lower readmission rates (p < 0.001, Table 5), regardless of the actual result. The `none` category is not a missing value — it is the clinical state of a patient for whom the test was not ordered, and it must be retained as its own category. The HbA1c test was ordered in only 18.4% of encounters in this dataset (Strack et al., 2014).

**`max_glu_serum`** records the result of a glucose serum test, with values `>200`, `>300`, `Norm`, and `None`. Identical logic applies — `?` in the raw CSV means the test was not ordered, filled with `'none'`, and treated as a clinically meaningful category rather than missingness.

Both features are one-hot encoded.

### 2.6 Medical Specialty

**`medical_specialty`** records the specialty of the admitting physician. It is missing in 53% of encounters (Strack et al., 2014, Table 1) — the single highest missingness rate in the dataset. I handle this in two steps. First, missing values are filled with `'Unknown'`, giving them their own dummy column after OHE. The `Unknown` category corresponds to 48.1% of encounters and is itself a statistically significant predictor in Strack et al.'s logistic regression model (coefficient 0.463, p = 0.002, Table 4). Second, specialties appearing in fewer than 1% of encounters (~970 rows) are grouped into `Other_specialty`. Without this grouping, the raw 84 specialty categories would produce many near-zero-variance dummy columns that add noise without contributing signal. After grouping, the specialty column produces approximately 10 meaningful dummy columns.

### 2.7 Medication Features

The dataset contains 23 medication columns recording whether each drug dosage was `No` (not prescribed), `Steady` (prescribed, no change), `Up` (dose increased), or `Down` (dose decreased). I binary-encode them: `No` and `Steady` → 0 (medication unchanged), `Up` and `Down` → 1 (medication actively adjusted). The direction of change is not retained because the readmission signal lies in whether the medication was actively managed, not in which direction it was adjusted (Sazdov et al., 2023).

Ten of the 23 medications are dropped after binary encoding because they appear with a dosage change in fewer than 0.1% of encounters: `acetohexamide`, `tolbutamide`, `troglitazone`, `tolazamide`, `examide`, `citoglipton`, `glipizide-metformin`, `glimepiride-pioglitazone`, `metformin-rosiglitazone`, and `metformin-pioglitazone`. These drugs are near-constant columns after encoding — they add noise without predictive value (Sazdov et al., 2023, Section IV-C). The remaining 13 medication columns are retained.

Two binary flags are also included: **`change`** (1 if any medication was changed during the encounter, 0 otherwise) and **`diabetesMed`** (1 if the patient was prescribed any diabetes medication, 0 otherwise).

### 2.8 Numeric Clinical Features

These are the raw count features from the encounter record, already numeric in the source data:

| Feature | Description |
|---|---|
| `time_in_hospital` | Length of stay in days (1–14) |
| `num_lab_procedures` | Number of lab tests performed |
| `num_procedures` | Number of non-lab procedures performed |
| `num_medications` | Number of distinct medications administered |
| `number_outpatient` | Outpatient visits in the 12 months prior |
| `number_emergency` | Emergency visits in the 12 months prior |
| `number_inpatient` | Inpatient visits in the 12 months prior |
| `number_diagnoses` | Number of diagnoses recorded for this encounter |

### 2.9 Engineered Features

Three features are added on top of the raw dataset, motivated by prior literature:

**`service_utilization`** is defined as `number_inpatient + number_outpatient + number_emergency`. It captures the total volume of healthcare contacts in the 12 months preceding this encounter. It is identified as a high-importance feature by both Sazdov et al. (2023) and Jauhari et al. (2021). Since all three source columns encode a trailing yearly window, this is a row-wise sum with no temporal ordering required.

**`medication_count`** is the number of medications actively adjusted during this encounter — the sum of binary-encoded medication columns. It is computed *before* dropping the near-zero-variance medications, so the count reflects all 23 original drugs. This is important: the count represents "how much medication management happened during this stay", and dropping rare drugs should not artificially deflate that count. Identified as high-importance by Sazdov et al. (2023) and Goudjerkan & Jayabalan (2019).

**`HbA1c_diabetes_interaction`** is a binary flag set to 1 when the HbA1c test was ordered *and* the primary diagnosis was diabetes mellitus. Strack et al. (2014) identify this as the strongest interaction term in their logistic regression model (p < 0.001, Table 5). The feature requires two conditions to hold simultaneously: `A1Cresult != 'none'` (test was ordered) and `diag_1 == 'diabetes'` (primary diagnosis is diabetes). It is computed after the ICD-9 grouping step, because the diagnosis grouping must have already mapped `diag_1` to the string `'diabetes'` before this check is valid.

### 2.10 Final Dataset Dimensions

The pipeline produces the following transformation:

```
Raw CSV:          101,766 rows × 55 columns
After cleaning:    97,825 rows × 40 columns  (pre-OHE)
After encoding:    97,825 rows × 99 features + 1 target
```

The jump from 40 to 99 columns is entirely due to one-hot encoding expanding categorical variables into binary dummy columns. The 59 new columns carry structural clinical information but in a sparse, low-signal-per-column way — for any given encounter, most OHE columns will be zero. The 13 numeric features carry the bulk of the discriminative variance.

---

## 3. Class Imbalance

```
Class 0 (not readmitted within 30 days):   86,618   →   88.54%
Class 1 (readmitted within 30 days):       11,207   →   11.46%
Imbalance ratio:                            7.73 : 1
```

Only 11.46% of encounters end in a 30-day readmission. At first glance this seems manageable, but it is worth thinking carefully about what this means in the context of federated learning.

### 3.1 Why This Is a Problem in the Centralized Setting

If I built the simplest imaginable model — one that always predicts "not readmitted" regardless of input — it would be correct 88.54% of the time. This is why accuracy is a useless metric for this problem and I use AUC-ROC instead. AUC-ROC measures whether the model correctly ranks a randomly chosen positive case above a randomly chosen negative case. It is completely insensitive to the class balance and rewards the model for its *discriminative ability*, not for predicting the majority class.

Class imbalance is addressed in two ways. For XGBoost, the `scale_pos_weight` parameter is set to the imbalance ratio (7.73), which tells the model to treat each positive example as if it appeared 7.73 times. For the MLP, `compute_class_weight('balanced')` is used to assign higher loss weights to positive examples during training.

### 3.2 Why This Is a Bigger Problem in the Federated Setting

In the centralized setting, all 11,207 positive examples are always visible. The model is exposed to the full diversity of readmission-prone patients at every training step. In federated learning, the positive cases are partitioned across 5 clients according to the Dirichlet distribution. Even in the near-iid case (α = 5.0), each client gets roughly 1/5 of the positive cases — approximately 2,241. In the heterogeneous cases, some clients get far fewer. A client with very few positive examples cannot learn the patterns that distinguish readmitted from non-readmitted patients, no matter how sophisticated the model. This is the core data problem that federated learning on medical data must confront.

---

## 4. Class-Conditional Feature Distributions: Continuous Features

For each continuous feature, I ask: does this feature look different between patients who get readmitted and patients who do not? To answer this formally I use the **Kolmogorov-Smirnov (KS) test**.

The KS statistic is defined as:

$$D = \sup_x |F_0(x) - F_1(x)|$$

where $F_0$ and $F_1$ are the empirical cumulative distribution functions of the feature for class 0 and class 1 respectively. The statistic measures the **maximum vertical gap** between the two CDFs at any single point. It answers the question: "at the single value of this feature where the two classes diverge most, how large is that divergence?"

A key property of the KS test is that it only cares about the single worst point — it does not integrate over the full distribution. A feature that differs dramatically at one threshold but is similar elsewhere will produce a high KS statistic. A feature that differs moderately everywhere will produce a more modest value despite representing greater total divergence.

The p-values in this analysis are essentially uninterpretable for magnitude: with 97,825 samples, the test has near-infinite statistical power and will flag any real difference as significant no matter how tiny. What matters is the **magnitude of the KS statistic**, not the p-value.

### Results

```
Feature                  KS Statistic    p-value      Significant
number_inpatient         0.183           9.7e-292     ***
service_utilization      0.165           1.7e-236     ***
─────────────────────── cliff ────────────────────────────────────
time_in_hospital         0.071           2.1e-44      ***
number_diagnoses         0.069           2.2e-41      ***
num_medications          0.064           7.8e-36      ***
number_emergency         0.061           1.2e-32      ***
medication_count         0.052           6.4e-24      ***
number_outpatient        0.040           2.4e-14      ***
num_lab_procedures       0.036           2.1e-11      ***
age                      0.035           9.6e-11      ***
num_procedures           0.018           4.3e-03      (not significant after correction)
```

### Reading the Results Feature by Feature

**`number_inpatient` (KS = 0.183)** is the strongest individual predictor of 30-day readmission in this dataset. It records how many times the patient was hospitalised as an inpatient in the 12 months before this encounter. Patients who come back within 30 days tend to have more prior hospitalisations. This is clinically intuitive: a patient who has been hospitalised multiple times in the past year is sicker, more fragile, and less stable after discharge. Their prior hospital history is essentially a summary of their disease burden. This feature alone is more than twice as discriminative as anything outside the top two.

**`service_utilization` (KS = 0.165)** is `number_inpatient + number_outpatient + number_emergency` — an engineered feature. Its KS is almost as high as `number_inpatient` because it is largely driven by `number_inpatient`. It is not an independent finding. Both are capturing the same clinical phenomenon: frequent healthcare utilisation predicts readmission. The total utilisation metric amplifies this signal slightly by also counting outpatient and emergency contacts, but the marginal gain over `number_inpatient` alone is modest.

**The cliff between 0.165 and 0.071** is the most structurally important observation in the KS analysis. There are exactly two features with KS above 0.1 (prior utilisation history), and then a gap before the rest of the field. This tells me that my dataset essentially has two tiers of features: prior utilisation history (strongly discriminative) and everything else (weakly to moderately discriminative). No single other feature is anywhere near as informative.

**`time_in_hospital` (KS = 0.071)** — patients who stay longer in hospital are somewhat more likely to be readmitted. This is consistent with longer stays indicating more severe conditions that are harder to stabilise.

**`number_diagnoses` (KS = 0.069)** — encounters where more diagnoses are recorded tend to lead to more readmissions. More diagnoses means a more complex patient with more comorbidities.

**`num_medications` (KS = 0.064)** — patients on more medications are slightly more likely to be readmitted. More medications implies a more complex disease burden.

**`number_emergency` (KS = 0.061)** — prior emergency visits are predictive of readmission, which makes sense since emergency visits also reflect disease instability.

**`medication_count` (KS = 0.052)** — the number of medications actively changed during this encounter has a moderate association with readmission. Active medication management suggests a patient whose condition required titration, implying instability.

**`number_outpatient` (KS = 0.040)**, **`num_lab_procedures` (KS = 0.036)**, **`age` (KS = 0.035)** — all carry statistically detectable but weak signals at the individual level.

**`age`** deserves a comment. It is significant but weak (KS = 0.035). Older patients are slightly more likely to be readmitted, but once you condition on utilisation history the age signal largely disappears. Age is a proxy for frailty, and `number_inpatient` already captures frailty more directly.

**`num_procedures` (KS = 0.018)** — effectively uninformative at the marginal level. The number of procedures performed during this encounter does not predict readmission. What matters is the patient's prior history, not what was done during this stay. This feature would likely not survive a Bonferroni correction for 11 simultaneous tests (corrected threshold: 0.05/11 ≈ 0.0045, corresponding p-value would be much smaller than 4.3e-03 only if the sample is large — in this case it survives barely, but the KS magnitude is so small as to be practically irrelevant).

---

## 5. Class-Conditional Distributions: Categorical Features

Beyond the continuous features, several categorical features show meaningful class-conditional variation. I examined five key groupings after the cardinality-reduction steps described in Section 2.

**`discharge_disposition_id`** shows the clearest categorical signal. Patients discharged to a transfer facility (another care setting) have elevated readmission rates compared to patients discharged home. Patients discharged to hospice have very high readmission rates by definition — they are the sickest, though the clinical meaning of "readmission" for palliative patients is different. Patients who leave against medical advice (`ama`) also show elevated rates because they are discharged without completing treatment.

**`admission_type_id`** — emergency admissions show slightly higher readmission rates than elective admissions. This is consistent with emergency admissions representing acute events that may not be fully resolved by discharge.

**`A1Cresult`** — encounters where the HbA1c test was ordered and returned an abnormal result (`>8`) show elevated readmission rates. Importantly, encounters where the test was `not ordered` (`none`) show *lower* readmission rates than those with abnormal results. This is the mechanism behind the `HbA1c_diabetes_interaction` feature: physicians who identify a poorly controlled diabetic and order the test are already dealing with a higher-risk patient. The act of ordering the test is a signal of clinical concern, not just the result itself (Strack et al., 2014, Table 5).

**`max_glu_serum`** — similar pattern to `A1Cresult`. Abnormal glucose serum levels are associated with higher readmission rates. Encounters where the test was not ordered are associated with lower rates.

**`diag_1`** — the primary diagnosis category is informative. Encounters primarily coded as `circulatory` (heart disease, stroke) or `diabetes` show higher readmission rates than encounters coded as `injury` or `respiratory`. This reflects the chronic, relapsing nature of cardiovascular and metabolic disease.

---

## 6. Engineered Features: Clinical Validation

Having added three engineered features based on the literature, I need to verify that they actually carry the signals their authors claimed. A feature engineered from the literature but unsupported by the data is a noise source, not a signal.

**`service_utilization`** — the distribution of total visits in the 12 prior months differs strongly between classes (confirmed by its KS = 0.165). Patients with more prior contacts across all visit types are meaningfully more likely to be readmitted. The engineering step of summing the three components adds marginal information over `number_inpatient` alone (given that `number_outpatient` and `number_emergency` each have lower individual KS statistics), but the aggregate is easier for the model to use and is consistent with clinical literature (Sazdov et al., 2023; Jauhari et al., 2021).

**`medication_count`** — as `medication_count` increases from 0 to its maximum value, the readmission rate rises above the global baseline of 11.46%. Encounters with no medication changes have the lowest readmission rates; encounters with many simultaneous medication changes have elevated rates. This validates the claim that active medication management is a marker of clinical instability (Sazdov et al., 2023; Goudjerkan & Jayabalan, 2019).

**`HbA1c_diabetes_interaction`** — encounters where *both* conditions hold (HbA1c tested AND primary diagnosis is diabetes) show a higher readmission rate than encounters where neither holds. This confirms the Strack et al. (2014) finding that the interaction between HbA1c testing and diabetic diagnosis is the strongest interaction term in the readmission model (p < 0.001, Table 5). It is important to note the directionality: this is not saying "testing HbA1c *causes* readmission" — it is saying that the combination of being a diabetic patient whose physician felt the need to test HbA1c is already a marker of elevated risk.

---

## 7. Feature Correlations

Before examining heterogeneity, I need to understand which features move together. If two features are highly correlated, then a Dirichlet-induced shift in one will automatically produce a correlated shift in the other. This means the effective dimensionality of the heterogeneity problem is lower than the raw feature count suggests.

The most important correlation in this dataset is the one I already described: `service_utilization`, `number_inpatient`, `number_outpatient`, and `number_emergency` are all correlated by construction. `service_utilization` is a linear combination of the other three. Their correlations with the target are all positive and in the same direction.

`num_medications` and `medication_count` are related but not identical. `num_medications` counts all medications prescribed regardless of change, while `medication_count` counts only those actively adjusted. They share moderate positive correlation.

`time_in_hospital` has weak positive correlations with `num_lab_procedures` and `num_medications`, which makes intuitive sense: longer stays involve more tests and medication management.

`age` is weakly correlated with most other features, confirming its limited marginal predictive power. It correlates positively with `number_inpatient` (older patients have more prior hospitalisations) and negatively with `num_procedures` (younger patients may undergo more active procedures), but neither strongly.

The key implication for the federated experiments: since `number_inpatient` and `service_utilization` are correlated and are the two most discriminative features, any client whose population shifts on the utilisation axis will shift on *both* simultaneously. The heterogeneity in these two features is not independent.

---

## 8. Dirichlet Partitioning: Simulating Hospital Heterogeneity

### 8.1 What the Dirichlet Distribution Does

In a real hospital federation, each hospital would see a different patient mix depending on its geography, specialisation, and referral network. To simulate this in a controlled way, I use a **Dirichlet distribution** to assign patients to simulated clients.

The Dirichlet distribution over K categories produces a probability vector $(p_1, p_2, ..., p_K)$ that sums to 1. The concentration parameter α controls how spread or concentrated those probabilities are:

- When **α is large** (e.g., α = 5.0), draws are concentrated near the center of the simplex — all $p_k ≈ 1/K$. Every client gets approximately the same proportion of patients. This is the near-iid scenario.
- When **α is small** (e.g., α = 0.1), draws are concentrated near the corners of the simplex — almost all mass goes to one client, with nearly nothing for the others. This produces extreme heterogeneity.

In my implementation, the Dirichlet draw is made **separately for positive and negative patients**, then the two groups are combined per client. This means the label distribution (positive rate) can differ dramatically across clients while maintaining the targeted total size proportions.

### 8.2 The Partition Results

```
α = 0.1
  Client 0:     513 patients    |  99.4% positive
  Client 1:  62,604 patients    |   1.2% positive
  Client 2:     278 patients    |  42.8% positive
  Client 3:   7,633 patients    |  90.9% positive
  Client 4:     490 patients    |  99.8% positive

α = 0.5
  Client 0:   2,887 patients    |  96.4% positive
  Client 1:  14,062 patients    |  12.2% positive
  Client 2:  10,057 patients    |  18.2% positive
  Client 3:  14,605 patients    |   2.0% positive
  Client 4:  29,907 patients    |   7.4% positive

α = 1.0
  Client 0:  32,017 patients    |   3.5% positive
  Client 1:   6,004 patients    |  64.4% positive
  Client 2:  10,097 patients    |   6.4% positive
  Client 3:  12,247 patients    |   3.6% positive
  Client 4:  11,153 patients    |  24.6% positive

α = 5.0
  Client 0:  11,300 patients    |  11.5% positive
  Client 1:  17,128 patients    |   6.5% positive
  Client 2:  16,338 patients    |   9.1% positive
  Client 3:  12,244 patients    |  20.3% positive
  Client 4:  14,508 patients    |  16.9% positive
```

### 8.3 Reading α = 0.1

This partition is clinically absurd. Client 1 receives 62,604 patients — 85% of the entire dataset — with only a 1.2% positive rate. The remaining four clients together have 9,414 patients but collectively see positive rates between 43% and 99.8%. No real hospital system looks like this. In practice:

- **FedAvg will fail here.** When averaging model parameters, each client's update is weighted by its sample size. Client 1 contributes 85% of the total weight to the global model. Its 1.2% positive rate will drive the model toward predicting "not readmitted" almost always. The other four clients, despite having highly informative local data, are mathematically drowned out.

- **FedGen's label prior estimation will fail here.** FedGen estimates the global label prior $\hat{p}(y)$ by collecting label counts from each client and weighting by size. Client 1's 1.2% rate, weighted at 85%, drags the estimated prior toward near-zero. The generator will be told that almost no patients get readmitted, and will almost never produce training examples for the positive class.

I treat α = 0.1 as a **stress test** in the thesis rather than a clinically meaningful scenario. It reveals failure modes rather than realistic performance.

### 8.4 Reading α = 0.5

This is the **hardest realistic scenario**. The split is extreme but not absurd:

- Client 0 (2,887 patients, 96.4% positive): this could represent a specialist readmission unit that only admits high-risk patients.
- Client 3 (14,605 patients, 2.0% positive): this could represent a general wellness or preventive care clinic that rarely sees acute readmission cases.
- Client 4 (29,907 patients, 7.4% positive): the largest client, with a below-average positive rate, will again exert disproportionate influence in FedAvg.

The central challenge here is that clients are seeing fundamentally different patient populations, but they are all supposed to be learning the same prediction task. A client that only sees 2% positive cases will learn that "not readmitted" is the near-certain outcome, and its model will be badly miscalibrated. When these miscalibrated models are averaged in FedAvg, the aggregate inherits the miscalibration weighted by client size.

### 8.5 Reading α = 1.0

At α = 1.0 we might expect moderate heterogeneity, but the results are still striking:

- Client 1 (6,004 patients, 64.4% positive): more than half its patients are readmitted within 30 days — far above the global 11.5%.
- Clients 0, 2, and 3 all have positive rates between 3.5% and 6.4%.
- Client 4 has 24.6% — above global but not extreme.

This is heterogeneous in a structurally different way from α = 0.5. At α = 0.5, the heterogeneity was concentrated in the label distribution with one very large "normal" client dominating. At α = 1.0, the largest client (Client 0, 32k patients) has a near-normal positive rate but the second-largest block of signal sits in Client 1, which is severely enriched for positives. This distinction will become important when we compare the two α values using Wasserstein distances.

### 8.6 Reading α = 5.0

This is the **near-iid baseline**. Positive rates range from 6.5% to 20.3%, compared to the global 11.5%. Client sizes are relatively balanced (11k to 17k). This represents the best-case federated scenario: hospitals are seeing similar patient mixes and a simple FedAvg should perform reasonably well. Any FedGen advantage at this α should be small, and if it is not, something unusual is happening.

---

## 9. Wasserstein Distances: Feature Distribution Shift

### 9.1 What the Wasserstein Distance Measures

The Dirichlet partitions in Section 8 quantified how different the **label distributions** are across clients (what fraction of patients are readmitted). The Wasserstein distance analysis asks a different but related question: **how different are the actual feature value distributions?**

The 1-Wasserstein distance between distributions P and Q is:

$$W_1(P, Q) = \int_{-\infty}^{\infty} |F_P(x) - F_Q(x)| \, dx$$

This is the area between the two cumulative distribution functions — the total accumulated difference across the entire range of the feature. Intuitively: if you think of each distribution as a pile of sand, $W_1$ is the minimum amount of work you would need to do to reshape one pile into the other, where work = mass × distance moved.

This is importantly different from the KS statistic. The KS test looks for the single largest gap between the two CDFs. The Wasserstein distance integrates all the gaps. A feature with a small KS but large W₁ is one that differs moderately everywhere. A feature with a large KS but smaller W₁ is one that differs dramatically at one point but is similar elsewhere.

I compute W₁ between each client's feature distribution and the **global** (whole-dataset) feature distribution. This tells me how far each client has drifted from the population-level distribution for each feature, as a direct consequence of the Dirichlet partition.

### 9.2 Results

```
Feature              α=0.1    α=0.5    α=1.0    α=5.0
age                  0.110    0.025    0.034    0.022
medication_count     0.048    0.018    0.018    0.005
num_medications      0.771    0.294    0.297    0.121
number_emergency     0.147    0.078    0.068    0.030
number_inpatient     0.726    0.282    0.303    0.100
service_utilization  0.967    0.397    0.415    0.141
time_in_hospital     0.336    0.142    0.128    0.036
```

These values are **means across the 5 clients** for each (α, feature) combination. They represent how far an average client has drifted from the global distribution.

### 9.3 Reading the Results Column by Column

**`service_utilization`** is the most shifted feature at every single α level. At α = 0.1, the mean W₁ is 0.967 — the total yearly visit count for an average client is nearly one unit away from the global distribution. This is not a coincidence. `service_utilization` is the most discriminative feature (KS = 0.165) *and* it is causally linked to the label. Clients with more positive cases also have patients with higher prior utilisation, because sicker patients both get readmitted more often *and* have more prior hospital contacts. The label heterogeneity and the feature heterogeneity are entangled through the clinical relationship between utilisation and readmission.

**`medication_count`** is the most stable feature across all α values (W₁ = 0.005 to 0.048). The number of medications adjusted during a single encounter is essentially independent of which patient population a client sees. Medication management decisions are made by the attending physician based on the patient's immediate clinical state, not on population demographics. This feature does not drift with the Dirichlet partition.

**`age`** is also very stable (W₁ = 0.022 to 0.110). The age distribution of patients does not change much across partitions. This is because the Dirichlet split is based on readmission labels, not on age directly, and the KS = 0.035 between classes means age is only weakly correlated with the label. Low correlation → low feature drift from label-based partitioning.

**`num_medications`** drifts substantially (W₁ = 0.771 at α = 0.1, 0.294 at α = 0.5). Patients who get readmitted are on more medications — this is captured by the KS = 0.064 signal. So clients enriched for positive cases will see systematically higher `num_medications` values, creating distribution shift.

### 9.4 The α = 0.5 vs α = 1.0 Anomaly

The most surprising finding in the entire analysis:

```
Mean W₁ across features:
  α = 0.5  →  0.1766
  α = 1.0  →  0.1805
```

Mathematically, α = 0.5 should produce *more* heterogeneous feature distributions than α = 1.0, because smaller α means more extreme Dirichlet draws. But the Wasserstein distances are nearly identical — and if anything, α = 1.0 is marginally *higher*.

This is not a bug in the code or a coincidence. It reveals a fundamental limitation of using α as a proxy for heterogeneity: **α controls the skewness of the label distribution, not the magnitude of the feature distribution shift.** The connection between the two depends on how correlated each feature is with the label, and on the specific random draw from the Dirichlet distribution for this seed.

Looking at `number_inpatient` specifically: 0.282 at α = 0.5 vs 0.303 at α = 1.0. The feature distribution is marginally *more* shifted at α = 1.0. Why? Because at α = 1.0 we happened to draw a partition where Client 1 got 6,004 patients with a 64.4% positive rate — a small, highly enriched group. Its feature distributions are extreme even though its client size is small. At α = 0.5, Client 0 (96.4% positive) only has 2,887 patients, and Client 4 (7.4% positive, 29,907 patients) dominates the mean W₁ calculation by being large and close to normal.

The practical consequence for the thesis is important: **I cannot use α as a reliable single measure of how hard the federated problem is.** The mean W₁ across clients and features is the empirically grounded measure of actual feature heterogeneity, and α is only a noisy proxy for it. This is why the Pareto analysis (AUC vs communication cost) should be indexed against W₁ directly, not against α alone.

---

## 10. Label Shift vs Covariate Shift

```
          JS Divergence    Mean W₁
          (label shift)    (feature shift)
α = 0.1      0.582          0.444
α = 0.5      0.222          0.177
α = 1.0      0.193          0.181
α = 5.0      0.055          0.065
```

These two numbers measure fundamentally different aspects of heterogeneity, and it is important to keep them separate.

**Label shift** (Jensen-Shannon divergence) measures how different each client's class distribution is from the global distribution. JS divergence is the symmetric version of KL divergence — it measures the information-theoretic distance between the client's label prior $p_k(y)$ and the global label prior $p(y)$. A JS of 0.582 at α = 0.1 means the label distribution at an average client is very far from the global distribution. A JS of 0.055 at α = 5.0 means the distributions are nearly identical.

**Covariate shift** (mean W₁) measures how different the feature distributions are, as described in Section 9.

The key observation is that **label shift and covariate shift scale together with α but are not proportional to each other**. At α = 0.5 vs α = 1.0, the JS values diverge more clearly (0.222 vs 0.193) than the W₁ values (0.177 vs 0.181). This means at α = 0.5, the problem is primarily a **label shift problem** — the client class distributions are more skewed than their feature distributions. At α = 1.0, the feature and label shifts are more balanced.

For FedGen specifically, label shift matters because it distorts the generator's label prior estimate $\hat{p}(y)$. Covariate shift matters because it determines how far the client's feature representations are from the global distribution in latent space — which is what the generator's inductive bias is trying to correct.

---

## 11. Distribution of W₁ Across Clients

The mean W₁ values in Section 9 hide an important story: the distribution of W₁ *across clients within a given α*. The mean can be high because all clients drift moderately, or because one or two clients drift dramatically while the others are close to normal. These are very different situations.

At α = 0.1, the W₁ distribution across clients is extremely wide. Client 1 (62,604 patients, 1.2% positive) has W₁ close to zero on most features — it basically *is* the global distribution, because it contains 85% of all patients. But Clients 0, 3, and 4 have enormous W₁ values because their feature distributions are dominated by the positive class. The high mean W₁ at α = 0.1 is driven by a small number of extreme outlier clients, not by uniform drift across all five.

At α = 5.0, the W₁ values are uniformly small across all clients. All five clients are close to the global distribution on all features.

At α = 0.5 and α = 1.0, the picture is intermediate — some clients are close to global, others drift significantly on the utilisation-related features.

This client-level heterogeneity in feature drift is directly relevant to the equity analysis in the thesis. A client with W₁ near zero will produce a reasonable local model even with FedAvg, because its data looks like the global distribution. A client with high W₁ will produce a biased local model, and whether FedGen's generator corrects that bias is a measurable, per-client question.

---

## 12. Effective Minority-Class Sample Size

```
            α = 0.1   α = 0.5   α = 1.0   α = 5.0
Client 0:      510     2,784     1,134     1,301
Client 1:      779     1,715     3,869     1,114
Client 2:      119     1,829       642     1,489
Client 3:    6,937       292       443     2,485
Client 4:      489     2,214     2,746     2,445

Below 200 positive samples: Client 2 at α=0.1 (119 positive patients)
```

This table answers a direct question: **how many patients who actually get readmitted does each client see?** This is the effective training signal for the minority class at each client.

At α = 0.1, Client 2 has only 119 positive patients. With 119 examples, the model cannot learn the diversity of clinical presentations that lead to readmission. It will overfit to the narrow subset of readmission patterns present in those 119 cases. The threshold of 200 positive samples is a rough practical minimum — below this, local minority-class learning is unreliable.

The situation at α = 0.5 is more nuanced. Client 3 has 292 positive patients — above the 200 threshold, but still sparse. Its positive rate is only 2%, meaning that for every readmitted patient it sees, it sees 49 non-readmitted patients. Its local model will be heavily pressured toward the negative class regardless of class weighting. This is the scenario where FedGen's inductive bias from the generator is most valuable: by providing synthetic positive-class representations from the global distribution, it gives Client 3 a proxy for positive cases it has almost never seen.

At α = 5.0, all clients have between 1,114 and 2,485 positive patients — sufficient for reasonable minority-class learning. The federated problem at this α level is manageable without any knowledge distillation.

---

## 13. Label Prior Distortion in FedGen

```
α         Estimated p̂(y=1)    True global rate    Distortion
0.1       0.1235               0.1146              +7.8%
0.5       0.1235               0.1146              +7.8%
1.0       0.1235               0.1146              +7.8%
5.0       0.1235               0.1146              +7.8%
```

FedGen estimates the global label prior $\hat{p}(y)$ by collecting each client's label counts and weighting by sample size. The estimated positive rate across all α values is 0.1235 — a 7.8% overestimate relative to the true global rate of 0.1146.

The fact that the estimated prior is identical across all four α values tells me something about my implementation: the label prior distortion in this case is coming from the way patients are counted, not from the heterogeneity of the Dirichlet split. The estimate is being computed from the patient-level partition (unique patients per client) rather than from the encounter-level counts. Since the same patient can appear multiple times in the encounter dataset (different visits), and the assignment is at the patient level, the encounter-level positive rate will differ slightly from the patient-level positive rate. This is a known property of the dataset: patients who get readmitted tend to have more total encounters than those who do not, so encounter-level positive rate (11.46%) is slightly lower than patient-level positive rate (12.35%).

This 7.8% overestimate is small and unlikely to meaningfully affect FedGen's generator training. However, it is worth documenting: in a real deployment, clients would report encounter-level counts, and if patients with more encounters are systematically different from one-time patients (which they are — they're sicker), the estimated prior would be biased in the same direction seen here.

---

## 14. Summary and Implications for the Federated Experiments

Bringing everything together, the distribution analysis establishes the following facts about the data:

**The discriminative signal is concentrated.** Prior hospitalisation history (`number_inpatient`, `service_utilization`) is more than twice as informative as any other individual feature. The model's predictive performance is largely determined by how well it captures this utilisation axis. Any federated heterogeneity that distorts this feature will be especially damaging to model quality.

**α is not a reliable proxy for heterogeneity.** The Wasserstein analysis shows that α = 0.5 and α = 1.0 produce nearly identical feature distribution shifts despite meaningfully different label distributions. The W₁ measure is the empirically grounded quantity, and the Pareto analysis should be indexed against it.

**α = 0.1 is pathological, not realistic.** The positive rate in individual clients reaches 99.8%, which is clinically impossible in a real hospital. This α level is useful for understanding failure modes but should be presented as a stress test rather than a realistic scenario.

**Label shift and feature shift are coupled but not equivalent.** They both scale with α but capture different aspects of the problem. FedGen's generator addresses feature shift in latent space; the label prior estimation addresses label shift. Both mechanisms are needed.

**Minority-class sample size is the binding constraint.** At α = 0.5, some clients have as few as 292 positive examples. This is the specific condition where FedGen's synthetic positive-class examples provide the most value — and where the equity analysis (per-client AUC) will show the most variation across methods.

---

## References

- Goudjerkan, T. & Jayabalan, M. (2019). Predicting 30-day hospital readmission for diabetes patients using multilayer perceptron. *International Journal of Advanced Computer Science and Applications*, 10(5).
- Jauhari, A. et al. (2021). Hospital readmission prediction: A systematic review. *Journal of Healthcare Engineering*.
- Sazdov, M. et al. (2023). A comprehensive analysis of machine learning approaches for predicting hospital readmission in diabetic patients. *IEEE Access*.
- Strack, B. et al. (2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. *BioMed Research International*. https://doi.org/10.1155/2014/781670
- Zhu, Z., Hong, J., & Zhou, J. (2021). Data-free knowledge distillation for heterogeneous federated learning. *Proceedings of the 38th International Conference on Machine Learning (ICML)*. PMLR 139.
