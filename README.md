# Breast Cancer Exploratory Data Analysis (EDA)

이 프로젝트는 Scikit-learn에서 제공하는 유방암(Breast Cancer Wisconsin) 데이터셋을 활용하여, 종양의 형태학적 특징(Morphological features)이 악성(Malignant) 및 양성(Benign) 판별에 미치는 영향을 생물학적 관점에서 탐색한 EDA 포트폴리오입니다.

## 1. Dataset Overview
*   **Source:** Scikit-learn 내장 데이터셋 (`load_breast_cancer`)
*   **Instances (총 샘플 수):** 569개 종양 데이터
*   **Features (특징 변수):** 30개의 수치형 변수 (종양의 반경, 질감, 면적 등 세포핵의 디지털 이미지에서 추출된 형태학적 계산값)
*   **Target (분류 클래스):** 악성(Malignant, 212개) vs 양성(Benign, 357개)

---
## 2. Data Preprocessing & Quality Control (QC)

데이터의 무결성을 검증하고, 본격적인 탐색 전 클래스 분포를 확인했습니다.

*   **Missing Values (결측치):** 0개. 추가적인 결측치 대체(Imputation) 파이프라인 불필요.
*   **Class Distribution (클래스 불균형):** 
    *   Benign (양성): 357건 (62.7%)
    *   Malignant (악성): 212건 (37.3%)
    *   **Insight:** 극단적 불균형은 아니나, 의료 도메인의 특성상 모델링 시 False Negative를 최소화하기 위해 Recall(재현율) 지표에 가중치를 두어야 함을 시사함.

### 🩺 생물학적 기초 통계 비교 (Malignant vs Benign)

| Target Label | Mean Radius (평균 반경) | Mean Area (평균 면적) | Biological Insight |
| :--- | :---: | :---: | :--- |
| **Benign (양성)** | 12.15 | 462.79 | 정상적인 세포 분열 주기를 유지하며 뚜렷한 경계를 가짐 |
| **Malignant (악성)** | **17.46** | **978.38** | 통제 불능의 **세포 증식(Proliferation)**으로 인해 세포 크기(Area)가 급격히 팽창함 |

---
## 3. Exploratory Data Analysis (EDA) & Visualizations

세포핵의 형태학적 특징이 악성 종양 판별에 미치는 영향을 시각적으로 탐색했습니다.

### 3.1. 형태학적 특징의 분포 (Distribution of Key Features)
![Violin Plots](./images/violin_plots.png)
*   **분석:** 악성(Malignant) 종양은 양성(Benign)에 비해 `mean radius`와 `mean texture`의 분산이 매우 넓습니다.
*   **생물학적 해석:** 이는 암세포 특유의 **종양 내 이질성(Intratumor Heterogeneity)**과 통제 불능의 비정상적 핵 분열을 시사합니다. 세포핵의 질감(Texture) 편차가 큰 것은 염색질(Chromatin)의 뭉침 현상 등 악성 종양의 병리학적 특징을 반영합니다.

### 3.2. 클래스 분리도 및 다중공선성 (Class Separability & Multicollinearity)
<p align="center">
  <img src="./images/scatter_radius_texture.png" width="45%" />
  <img src="./images/correlation_heatmap.png" width="45%" />
</p>

*   **Scatter Plot:** 크기(`mean radius`)와 질감(`mean texture`) 단 두 개의 변수만으로도 양성과 악성 데이터가 비교적 뚜렷하게 군집화(Clustering)되는 것을 확인했습니다. 이는 선형 분류기(Linear Classifier)로도 준수한 성능을 낼 수 있음을 암시합니다.
*   **Correlation Heatmap:** `radius`, `perimeter`, `area` 간의 상관계수가 0.99 이상으로 나타납니다. 향후 예측 모델링 단계에서는 다중공선성(Multicollinearity)을 방지하고 모델을 경량화하기 위해, 물리적으로 동일한 차원을 설명하는 변수들에 대한 Feature Selection이 필요합니다.

### 3.3. 종양의 형태학적 붕괴와 유의성 (Morphological Irregularity)
![Shape Features](./images/shape_features_boxplot.png)
*   **생물학적 고찰:** 악성 종양은 주변 조직으로의 침윤(Invasion)과 유전적 불안정성으로 인해 세포핵 경계가 심하게 붕괴됩니다. 데이터 분석 결과, 악성 데이터군에서 **오목함(Concavity)** 수치가 급격히 상승하며, 정상적인 **대칭성(Symmetry)**을 상실하는 것을 Boxplot과 T-test(p-value < 0.05)를 통해 통계적으로 교차 검증했습니다.

---
## 4. Machine Learning Scale-up: Feature Engineering

선형 기반 기계학습 모델의 해석력을 높이고 다중공선성(Multicollinearity)을 철저히 해소하기 위해, **VIF(Variance Inflation Factor)** 기반의 Feature Selection을 수행했습니다.

*   **배경:** `mean radius`, `mean perimeter`, `mean area` 등의 변수는 물리적/기하학적으로 강하게 종속되어 있어 모델 가중치(Coefficient)의 심각한 왜곡을 유발합니다.
*   **해결 전략:** 임계값(Threshold)을 10으로 설정하고, VIF가 가장 높은 변수부터 반복적으로 제거하는 Stepwise 알고리즘을 적용했습니다.
*   **최종 선택된 변수 (Selected Features):**
    *   `mean area`: 종양의 제어 불가능한 세포 증식(Proliferation)을 대변.
    *   `mean concavity`: 주변 조직 침윤에 의한 형태학적 붕괴 대변.
    *   `mean symmetry`: 유전적 불안정성에 의한 비대칭 성장 대변.
*   **결과:** 중복되는 기하학적 특성을 덜어내고, 생물학적 독립성을 가진 3개의 핵심 변수만으로 모델을 성공적으로 경량화했습니다.

---

## 5. Interpretable Baseline Modeling: Logistic Regression

블랙박스 모델을 적용하기 전, 추출된 핵심 변수들이 암 발현에 미치는 생물학적 가중치를 수치화하여 증명하기 위해 로지스틱 회귀(Logistic Regression) 베이스라인 모델을 구축했습니다.

### 5.1. 모델링 전략 및 전처리
*   **Target Alignment:** 의료 예측 모델의 표준 논리에 맞추어 악성(Malignant)을 Positive(1), 양성(Benign)을 Negative(0)로 재매핑하여 리스크를 예측하도록 설계했습니다.
*   **Standard Scaling:** 단위가 서로 다른 변수(`area`와 `concavity` 등)가 회귀 계수에 미치는 왜곡을 방지하기 위해 `StandardScaler`를 적용하여 데이터의 분포를 정규화했습니다.
*   **Data Leakage Prevention:** Train set(80%)에만 스케일러를 Fit하여 검증 데이터의 정보 누수를 철저히 차단했습니다.

### 5.2. Coefficient Analysis (생물학적 가중치 해석)
![Feature Importance](./images/logistic_coefficients.png)
*   **해석:** 학습된 회귀 모델의 계수(Coefficient)를 추출한 결과, `mean area`가 약 +3.43으로 가장 강력한 악성 판단 가중치를 가짐을 증명했습니다. 이는 형태의 붕괴(`concavity`, `symmetry`)도 중요하지만, **근본적인 세포 증식(Proliferation)에 의한 부피 팽창이 암세포를 특정하는 가장 지배적인 물리적 지표**임을 수학적으로 뒷받침합니다.

---
## 6. Clinical Metrics Evaluation (임상 지표 검증)

단순히 전체 정확도(Accuracy)를 높이는 것을 넘어, 의료 인공지능의 핵심인 **'위음성(False Negative) 최소화'** 관점에서 모델을 평가했습니다.

### 6.1. Confusion Matrix & ROC-AUC
![Clinical Metrics](./images/clinical_metrics.png)

*   **분석 결과:** 
    *   **Accuracy:** 약 89.5%
    *   **ROC-AUC:** 0.979 (모델의 전반적인 암/정상 분류 성능이 매우 우수함)
    *   **Recall (재현율):** 76.2%
*   **임상적 고찰:** 본 베이스라인 모델은 정밀도(Precision, 94.1%)는 매우 높으나, 재현율(Recall) 측면에서 실제 암 환자의 일부를 정상으로 오진(False Negative)하는 한계를 보였습니다. 생명이 직결된 종양학 데이터에서는 불필요한 추가 검사 비용(False Positive)을 감수하더라도, 병을 놓치는 위음성(False Negative)을 극도로 통제해야 합니다.

이러한 선형 베이스라인의 한계(Recall 최적화의 어려움)를 극복하기 위해, 향후 다중 오믹스 데이터를 결합한 딥러닝 앙상블 모델을 구축하고 Decision Threshold(임계값)를 동적으로 조정하는 연구로 확장할 계획입니다.

### 6.2. Precision-Recall Curve 분석 및 과적합(overfitting) 방어 원칙

의료 데이터의 특성상 위음성(False Negative)을 통제하기 위해 재현율(Recall)을 최우선으로 고려해야 합니다. 이를 시각적으로 확인하기 위해 Precision-Recall Curve를 도출했습니다.

* **분석 결과:** 기본 임계값(Threshold = 0.5)에서 이미 높은 수준의 정밀도와 재현율 방어력을 보여줍니다. (ROC-AUC 0.979)
* **Data Leakage 방어 원칙:** * 현재 테스트 데이터(Test Data)에서 도출된 PR Curve를 바탕으로 임계값을 조정하여 재현율을 억지로 끌어올리는 방식은 **'정답지를 보고 오답 노트를 수정하는 행위(Data Leakage)'**입니다. 
    * 테스트 데이터의 무결성을 완벽히 보존하고 과적합을 방지하기 위해, 본 베이스라인 모델에서는 기본 임계값(0.5)을 유지하여 일반화(Generalization) 성능을 객관적으로 증명하는 데 집중했습니다.

### 6.3. 임상 지표 최적화를 위한 3가지 정석적 대안 (Future Work)

테스트 데이터의 오염 없이 재현율(Recall)을 합법적으로 극대화하기 위해, 향후 다음과 같은 3가지 정석적인 고도화 기법을 파이프라인에 적용할 계획입니다.

1.  **K-Fold 교차 검증을 통한 임계값 탐색 (Threshold Tuning via CV):**
    * 테스트 데이터는 완전히 격리한 상태에서, 학습 데이터(Train Data) 내부를 5~10개의 Fold로 나누어 교차 검증을 수행합니다. 이를 통해 학습 환경 내에서만 최적의 임계값(예: 0.3)을 탐색하고, 최종 확인 단계에서만 테스트 데이터에 단 한 번 적용합니다.
2.  **가중치 조절을 통한 모델 재학습 (Class Weight Adjustment):**
    * 임계값을 물리적으로 조작하는 대신, 알고리즘이 학습하는 단계에서부터 악성(Malignant) 클래스의 오답 페널티를 강하게 부여합니다. (e.g., `LogisticRegression(class_weight={0: 1, 1: 5})`)
3.  **확률 분포 분석 및 보정 (Probability Calibration):**
    * 모델이 출력하는 확률값(Probability)의 분포 자체를 분석하여, 악성 종양에 대한 예측 확률이 0.5 부근에서 불안정하게 진동하는지, 아니면 양극단으로 확실히 분리되는지를 검증하고 신뢰도를 보정합니다.

---


## 7. Model Stabilization & Clinical Optimization 

### 7.1. GridSearchCV 아키텍처 및 결과
* **최적화 목표 (Scoring):** `Accuracy`가 아닌 `Recall`을 타겟으로 탐색을 진행하여 위음성(False Negative) 방어 체계를 구축했습니다.
* **최적 파라미터 도출:** `{'C': 0.1, 'class_weight': {0: 1, 1: 5}, 'penalty': 'l2'}`
* **수학적/임상적 의의:**
    * **강력한 규제 (`C=0.1`):** 소규모 오믹스 데이터 특성상 발생할 수 있는 과적합을 방지하기 위해 강한 L2 규제가 도출되었습니다.
    * **클래스 가중치 (`1:5`):** 악성 종양 오분류에 5배의 수학적 페널티를 부여하여, 테스트 데이터의 임계값 조작(Leakage) 없이 모델의 손실 함수(Loss Function) 자체가 암 진단에 민감해지도록 재설계되었습니다.

### 7.2. 성능 갱신 검증 (Tuned Confusion Matrix)
![Tuned Confusion Matrix](./images/tuned_confusion_matrix.png)
* **핵심 성과:** 하이퍼파라미터 최적화 적용 결과, 고립된 테스트 데이터 환경에서 **암 환자 누락(False Negative) 건수를 기존 10명에서 단 1명으로 획기적으로 감소시켰습니다 (Recall 향상: 76.2% -> 97.6%).**
* 증가한 위양성(False Positive, 11건)은 임상 환경에서 조직검사 등의 2차 스크리닝으로 커버 가능한 영역이므로, 환자의 생존율을 최우선으로 담보하는 성공적인 최적화 모델임을 입증했습니다.

---

## 8. Advanced Ensemble Modeling
로지스틱 회귀가 가지는 선형 분리의 한계를 극복하고, 종양 형태 변수들 간의 복잡한 다차원적 상호작용(Interaction)을 포착하기 위해 비선형 앙상블(Ensemble) 알고리즘을 도입했습니다.

### 8.1. 집단 지성 및 부스팅(Boosting) 아키텍처 도입
* **Random Forest:** 배깅(Bagging) 방식을 통해 다수의 의사결정 나무를 생성하여 예측의 분산(Variance)을 낮추고 노이즈에 대한 강건성(Robustness)을 확보했습니다. (`class_weight={0: 1, 1: 5}` 적용)
* **XGBoost:** 부스팅(Boosting) 기법을 활용하여 이전 모델이 놓친 위음성(False Negative) 오차를 순차적으로 보정하는 정밀한 결정 경계를 구축했습니다. (`scale_pos_weight=5` 적용)

### 8.2. 앙상블 모델 성능 평가
![Ensemble Comparison](./images/ensemble_comparison.png)
* **분석 결과:** 두 앙상블 모델 모두 약 0.98 이상의 압도적인 ROC-AUC 수치를 기록하며 뛰어난 클래스 분리 능력을 증명했습니다.
* **임상적 고찰:** 변수가 3개로 압축된 현재 데이터셋에서는 선형 로지스틱 회귀(Tuned)도 훌륭한 재현율을 보였으나, 앙상블 모델은 선형 모델이 포착하지 못하는 **"면적(Area) 팽창과 오목함(Concavity)의 동시다발적 붕괴"**와 같은 비선형적 상호작용을 수학적으로 학습할 수 있음을 확인했습니다. 이는 향후 대용량 다중 오믹스 데이터(Multi-omics)로 스케일업할 때 앙상블 모델이 핵심 파이프라인이 될 것임을 시사합니다.


---


## 9. Explainable AI (XAI) & Biological Interpretation 

블랙박스(Black-box) 앙상블 모델의 임상적 신뢰성을 확보하기 위해, 게임 이론 기반의 **SHAP(SHapley Additive exPlanations)** 분석을 수행하여 형태학적 변수들의 독립적인 생물학적 기여도를 해부했습니다.

### 9.1. SHAP Summary Plot 분석
![SHAP Summary Plot](./images/shap_summary_plot.png)

* **해석 방법:** X축은 해당 변수가 '악성(Malignant) 종양으로 판별하는 데 기여한 영향력'을 의미하며, 점의 색상(Blue to Red)은 실제 변수의 Feature Value(낮음~높음)를 나타냅니다.
* **생물학적 인사이트 도출:**
    1. **면적 팽창의 지배성 (`mean area`):** 형태학적 특성 중 기여도가 가장 높으며, 붉은색 점(수치가 큼)이 우측(악성 판정)에 길게 분포합니다. 이는 제어 불가능한 세포 증식(Proliferation)이 앙상블 트리 노드 분류의 최상단 기준점임을 시사합니다.
    2. **침윤성 경계 붕괴 (`mean concavity`):** 크기(Area)와 대등한 수준의 Gini 중요도(약 0.41)를 가집니다. 종양이 커지더라도 경계가 매끄러우면 악성 확률이 낮아지나, 세포 윤곽에 굴곡(Concavity)이 깊어지는 순간 악성 예측 가중치가 폭발적으로 상승함을 XAI 시각화를 통해 증명했습니다.


---

## 10. Final Conclusion & Wrap-up
본 프로젝트는 단순한 EDA를 넘어 다중공선성 제어, Data Leakage 방어 기반의 하이퍼파라미터 튜닝, 비선형 앙상블 모델링, 그리고 SHAP XAI 분석까지 아우르는 Full-cycle Data Science 파이프라인을 독립된 Linux 환경에서 성공적으로 구축했습니다. 향후 이 방법론을 확장하여 'AI 기반 인체모사 비임상 시험 및 표적 신약 가상 스크리닝 파이프라인'을 고도화할 역량이 있음을 본 포트폴리오를 통해 입증합니다.


## 11. Limitations & Future Work

*   **한계점:** 본 분석은 기초적인 형태학적 변수에만 의존한 EDA로, 실제 임상에서 쓰이는 전사체(Transcriptome) 데이터 등 분자생물학적 변수가 누락되어 있습니다.
*   **Next Step (모델링 고도화):** 현재 구축된 독립적인 리눅스(Fedora) Conda 환경과 16GB RAM 이상의 가용 메모리 인프라를 적극 활용하여, 향후 대용량 다중 오믹스(Multi-omics) 데이터를 병렬 처리하고 딥러닝 앙상블 모델(e.g., 표적 신약 가상 스크리닝 파이프라인)을 구축하는 방향으로 연구를 확장할 계획입니다.

---
*Maintained by 3rd-year Undergraduate Student, Soongsil University