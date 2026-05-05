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

