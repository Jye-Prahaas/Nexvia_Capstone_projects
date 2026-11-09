# Machine Learning & AI Projects — Combined README

This repository/document contains documentation for two projects:

1. **AIML-01 — Financial Fraud Detection Platform**
2. **AIML-02 — AI Medical Imaging**

---

# AIML-01 — Financial Fraud Detection Platform

## Financial Fraud Detection Platform

This project is an end-to-end Machine Learning-based Financial Fraud Detection Platform designed to identify potentially fraudulent financial transactions. It combines data processing, feature engineering, XGBoost classification, real-time transaction streaming, REST API development, data drift monitoring, and an interactive analyst dashboard.

The platform uses the PaySim synthetic mobile-money transaction dataset to train and evaluate a fraud detection model. It also explores historical account features using Feast and provides a Kafka-based transaction streaming pipeline for transaction scoring.

The project demonstrates how machine learning models can be integrated into an application with prediction services, monitoring, and analyst feedback.

> **Note:** This project uses synthetic PaySim data and is intended for educational and portfolio purposes. It is not a production-ready financial fraud detection system.

---

### Table of Contents

- [Project Objectives](#project-objectives)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Running the API](#running-the-api)
  - [Running Kafka Streaming](#running-kafka-streaming)
  - [Running the Dashboard](#running-the-dashboard)
- [Implementation Details](#implementation-details)
  - [Machine Learning Model](#machine-learning-model)
  - [Feature Engineering and Feast](#feature-engineering-and-feast)
  - [Real-Time Transaction Processing](#real-time-transaction-processing)
  - [Fraud Prediction API](#fraud-prediction-api)
  - [Data Drift Monitoring](#data-drift-monitoring)
  - [Analyst Dashboard](#analyst-dashboard)
  - [Docker and Kubernetes Deployment](#docker-and-kubernetes-deployment)
  - [GraphSAGE Experiment](#graphsage-experiment)
- [Model Evaluation](#model-evaluation)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

---

### Project Objectives

1. Develop a machine learning model to identify potentially fraudulent financial transactions.
2. Handle severe class imbalance in financial transaction datasets.
3. Perform feature engineering to improve fraud detection.
4. Evaluate model performance using precision, recall, F1-score, and PR-AUC.
5. Develop a REST API for transaction prediction.
6. Implement Kafka-based transaction streaming and processing.
7. Explore historical account features using Feast.
8. Monitor incoming transaction data for distribution drift.
9. Build an interactive dashboard for fraud analytics and case management.
10. Containerize and deploy the API using Docker and Kubernetes.

---

### Key Features

- **Fraud Classification:** XGBoost-based machine learning model.
- **Imbalanced Data Handling:** Class weighting using `scale_pos_weight`.
- **Feature Engineering:** Transaction amount, balance-related and transaction-type features.
- **Feature Store:** Feast feature definitions and historical feature preparation.
- **Experiment Tracking:** MLflow.
- **REST API:** FastAPI-based transaction prediction service.
- **Real-Time Streaming:** Kafka producer and consumer.
- **Persistence:** SQLite and Parquet monitoring storage.
- **Drift Monitoring:** Evidently and Kolmogorov-Smirnov statistical testing.
- **Interactive Dashboard:** Streamlit and Plotly.
- **Case Management:** Fraud alerts, status updates and analyst feedback.
- **Containerization:** Docker.
- **Orchestration:** Kubernetes and Minikube.

---

### System Architecture

The platform is organized into offline model development, online transaction scoring, monitoring, and deployment components.


<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/80a57e65-ccc5-4e9a-bef1-dfc30879d39f" />



#### Architecture Workflow

1. **Data Source:** PaySim transaction data is used as the input dataset.
2. **Data Processing:** Transactions are cleaned and transformed into machine learning features.
3. **Model Training:** XGBoost is trained and evaluated, with MLflow used for experiment tracking.
4. **Prediction Service:** The trained model is loaded into the FastAPI application.
5. **Transaction Streaming:** Kafka producer publishes transactions to a Kafka topic. The consumer retrieves transactions and sends them to the prediction API.
6. **Storage:** Consumer state and monitoring records are maintained using SQLite and Parquet.
7. **Drift Monitoring:** Evidently and statistical tests compare transaction feature distributions and generate drift reports.
8. **Dashboard:** Streamlit displays prediction results, transaction analytics, fraud cases and monitoring information.
9. **Deployment:** Docker packages the API, while Kubernetes manages its Deployment and Service.

> **Implementation note:** The current Kubernetes configuration deploys the FastAPI application only. Kafka, the consumer and the Streamlit dashboard run separately. Live account-history enrichment in the tested Kafka pipeline remains an identified limitation.

---

### Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | XGBoost, Scikit-learn |
| Feature Store | Feast |
| Experiment Tracking | MLflow |
| API Development | FastAPI, Uvicorn |
| Streaming | Apache Kafka, kafka-python |
| State Persistence | SQLite |
| Monitoring Storage | Parquet |
| Drift Monitoring | Evidently, SciPy |
| Dashboard | Streamlit, Plotly |
| Graph Experiment | PyTorch, PyTorch Geometric |
| Containerization | Docker |
| Orchestration | Kubernetes, Minikube |

---

### Dataset

The project uses the PaySim synthetic mobile-money transaction dataset.

The dataset contains approximately 6.36 million transactions with transaction details, account balances, transaction types and fraud labels.

#### Dataset Summary

| Attribute | Value |
|---|---:|
| Total Transactions | 6,362,620 |
| Legitimate Transactions | 6,354,407 |
| Fraudulent Transactions | 8,213 |
| Fraud Percentage | 0.1291% |
| Dataset Columns | 11 |

#### Transaction Types

- `CASH_IN`
- `CASH_OUT`
- `TRANSFER`
- `PAYMENT`
- `DEBIT`

Fraudulent transactions in this dataset occur in the `TRANSFER` and `CASH_OUT` categories.

#### Dataset Setup

Download the PaySim dataset and place the transaction CSV file at:

```text
data/raw/Transactions.csv
```

The raw dataset is not included in the repository.

---

### Project Structure

```text
Financial_Fraud_Classifier/
│
├── data/
│   ├── raw/
│   │   └── Transactions.csv
│   │
│   ├── processed/
│   │   ├── features.parquet
│   │   └── labels.parquet
│   │
│   └── monitoring/
│       ├── kafka_transactions.parquet
│       └── kafka_consumer_state.db
│
├── feature_store/
│   └── fraud_features/
│       └── feature_repo/
│           ├── feature_store.yaml
│           ├── feature_definitions.py
│           ├── prepare_feast_data.py
│           ├── build_account_features.py
│           ├── feast_history.py
│           └── data/
│
├── models/
│   ├── xgboost_fraud_model.joblib
│   └── graphsage_fraud_model.pt
│
├── reports/
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   ├── feature_ablation_results.csv
│   ├── fraud_by_transaction_type.csv
│   ├── fraud_by_balance_condition.csv
│   ├── misclassified_test_transactions.csv
│   ├── api_consistency_results.csv
│   ├── data_drift_report.html
│   ├── retraining_trigger.json
│   └── analyst_cases.json
│
├── src/
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   ├── fraud_patterns.py
│   ├── train_model.py
│   ├── feature_comparison.py
│   ├── feature_importance.py
│   ├── Api.py
│   ├── kafka_producer.py
│   ├── kafka_consumer.py
│   ├── drift_monitor.py
│   ├── dashboard.py
│   ├── prepare_graph_data.py
│   └── graph_fraud_model.py
│
├── k8s/
│   └── api.yaml
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

Generated datasets, monitoring databases and local analyst case records should not be committed unless required.

---

### Getting Started

#### Prerequisites

- Python 3.12 or later.
- Git.
- Docker Desktop.
- Kubernetes environment such as Minikube.
- PaySim dataset.

#### Installation

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/Financial_Fraud_Classifier.git

cd Financial_Fraud_Classifier
```

**2. Create a virtual environment**

Windows PowerShell:

```powershell
python -m venv .venv

.venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

If development dependencies are required:

```bash
pip install -r requirements-dev.txt
```

---

### Data Preparation

Place the PaySim CSV file inside:

```text
data/raw/Transactions.csv
```

Run data preparation:

```bash
python -m src.data_preparation
```

Run feature engineering:

```bash
python -m src.feature_engineering
```

The processed datasets are generated under:

```text
data/processed/
```

---

### Model Training

Run the training script from the project root:

```bash
python -m src.train_model
```

The trained XGBoost model is saved at:

```text
models/xgboost_fraud_model.joblib
```

The model uses class imbalance weighting through `scale_pos_weight`.

#### Model Configuration

- Algorithm: XGBoost.
- Input features: 17.
- Imbalance handling: `scale_pos_weight`.
- Decision threshold: 0.5.
- Evaluation: Chronological train/test split.
- Experiment tracking: MLflow.

To launch the MLflow interface:

```bash
mlflow ui
```

---

### Running the Application

#### Running the API

Start the FastAPI service from the project root:

```bash
python -m uvicorn src.Api:app --reload
```

API URL:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

#### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Checks API health and model availability. |
| `/predict` | POST | Predicts whether a transaction is potentially fraudulent. |

#### Health Check

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### Prediction Request Example

Use the exact input schema supported by the running API:

```json
{
  "step": 600,
  "transaction_type": "CASH_OUT",
  "amount": 100000,
  "oldbalanceOrg": 100000,
  "oldbalanceDest": 0,
  "previous_transaction_count": 0,
  "previous_average_amount": 0
}
```

The API response includes:

- Predicted transaction class.
- Fraud probability.
- Decision threshold.
- Model version.

---

### Running Kafka Streaming

The project uses Apache Kafka to replay transaction records and process them through the FastAPI prediction service.

#### 1. Start Kafka

From the project root:

```bash
docker compose up -d
```

Kafka broker:

```text
localhost:9092
```

Kafka topic:

```text
fraud-transactions
```

#### 2. Start the API

In one terminal:

```bash
python -m uvicorn src.Api:app --reload
```

#### 3. Start the Kafka Consumer

In a second terminal:

```bash
python -m src.kafka_consumer
```

#### 4. Start the Kafka Producer

In a third terminal:

```bash
python -m src.kafka_producer --limit 5000 --delay 0.01
```

The producer publishes transactions, and the consumer processes them by sending prediction requests to the FastAPI service.

#### Monitoring Output

The consumer stores monitoring records in:

```text
data/monitoring/kafka_transactions.parquet
```

SQLite state is maintained in:

```text
data/monitoring/kafka_consumer_state.db
```

**Known limitation:** In the tested streaming run, 100,435 monitoring records were saved, but both account-history features remained zero. Transaction streaming and monitoring persistence work, but live account-history enrichment still needs investigation.

---

### Running the Dashboard

Start the Streamlit dashboard:

```bash
streamlit run src/dashboard.py
```

The dashboard provides an interface for manual transaction scoring, fraud analytics, case management and drift monitoring.

---

### Implementation Details

#### Machine Learning Model

The fraud detection model uses XGBoost to classify transactions as potentially fraudulent or legitimate.

The dataset has severe class imbalance, with fraudulent transactions accounting for approximately 0.13% of the total data.

To address this, the model uses:

- Class weighting through `scale_pos_weight`.
- Chronological train/test splitting.
- Feature engineering.
- Precision, recall and F1-score evaluation.
- Precision-Recall AUC evaluation.

MLflow is used for experiment tracking.

#### Feature Engineering and Feast

The project includes transaction-level feature engineering and a Feast feature repository.

The feature engineering process includes:

- Transaction amount transformations.
- Account balance-related features.
- Transaction-type encoding.
- Historical account feature preparation.

Two historical features are explored:

- `previous_transaction_count`
- `previous_average_amount`

The reference feature-generation process calculates historical values using earlier transaction steps while excluding transactions from the current step.

The reference dataset contains 9,266 transactions with non-zero historical account information.

**Implementation note:** Feast is currently used for feature definitions and historical retrieval experiments. The training and API pipelines also have their own data preparation and feature calculation paths, rather than being fully integrated through the Feast online store.

#### Real-Time Transaction Processing

Apache Kafka is used to replay transaction records through a producer-consumer architecture.

The processing workflow is:

1. Kafka producer publishes transaction records.
2. Transactions are received through the `fraud-transactions` topic.
3. Kafka consumer reads the transactions.
4. Consumer sends transactions to the FastAPI prediction service.
5. Prediction and monitoring information is persisted.
6. SQLite maintains consumer state.
7. Parquet stores monitoring records for further analysis.

The current tested Kafka output contains zero values for both account-history features. Therefore, live historical feature enrichment has not yet been verified.

#### Fraud Prediction API

FastAPI provides a REST-based interface for transaction scoring.

The API provides:

- `/health` for health checking.
- `/predict` for transaction classification.
- Fraud probability output.
- Model version information.
- Input validation.

The trained XGBoost model is loaded by the API to generate predictions.

#### Data Drift Monitoring

The project uses Evidently and the Kolmogorov-Smirnov statistical test to monitor changes in transaction feature distributions.

The monitoring process compares reference transaction data with current streaming data.

Generated outputs:

```text
reports/data_drift_report.html
reports/retraining_trigger.json
```

#### Latest Recorded Drift Result

| Metric | Result |
|---|---:|
| Drifted Feature Share | 70.6% |
| Drifted Features | 12 of 17 |
| Configured Trigger Threshold | 30% |
| Retraining Signal | Required |

The comparison used 20,000 reference records and 3,887 replayed Kafka records.

**Important:** The reference and current datasets represent different transaction windows. Some detected drift may therefore reflect temporal differences rather than genuine model degradation.

The drift monitor generates a retraining signal only. It does not automatically train or deploy a new model.

#### Analyst Dashboard

The Streamlit dashboard provides a user interface for interacting with the fraud detection platform.

Main features include:

**1. Transaction Scoring**
- Manually submit transaction information.
- Obtain predictions through FastAPI.
- View fraud probability.

**2. Fraud Analytics**
- Prediction distribution.
- Fraud probability by case.
- Analyst feedback distribution.

These fraud analytics are based on manually submitted dashboard cases, not the entire Kafka monitoring dataset.

**3. Transaction Analytics**
- Transaction amount distribution.
- Transaction volume by step.

**4. Drift Monitoring**
- Drifted feature share.
- Dataset drift status.
- Retraining trigger status.
- Access to the generated HTML drift report.

**5. Fraud Alerts and Case Management**
- Browse fraud alerts.
- Select individual cases.
- Update case status.
- Record analyst feedback:
  - Confirmed Fraud.
  - False Positive.
  - Pending.

Dashboard case records are stored locally in:

```text
reports/analyst_cases.json
```

Analyst feedback is recorded for review but is not automatically used to retrain the model.

#### Docker and Kubernetes Deployment

The FastAPI application is containerized using Docker.

Build the image:

```bash
docker build -t financial-fraud-classifier:latest .
```

Run the container:

```bash
docker run -p 8000:8000 financial-fraud-classifier:latest
```

The Kubernetes configuration is located at:

```text
k8s/api.yaml
```

For Minikube:

```bash
minikube image load financial-fraud-classifier:latest

kubectl apply -f k8s/api.yaml
```

Check the deployment:

```bash
kubectl get deployments

kubectl get services

kubectl get pods
```

Forward the API service port:

```bash
kubectl port-forward service/fraud-api 8000:8000
```

The verified Kubernetes configuration uses a single API replica. Kafka, the consumer and the Streamlit dashboard are not deployed as part of this Kubernetes configuration.

#### GraphSAGE Experiment

The project also explores graph-based fraud detection using PyTorch Geometric.

The experiment represents accounts as graph nodes and transactions as edges.

Relevant scripts:

```text
src/prepare_graph_data.py

src/graph_fraud_model.py
```

The experimental model is stored at:

```text
models/graphsage_fraud_model.pt
```

GraphSAGE is an experimental model and is not integrated into the FastAPI prediction pipeline.

---

### Model Evaluation

The XGBoost model was evaluated using a chronological held-out test dataset.

#### Test Dataset

| Attribute | Value |
|---|---:|
| Test Transactions | 123,580 |
| Fraudulent Transactions | 1,654 |
| Legitimate Transactions | 121,926 |

#### XGBoost Performance

| Metric | Result |
|---|---:|
| Precision | 94.57% |
| Recall | 99.94% |
| F1-score | 97.18% |
| PR-AUC | 0.9934 |

#### Confusion Matrix

|  | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 121,831 | 95 |
| Actual Fraud | 1 | 1,653 |

The model detected 1,653 of the 1,654 fraudulent transactions in the held-out test dataset.

PR-AUC is particularly useful for this project because fraudulent transactions represent a very small percentage of the dataset.

These results are specific to the synthetic PaySim dataset and should not be interpreted as expected performance on real financial transactions.

#### Feature Ablation Analysis

| Experiment | Features | Precision | Recall | F1-score | PR-AUC |
|---|---:|---:|---:|---:|---:|
| All Features | 15 | 0.946 | 0.999 | 0.972 | 0.993 |
| Without Balance Flag | 14 | 0.916 | 0.999 | 0.956 | 0.991 |
| Without Transaction Type | 8 | 0.919 | 0.941 | 0.930 | 0.980 |

The ablation experiment was conducted using an earlier 15-feature model configuration, before the final account-history feature set.

Feature importance reports are available at:

```text
reports/feature_importance.csv

reports/feature_importance.png
```

---

### Limitations

1. **Synthetic Dataset:** PaySim does not fully represent real-world financial transaction behavior.
2. **Class Imbalance:** Fraudulent transactions account for only around 0.13% of the dataset.
3. **Account History:** The reference feature-generation process produces historical values, but the tested Kafka consumer output currently contains zero values for both history features.
4. **Feature Integration:** Feast is not yet the single shared feature pipeline for model training and online API serving.
5. **Drift Comparison:** Reference and current monitoring windows may differ, potentially affecting drift results.
6. **Manual Feedback:** Analyst feedback is recorded but is not yet used for automated model retraining.
7. **Local Persistence:** Dashboard cases and consumer state use local JSON and SQLite storage.
8. **Deployment Scope:** Kubernetes currently deploys the API only, using a single replica.
9. **Graph Model:** GraphSAGE remains an experiment rather than part of the live prediction service.
10. **Production Readiness:** Authentication, authorization, distributed persistence, automated testing, CI/CD and production monitoring require further work.

---

### Future Improvements

- Resolve the Kafka account-history feature calculation issue.
- Integrate Feast more consistently across training and serving.
- Improve drift monitoring with matching reference and current time windows.
- Tune the classification threshold using fraud detection costs.
- Incorporate analyst feedback into future training datasets.
- Implement automated retraining and model promotion.
- Add API authentication and stronger security controls.
- Introduce automated testing and CI/CD.
- Expand Kubernetes deployment to include streaming and monitoring components.
- Investigate graph-based fraud detection and model integration.

---

---

### Author

**Jye Prahaas**

Financial Fraud Detection Platform — Machine Learning and Deployment Project.

GitHub:https://github.com/Jye-Prahaas

LinkedIn:www.linkedin.com/in/jye-prahaas-pentlavelli

---

# AIML-02 — AI Medical Imaging

## AI Medical Imaging

An AI-assisted medical imaging research prototype for chest X-ray
classification, CT-volume segmentation, and visual model explainability.
The application combines a DenseNet121 classifier, a MONAI 3D U-Net,
Grad-CAM visualizations, a FastAPI backend, a React frontend,
Docker-based local deployment, audit logging, and Orthanc integration.

> **Research-use notice:** This project is an educational/research
> prototype. It is not a medical device and must not be used to
> diagnose, treat, or make clinical decisions. Model outputs require
> review by qualified medical professionals.

------------------------------------------------------------------------

### Table of Contents

-   [Project Overview](#project-overview)
-   [Key Features](#key-features)
-   [System Architecture](#system-architecture)
-   [Technology Stack](#technology-stack)
-   [AI Models](#ai-models)
-   [Dataset and Preprocessing](#dataset-and-preprocessing)
-   [Model Evaluation](#model-evaluation)
-   [Project Structure](#project-structure)
-   [Prerequisites](#prerequisites)
-   [Run Locally with Docker](#run-locally-with-docker)
-   [Run the Backend Without Docker](#run-the-backend-without-docker)
-   [API Documentation](#api-documentation)
-   [Frontend](#frontend)
-   [Orthanc Integration](#orthanc-integration)
-   [Audit Logging](#audit-logging)
-   [Configuration](#configuration)
-   [Limitations](#limitations)
-   [Future Improvements](#future-improvements)
-   [Responsible Use](#responsible-use)

------------------------------------------------------------------------

### Project Overview

AI Medical Imaging provides a unified interface for two separate
medical-imaging workflows:

1.  **Chest X-ray classification:** DenseNet121 predicts the
    classification result and provides a Grad-CAM visualization to
    highlight image regions that influenced the model.
2.  **3D CT analysis:** A MONAI 3D U-Net processes NIfTI CT volumes or
    DICOM series converted to NIfTI, produces a 3D segmentation mask,
    and provides slice-level CT and Grad-CAM visualizations.

The backend exposes these capabilities through FastAPI. The frontend
communicates with the API to upload scans, browse CT slices, inspect
segmentation and heatmap overlays, and access review/audit
functionality.

### Key Features

-   **Chest X-ray classification** using a trained DenseNet121 model.
-   **Chest X-ray Grad-CAM** visualization returned as a PNG overlay.
-   **3D CT segmentation** using a MONAI 3D U-Net.
-   **NIfTI upload support** for `.nii` and `.nii.gz` volumes.
-   **DICOM series upload** through a ZIP archive containing DICOM
    slices.
-   **CT slice viewer** with slice navigation.
-   **Segmentation overlay** to display the predicted mask on CT slices.
-   **3D CT Grad-CAM** heatmap endpoint for slice-level visualization.
-   **Reviewer decisions** with approved/rejected status and comments.
-   **Audit log** and audit-integrity verification endpoint.
-   **Orthanc integration** with background result monitoring.
-   **Docker Compose deployment** for the backend and frontend.

### System Architecture

``` mermaid
flowchart TD
    U[User] --> F[React Frontend]
    F --> API[FastAPI Backend]

    API --> C[DenseNet121 Classifier]
    API --> S[MONAI 3D U-Net]
    API --> D[DICOM / NIfTI Processing]

    C --> X[Classification Result]
    C --> XG[Chest X-ray Grad-CAM]

    S --> M[3D Segmentation Mask]
    S --> TG[3D CT Grad-CAM]

    API --> V[CT Slice and Overlay Endpoints]
    API --> A[Audit Logging]
    API --> O[Orthanc Integration]

    X --> F
    XG --> F
    M --> F
    TG --> F
    V --> F
    A --> F
    O --> F
```

#### CT processing flow

``` mermaid
flowchart LR
    I[Upload NIfTI or DICOM ZIP] --> P[Volume Preprocessing]
    P --> N[3D U-Net Sliding-Window Inference]
    N --> R[Segmentation Probabilities]
    R --> T[Predicted Mask]
    N --> G[3D Grad-CAM]
    T --> V[Slice and Mask Endpoints]
    G --> H[Grad-CAM Slice Endpoint]
    V --> UI[Frontend CT Viewer]
    H --> UI
```

#### Request lifecycle

1.  The frontend sends a scan to the FastAPI backend.
2.  The backend validates the upload and prepares a temporary input
    file.
3.  The appropriate model performs inference.
4.  CT image, mask, and Grad-CAM arrays are stored in the backend's
    in-memory scan cache.
5.  The `/predict` response contains JSON-safe metadata and a scan ID;
    image arrays are not serialized into JSON.
6.  The frontend uses the scan ID to request CT slices, mask overlays,
    and Grad-CAM overlays.

The scan cache is in memory and is limited to the latest three scans by
default. Restarting the backend clears this cache.

### Technology Stack

  Area                     Technology
  ------------------------ -----------------------------------
  Backend API              FastAPI
  API server               Uvicorn
  Chest X-ray model        PyTorch, DenseNet121
  3D segmentation          PyTorch, MONAI 3D U-Net
  Image processing         OpenCV, Pillow, NumPy
  Medical image handling   NiBabel, DICOM processing service
  Explainability           Grad-CAM
  Frontend                 React
  Containerization         Docker, Docker Compose
  PACS integration         Orthanc
  Audit records            Project audit service

### AI Models

#### 1. Chest X-ray classification --- DenseNet121

The classification workflow uses a DenseNet121 model trained for binary
chest X-ray classification.

-   Input size: 224 × 224.
-   Loss: `BCEWithLogitsLoss` with positive-class weighting.
-   Optimizer: Adam.
-   Learning rate used during training: `0.001`.
-   Training used mixed precision where supported.
-   A patient-level split was used to reduce patient overlap between
    training, validation, and test sets.

The classification endpoint returns a prediction result as JSON. The
separate Grad-CAM endpoint returns a PNG visualization with prediction
and probability in response headers.

#### 2. 3D CT segmentation --- MONAI 3D U-Net

The CT segmentation workflow uses a 3D U-Net implemented with MONAI.

Model configuration:

  Parameter              Value
  ---------------------- --------------------------
  Spatial dimensions     3
  Input channels         1
  Output channels        2
  Channels               `(16, 32, 64, 128, 256)`
  Strides                `(2, 2, 2, 2)`
  Residual units         2
  Normalization          Instance
  Dropout                0.1
  Inference patch size   96 × 96 × 96

Preprocessing configuration used during training/inference:

-   Target spacing: `(1.5, 1.5, 2.0)`.
-   CT intensity range: `-1000` to `400` HU.
-   Foreground cropping with margin 8.
-   Spatial padding to `96 × 96 × 96`.
-   Sliding-window inference for volume prediction.

The trained checkpoint used by the backend is
`models/best_3d_unet_v3.pth`.

#### 3. Explainability

The project provides two distinct explainability workflows:

-   **Chest X-ray Grad-CAM:** generated by the DenseNet121
    classification workflow.
-   **3D CT Grad-CAM:** generated from the 3D U-Net workflow and served
    per CT slice.

Grad-CAM is a model-interpretability visualization, not proof that a
highlighted region is medically meaningful or that a prediction is
correct.

### Dataset and Preprocessing

#### Chest X-ray dataset

The classification model was trained using a preprocessed RSNA Pneumonia
dataset distribution.

Reported dataset summary:

-   26,684 image records.
-   26,684 corresponding mask records.
-   Metadata contained 30,227 rows.
-   Binary target distribution: 20,672 negative and 6,012 positive image
    records.

Patient-level split:

  Split          Records
  ------------ ---------
  Training        18,678
  Validation       4,003
  Test             4,003

The training split contained 4,208 positive records; the validation and
test splits each contained 902 positive records.

#### CT dataset

The 3D segmentation model was trained using Medical Segmentation
Decathlon Task06_Lung data.

CT preprocessing includes resampling, intensity scaling, foreground
cropping, and spatial padding. The inference service applies the
corresponding preprocessing pipeline before sliding-window prediction.

> Dataset access and redistribution are governed by the original dataset
> providers' terms and licenses. Do not commit patient data or
> restricted medical images to this repository.

### Model Evaluation

#### DenseNet121 classification

Recorded test metrics after the five-epoch training run:

  Metric        Result
  ----------- --------
  Accuracy      0.7217
  Precision     0.4377
  Recall        0.8259
  F1-score      0.5722
  ROC-AUC       0.8345

A threshold analysis identified approximately `0.6123` as a threshold
giving precision around `0.500`, recall around `0.701`, and F1 around
`0.584` on the evaluated data.

These are experiment results from the specified dataset split, not
estimates of clinical performance.

#### 3D U-Net segmentation

The best sampled validation Dice recorded during training was `0.3982`
at epoch 250.

A separate full-volume validation experiment across 13 scans reported:

    Probability threshold   Mean Dice
  ----------------------- -----------
                      0.3      0.3891
                      0.4      0.4027
                      0.5      0.4141
                      0.6      0.4247
                      0.7      0.4350

These results are experimental and depend on preprocessing,
thresholding, and validation setup. They should not be interpreted as
clinical validation.

### Project Structure

``` text
Ai_Medical_Imaging/
│
├── app/
│   └──                  # Frontend application
│
├── configs/
│   └──                  # Configuration files
│
├── data/
│   ├── docker_runtime/
│   └──                  # Runtime data; keep datasets out of Git
│
├── models/
│   ├── classifier.py
│   ├── segmentation3d.py
│   ├── densenet121.pth
│   └── best_3d_unet_v3.pth
│
├── notebooks/
│   └──                  # Training and experimentation notebooks
│
├── reports/
│   └──                  # Evaluation and experiment reports
│
├── src/
│   ├── api/
│   │   └── main.py
│   │
│   └── services/
│       ├── classification_service.py
│       ├── Grad_cam.py
│       ├── segmentation3d_service.py
│       ├── dicom_service.py
│       ├── orthanc_service.py
│       ├── orthanc_worker.py
│       ├── upload_to_orthanc.py
│       └── audit.py
│
├── tests/
│
├── requirements.txt
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── .dockerignore
└── README.md
```

The tree is a high-level project map. Exact files may vary as the
project evolves. Do not commit model checkpoints, large datasets,
generated medical images, or runtime databases unless there is a
deliberate, appropriate reason.

### Prerequisites

For local Docker deployment:

-   Windows, macOS, or Linux.
-   Docker Desktop or Docker Engine with Docker Compose.
-   Git (if cloning the repository).
-   Model checkpoint files placed in the expected `models/` directory.

The backend Dockerfile uses Python 3.12 and installs CPU PyTorch wheels.
The currently verified local API health response reported `device: cpu`.

### Run Locally with Docker

Run the commands from the project root directory.

#### 1. Build the containers

``` bash
docker compose build
```

To build only the backend:

``` bash
docker compose build backend
```

To build only the frontend:

``` bash
docker compose build frontend
```

#### 2. Start the application

``` bash
docker compose up -d
```

#### 3. Check running containers

``` bash
docker compose ps
```

#### 4. Inspect backend logs

``` bash
docker compose logs --tail=50 backend
```

#### 5. Check backend health

Open:

-   `http://localhost:8000/health`
-   `http://localhost:8000/docs`

A successful health response should resemble:

``` json
{
  "status": "healthy",
  "device": "cpu",
  "classifier_loaded": true,
  "segmentation_loaded": true
}
```

#### 6. Open the frontend

Open `http://localhost:5173` in a browser.

#### 7. Stop the application

``` bash
docker compose down
```

#### Rebuild after source-code changes

Backend changes:

``` bash
docker compose build --no-cache backend
docker compose up -d --force-recreate backend
```

Frontend changes:

``` bash
docker compose build frontend
docker compose up -d --force-recreate frontend
```

### Run the Backend Without Docker

The backend is configured for Python 3.12 in the Docker image. For a
local Python environment, use a compatible Python 3.12 installation.

Create and activate a virtual environment in PowerShell:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

``` powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the API from the project root:

``` powershell
python -m uvicorn src.api.main:app --reload
```

The API is then available at:

-   `http://localhost:8000`
-   `http://localhost:8000/docs`

The model checkpoints must exist at the paths configured in
`src/api/main.py`.

### API Documentation

Interactive OpenAPI documentation is available at:

`http://localhost:8000/docs`

#### Main endpoints

  ---------------------------------------------------------------------------
  Method                  Endpoint                    Purpose
  ----------------------- --------------------------- -----------------------
  GET                     `/`                         API information

  GET                     `/health`                   Backend and model
                                                      health

  POST                    `/predict/classification`   Chest X-ray
                                                      classification

  POST                    `/predict/gradcam`          Chest X-ray Grad-CAM
                                                      PNG

  POST                    `/predict`                  NIfTI CT segmentation

  POST                    `/predict/dicom-series`     DICOM ZIP upload and CT
                                                      segmentation

  GET                     `/slice/{scan_id}/{z}`      CT slice PNG

  GET                     `/mask/{scan_id}/{z}`       Segmentation mask
                                                      overlay PNG

  GET                     `/gradcam/{scan_id}/{z}`    3D CT Grad-CAM overlay
                                                      PNG

  POST                    `/decision`                 Save reviewer decision

  GET                     `/audit`                    Retrieve audit records

  GET                     `/audit/verify`             Verify audit integrity

  GET                     `/orthanc/results`          Retrieve Orthanc
                                                      processing results
  ---------------------------------------------------------------------------

#### NIfTI segmentation request

`POST /predict`

-   Content type: `multipart/form-data`
-   Form field: `file`
-   Accepted extensions: `.nii`, `.nii.gz`

The response contains metadata such as `scan_id`, volume shape, slice
count, tumour voxel count, estimated volume, confidence, and selected
slice indices.

Example response shape:

``` json
{
  "scan_id": "example123456",
  "filename": "scan.nii.gz",
  "shape": [128, 128, 96],
  "num_slices": 96,
  "tumour_detected": true,
  "tumour_voxels": 1000,
  "tumour_volume_ml": 4.32,
  "confidence": 0.85,
  "first_tumour_slice": 20,
  "last_tumour_slice": 42,
  "best_slice": 31,
  "device": "cpu"
}
```

The values above are illustrative only; actual results depend on the
uploaded scan and model output.

The image, mask, and Grad-CAM arrays are kept in the backend's in-memory
cache. They are retrieved through the slice, mask, and Grad-CAM
endpoints using the returned `scan_id`.

#### DICOM series request

`POST /predict/dicom-series`

-   Content type: `multipart/form-data`
-   Form field: `file`
-   Accepted input: ZIP archive containing DICOM slices.
-   Current archive limits: 2,000 files and 500 MB extracted size.

The DICOM service reads the series, converts it to a NIfTI volume for
inference, and returns the same style of scan metadata.

#### Slice, mask, and Grad-CAM requests

After receiving a `scan_id`, request:

``` text
GET /slice/{scan_id}/{z}
GET /mask/{scan_id}/{z}
GET /gradcam/{scan_id}/{z}
```

Each endpoint returns a PNG image for the requested slice index `z`.

#### Reviewer decision request

`POST /decision`

Example JSON body:

``` json
{
  "scan_id": "example123456",
  "decision": "approved",
  "reviewer": "Reviewer",
  "comment": "Reviewed for demonstration."
}
```

The `decision` value must be `approved` or `rejected`.

#### Audit endpoints

``` text
GET /audit
GET /audit/verify
```

The audit list endpoint accepts a `limit` query parameter. The integrity
endpoint reports whether the stored audit chain passes verification.

### Frontend

The React frontend is served by the frontend Docker container on port
`5173`.

The interface supports:

-   NIfTI CT upload.
-   DICOM ZIP upload.
-   CT slice navigation.
-   Segmentation-mask overlay.
-   3D CT Grad-CAM overlay.
-   Reviewer decision workflow.
-   Audit log viewing.
-   Orthanc result viewing.

The chest X-ray classification and Grad-CAM routes remain available in
the backend API. The main dashboard is focused on the CT workflow and
does not need a separate chest X-ray upload section.

### Orthanc Integration

The backend includes an Orthanc service integration and a background
worker that checks for processing results.

The Docker Compose configuration uses:

``` text
ORTHANC_URL=http://host.docker.internal:8042
```

This configuration assumes Orthanc is reachable from the Docker host at
port `8042`. Adjust the environment variable if Orthanc is hosted
elsewhere.

Orthanc result summaries are exposed through:

`GET /orthanc/results`

Orthanc must be separately installed and running; starting this
project's Docker Compose services does not automatically provision an
Orthanc server.

### Audit Logging

Reviewer decisions are recorded through the audit service. The API
supports:

-   Recording an approval or rejection.
-   Associating the decision with a scan ID and filename.
-   Recording reviewer identity and an optional comment.
-   Retrieving audit records.
-   Verifying audit integrity.

The audit system is intended for project traceability and demonstration.
It is not a substitute for a validated clinical audit or hospital
information system.

### Configuration

Important backend settings are currently defined in `src/api/main.py`:

  Setting                            Current value
  ---------------------------------- ----------------------------------
  Classifier checkpoint              `models/densenet121.pth`
  Segmentation checkpoint            `models/best_3d_unet_v3.pth`
  Device selection                   CUDA if available, otherwise CPU
  Target spacing                     `(1.5, 1.5, 2.0)`
  In-memory scan cache               3 scans
  Maximum DICOM files                2,000
  Maximum extracted DICOM ZIP size   500 MB

The backend health endpoint reports the active inference device and
whether each model loaded successfully.

### Limitations

-   This is a research and demonstration prototype, not a clinically
    validated system.
-   Model performance is limited by the training data, labels,
    preprocessing, and evaluation setup.
-   The classification model's reported precision and recall show that
    false positives and false negatives remain possible.
-   3D segmentation results can vary with scan orientation, spacing,
    preprocessing, and threshold selection.
-   Grad-CAM visualizations are approximate explanations and should not
    be interpreted as confirmed disease locations.
-   The current scan cache is in memory, limited to three scans, and
    cleared when the backend restarts.
-   CPU inference may be slow for large CT volumes.
-   Orthanc must be configured and operated separately.
-   The local Docker deployment has been tested; a public cloud
    deployment requires separate configuration, security review, and
    testing.

### Future Improvements

-   Validate models on independent, representative datasets.
-   Improve segmentation performance and investigate preprocessing/label
    alignment.
-   Add persistent, access-controlled scan storage.
-   Add authentication, authorization, and secure upload controls.
-   Add structured monitoring, error tracking, and inference performance
    metrics.
-   Improve GPU inference support and optimize large-volume processing.
-   Expand automated tests for API endpoints and medical-image input
    validation.
-   Complete and document a secure cloud deployment.
-   Conduct clinical and regulatory review before considering any
    real-world medical use.

### Responsible Use

-   Use only datasets for which you have appropriate access and usage
    rights.
-   Do not upload identifiable patient information to public demo
    environments.
-   Keep model checkpoints, patient data, DICOM files, NIfTI scans, and
    runtime databases out of public source control unless explicitly
    permitted and properly protected.
-   Treat all predictions and visualizations as experimental outputs.
-   Do not use this project as the sole basis for medical decisions.
