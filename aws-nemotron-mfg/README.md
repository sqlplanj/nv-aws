# Manufacturing Defect Detection — AWS Bedrock + NVIDIA Nemotron

A Streamlit app that uses two NVIDIA Nemotron models on **AWS Bedrock** to detect manufacturing defects in images and generate a QA batch report.

| Role | Model | Bedrock ID |
|---|---|---|
| Vision / Defect Detection | Nemotron Nano 12B VL | `nvidia.nemotron-nano-12b-v2` |
| QA Report Generation | Nemotron Super 120B | `nvidia.nemotron-super-3-120b` |

---

## How it works

1. Upload one or more manufacturing images (JPG, PNG, WEBP).
2. Each image is sent to **Nemotron Nano 12B VL** via the Bedrock Converse API for visual inspection — defect type, location, severity, confidence, and recommended action.
3. All inspection findings are aggregated and sent to **Nemotron Super 120B** to produce a professional QA batch report.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Bedrock API key

The NVIDIA serverless models on Bedrock require a **bearer token** rather than standard IAM credentials. Export it in the same terminal session before launching the app:

```bash
export AWS_BEARER_TOKEN_BEDROCK=<your-bedrock-api-key>
```

The app reads this env var automatically and injects it as an `Authorization: Bearer` header on every request. The sidebar will show **"Bearer token: detected"** when it is found.

> **Note:** Standard IAM credentials (`~/.aws/credentials`) are used as a fallback if the bearer token is not set, but are not sufficient for the NVIDIA serverless models.

### 3. Configure AWS region

The default region is **`us-west-2`**, which is where the NVIDIA Nemotron serverless models are available. You can change the region in the sidebar at runtime. Supported regions:

| Region | Models available |
|---|---|
| `us-west-2` | Both Nemotron models (default) |
| `us-east-1` | Both Nemotron models |
| `us-east-2` | Both Nemotron models |

### 4. (Optional) Download sample images

```bash
python download_samples.py
```

### 5. Run the app

```bash
# Must be the same terminal where you exported AWS_BEARER_TOKEN_BEDROCK
streamlit run app.py
```

---

## Enabling models in AWS Bedrock

1. Go to **AWS Console → Amazon Bedrock → Model access**
2. Request access to:
   - `NVIDIA Nemotron Nano 12B v2 VL BF16`
   - `NVIDIA Nemotron 3 Super 120B`
3. Access is typically granted within minutes for on-demand serverless models.

---

## Project structure

```
aws_nemotron_mfg/
├── app.py                 # Main Streamlit app
├── download_samples.py    # Helper to download sample images
├── requirements.txt
├── README.md
├── assets/
│   └── nvidia_logo.png    # NVIDIA logo shown in sidebar
└── samples/               # Sample manufacturing images (optional)
```


