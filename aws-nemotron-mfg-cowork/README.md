# 🔬 Manufacturing Defect Detection — NVIDIA Nemotron on AWS Bedrock

A Streamlit web app that demonstrates a **two-stage AI pipeline** for manufacturing quality control using NVIDIA's Nemotron models on AWS Bedrock.

---

## Architecture

```
Image Input
    │
    ▼
┌─────────────────────────────────────────┐
│  Stage 1 — Vision Analysis              │
│  NVIDIA Nemotron Nano 12B VL            │
│  nvidia.nvidia-nemotron-nano-12b-v2-    │
│  vl-bf16 (AWS Bedrock)                  │
│  → Detects & describes visual defects   │
└──────────────────┬──────────────────────┘
                   │  Vision report
                   ▼
┌─────────────────────────────────────────┐
│  Stage 2 — Deep Reasoning               │
│  NVIDIA Nemotron 3 Super 120B           │
│  nvidia.nemotron-super-3-120b (AWS      │
│  Bedrock)                               │
│  → Root cause, severity, PASS/FAIL,     │
│    corrective actions                   │
└─────────────────────────────────────────┘
```

---

## Prerequisites

1. **AWS Account** with access to Amazon Bedrock
2. **Model access** enabled for both models in your AWS account:
   - `nvidia.nvidia-nemotron-nano-12b-v2-vl-bf16`
   - `nvidia.nemotron-super-3-120b`
3. **AWS credentials** configured (via `~/.aws/credentials`, environment variables, or IAM role)

### Enable Model Access
In the AWS Console:
1. Go to **Amazon Bedrock → Model access**
2. Request access for both NVIDIA Nemotron models
3. Wait for access approval (usually instant for serverless models)

---

## Setup & Run

```bash
# 1. Clone / download this folder

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate sample images
python generate_sample_images.py

# 4. Configure AWS credentials (if not already done)
aws configure
# OR set environment variables:
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# 5. Launch the app
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## Features

| Feature | Description |
|---|---|
| Sample Images | 4 pre-generated manufacturing images (PCB, metal, weld, casting) |
| Custom Upload | Upload any JPG/PNG manufacturing image |
| Stage 1 Analysis | Nemotron Nano 12B VL describes defects visually |
| Stage 2 Reasoning | Nemotron 3 Super 120B provides root cause, severity, PASS/FAIL |
| Editable Prompts | Customize the inspection prompt per use case |
| Metrics | Latency, token usage displayed for each model call |

---

## Sample Images Included

| Image | Defect Type |
|---|---|
| `pcb_solder_bridge_defect.jpg` | PCB solder bridge between adjacent pads |
| `metal_surface_crack.jpg` | Surface crack in metal part |
| `weld_porosity_defects.jpg` | Porosity pits in weld bead |
| `casting_no_defect_ok.jpg` | Good cast part (no defects — baseline) |

---

## AWS Bedrock Models

| Model | ID | Use |
|---|---|---|
| Nemotron Nano 12B VL | `nvidia.nvidia-nemotron-nano-12b-v2-vl-bf16` | Multimodal vision-language analysis |
| Nemotron 3 Super 120B | `nvidia.nemotron-super-3-120b` | Complex reasoning & recommendations |

Both models are accessed via the **Bedrock Converse API** for a consistent interface.
