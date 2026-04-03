"""
Manufacturing Defect Detection with NVIDIA Nemotron on AWS Bedrock

- Vision analysis: nvidia.nemotron-nano-12b-v2 (128K ctx, image + text)
- Report generation: nvidia.nemotron-super-3-120b (256K ctx, text only)
"""

import json
import base64
import os
import boto3
import streamlit as st
from pathlib import Path

# ── Model IDs ────────────────────────────────────────────────────────────────
VL_MODEL   = "nvidia.nemotron-nano-12b-v2"       # Vision-Language, defect detection
SUPER_MODEL = "nvidia.nemotron-super-3-120b"     # Text-only, report generation

REGION = "us-west-2"

# ── Prompts ───────────────────────────────────────────────────────────────────
DEFECT_PROMPT = """You are an expert quality-control inspector for a manufacturing facility.

Carefully examine this image and provide:
1. **Defect Detected** – Yes / No / Uncertain
2. **Defect Type** – (e.g., crack, scratch, deformation, discoloration, missing component, contamination, weld flaw, surface void, etc.) — list all you see
3. **Location** – where in the image the defect appears
4. **Severity** – Critical / Major / Minor / None
5. **Confidence** – your confidence in this assessment (Low / Medium / High)
6. **Recommended Action** – what should happen next (e.g., reject, rework, flag for reinspection)

Be concise and precise. Use bullet points."""

REPORT_PROMPT_TEMPLATE = """You are a manufacturing quality assurance manager.

Below are vision-inspection results for {n} image(s) analyzed during a production run.

--- INSPECTION RESULTS ---
{findings}
--- END RESULTS ---

Write a concise QA summary report that includes:
1. **Overall Pass/Fail Rate**
2. **Most Common Defect Types**
3. **Risk Assessment** for this batch
4. **Recommended Process Improvements** to reduce defects
5. **Immediate Actions Required**

Keep the report professional and actionable."""

# ── AWS client (cached per region) ───────────────────────────────────────────
@st.cache_resource
def get_bedrock_client(region: str):
    bearer_token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")

    if bearer_token:
        # Bearer token auth — used for NVIDIA serverless models on Bedrock.
        # Inject the Authorization header after SigV4 signing so it takes effect.
        client = boto3.client("bedrock-runtime", region_name=region)

        def _inject_bearer(request, **kwargs):
            request.headers["Authorization"] = f"Bearer {bearer_token}"

        client.meta.events.register("before-send.bedrock-runtime.*", _inject_bearer)
        return client

    # Fallback: standard IAM credentials from ~/.aws/credentials or env vars
    return boto3.client("bedrock-runtime", region_name=region)

# ── Inference helpers ─────────────────────────────────────────────────────────
def analyze_image(client, image_bytes: bytes, image_format: str) -> str:
    """Send image to Nemotron Nano 12B VL for defect analysis."""
    response = client.converse(
        modelId=VL_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"text": DEFECT_PROMPT},
                {
                    "image": {
                        "format": image_format,
                        "source": {"bytes": image_bytes}
                    }
                }
            ]
        }],
        inferenceConfig={"maxTokens": 1024, "temperature": 0.1}
    )
    return response["output"]["message"]["content"][0]["text"]


def generate_report(client, findings: list[str]) -> str:
    """Send aggregated findings to Nemotron Super 120B for report generation."""
    formatted = "\n\n".join(
        f"Image {i+1}:\n{f}" for i, f in enumerate(findings)
    )
    prompt = REPORT_PROMPT_TEMPLATE.format(n=len(findings), findings=formatted)

    response = client.converse(
        modelId=SUPER_MODEL,
        messages=[{
            "role": "user",
            "content": [{"text": prompt}]
        }],
        inferenceConfig={"maxTokens": 2048, "temperature": 0.3}
    )
    return response["output"]["message"]["content"][0]["text"]


# ── UI helpers ────────────────────────────────────────────────────────────────
def severity_badge(text: str) -> str:
    """Return colored badge markdown based on severity keyword in text."""
    t = text.lower()
    if "critical" in t:
        return "🔴 Critical"
    if "major" in t:
        return "🟠 Major"
    if "minor" in t:
        return "🟡 Minor"
    if "none" in t or "no defect" in t:
        return "🟢 None"
    return "⚪ Unknown"


EXT_TO_FORMAT = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png",
                 ".gif": "gif",  ".webp": "webp"}

# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Manufacturing Defect Detector",
        page_icon="🔬",
        layout="wide"
    )

    st.title("🔬 Manufacturing Defect Detection")
    st.markdown(
        "**Vision analysis** → `nvidia.nemotron-nano-12b-v2` (12B VL)  "
        "| **QA Report** → `nvidia.nemotron-super-3-120b` (120B)  "
        "| Powered by **AWS Bedrock**"
    )
    st.divider()

    # ── Logos (base64-encoded for offline reliability) ────────────────────────
    assets = Path(__file__).parent / "assets"

    def _b64(path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode()

    aws_logo_url = (
        "https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg"
    )
    nvidia_b64 = _b64(assets / "nvidia_logo.png") if (assets / "nvidia_logo.png").exists() else None

    nvidia_img_tag = (
        f'<img src="data:image/png;base64,{nvidia_b64}" alt="NVIDIA">'
        if nvidia_b64
        else '<span style="font-weight:bold;color:#76b900;">NVIDIA</span>'
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f"""
            <style>
            .sidebar-logos {{
                display: flex;
                flex-direction: row;
                align-items: center;
                gap: 8px;
                margin-bottom: 20px;
            }}
            .sidebar-logos .aws-logo {{ width: 60px; object-fit: contain; }}
            .sidebar-logos .nv-logo  {{ width: 100px; object-fit: contain; }}
            .sidebar-logos .plus {{
                font-size: 18px;
                font-weight: bold;
                color: #555;
                line-height: 1;
            }}
            </style>
            <div class="sidebar-logos">
                <img class="aws-logo" src="{aws_logo_url}" alt="AWS">
                <span class="plus">+</span>
                {nvidia_img_tag.replace('<img ', '<img class="nv-logo" ')}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.header("⚙️ Settings")
        aws_region = st.selectbox(
            "AWS Region",
            ["us-west-2", "us-east-1", "us-east-2"],
            index=0
        )
        st.markdown("---")
        st.markdown("**Models used**")
        st.markdown(f"- VL: `{VL_MODEL}`")
        st.markdown(f"- Super: `{SUPER_MODEL}`")
        st.markdown("---")
        if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
            st.success("Bearer token: detected")
        else:
            st.warning("Bearer token: not set\n\n`export AWS_BEARER_TOKEN_BEDROCK=...`")

    # ── File upload ───────────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload manufacturing images (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

    # Show sample images from /samples if they exist
    samples_dir = Path(__file__).parent / "samples"
    sample_files = sorted(samples_dir.glob("*")) if samples_dir.exists() else []
    if sample_files and not uploaded_files:
        st.info(f"💡 No files uploaded. Found {len(sample_files)} sample image(s) in `samples/`. "
                "Upload your own or click **Run on Samples** below.")
        use_samples = st.button("▶️ Run on Samples")
    else:
        use_samples = False

    # Determine which images to process
    images_to_process = []  # list of (name, bytes, format)

    if uploaded_files:
        for f in uploaded_files:
            ext = Path(f.name).suffix.lower()
            fmt = EXT_TO_FORMAT.get(ext, "jpeg")
            images_to_process.append((f.name, f.read(), fmt))
    elif use_samples:
        for p in sample_files:
            ext = p.suffix.lower()
            fmt = EXT_TO_FORMAT.get(ext, "jpeg")
            images_to_process.append((p.name, p.read_bytes(), fmt))

    if not images_to_process:
        st.stop()

    # ── Analyze ───────────────────────────────────────────────────────────────
    st.subheader(f"📸 Analyzing {len(images_to_process)} image(s)…")

    try:
        client = get_bedrock_client(aws_region)
    except Exception as e:
        st.error(f"Failed to create Bedrock client: {e}")
        st.stop()

    findings = []
    cols_per_row = min(len(images_to_process), 3)

    for idx in range(0, len(images_to_process), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_i, (name, img_bytes, fmt) in enumerate(
            images_to_process[idx: idx + cols_per_row]
        ):
            with cols[col_i]:
                st.image(img_bytes, caption=name, use_container_width=True)
                with st.spinner(f"Inspecting {name}…"):
                    try:
                        result = analyze_image(client, img_bytes, fmt)
                        findings.append(result)
                        badge = severity_badge(result)
                        st.markdown(f"**Severity:** {badge}")
                        with st.expander("📋 Full Inspection Report", expanded=False):
                            st.markdown(result)
                    except Exception as e:
                        err = f"Error analyzing {name}: {e}"
                        st.error(err)
                        findings.append(err)

    # ── QA Report ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Batch QA Report  *(Nemotron Super 120B)*")

    with st.spinner("Generating QA report with Nemotron Super 120B…"):
        try:
            report = generate_report(client, findings)
            st.markdown(report)
        except Exception as e:
            st.error(f"Failed to generate report: {e}")

    st.divider()
    st.caption(
        "Vision: `nvidia.nemotron-nano-12b-v2` · "
        "Report: `nvidia.nemotron-super-3-120b` · "
        f"AWS Bedrock · Region: {aws_region}"
    )


if __name__ == "__main__":
    main()
