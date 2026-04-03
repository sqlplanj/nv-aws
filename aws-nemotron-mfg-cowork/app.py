"""
Manufacturing Defect Detection Demo
Powered by AWS Bedrock – NVIDIA Nemotron Models

Models used:
  • nvidia.nvidia-nemotron-nano-12b-v2-vl-bf16  – Vision-Language model (image analysis)
  • nvidia.nemotron-super-3-120b               – Super reasoning model (deep analysis + recommendations)
"""

import base64
import io
import json
import os
import time

import boto3
import requests
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError
from PIL import Image

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Manufacturing Defect Detection | NVIDIA on AWS Bedrock",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Model IDs ─────────────────────────────────────────────────────────────────
VL_MODEL_ID = "nvidia.nemotron-nano-12b-v2"      # Vision-Language (Nemotron Nano 12B v2 VL)
SUPER_MODEL_ID = "nvidia.nemotron-super-3-120b"  # Reasoning (Nemotron 3 Super 120B)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_images")

SAMPLE_IMAGES = {
    "PCB – Solder Bridge Defect": "pcb_solder_bridge_defect.jpg",
    "Metal Surface – Crack": "metal_surface_crack.jpg",
    "Weld Bead – Porosity": "weld_porosity_defects.jpg",
    "Cast Part – No Defect (OK)": "casting_no_defect_ok.jpg",
}

DEFECT_PROMPTS = {
    "PCB – Solder Bridge Defect": (
        "You are a PCB quality control expert. Analyze this PCB image for manufacturing defects. "
        "Look specifically for: solder bridges, cold solder joints, missing components, misaligned pads, "
        "or lifted traces. Describe what you observe, the location of any defects, and their severity."
    ),
    "Metal Surface – Crack": (
        "You are a materials inspection engineer. Analyze this metal surface image for defects. "
        "Look for: surface cracks, fractures, pitting, corrosion, or abnormal surface texture. "
        "Describe the nature, location, length estimate, and severity of any defects found."
    ),
    "Weld Bead – Porosity": (
        "You are a certified welding inspector (CWI). Analyze this weld inspection image for defects. "
        "Look for: porosity (gas pores), undercut, lack of fusion, cracks, or uneven bead profile. "
        "Describe each defect found, its location, approximate size, and impact on weld integrity."
    ),
    "Cast Part – No Defect (OK)": (
        "You are a casting quality control inspector. Analyze this cast part image. "
        "Look for: surface porosity, shrinkage cavities, cold shuts, flash, misruns, or dimensional issues. "
        "Provide a thorough quality assessment."
    ),
    "Custom Upload": (
        "You are a manufacturing quality control expert with expertise in visual defect detection. "
        "Carefully analyze this manufacturing component image for any defects, anomalies, or quality issues. "
        "Describe: (1) what type of part/material you see, (2) any defects or anomalies present, "
        "(3) location and severity of each issue found, (4) whether the part appears to be PASS or FAIL."
    ),
}


# ── AWS Bedrock Client ────────────────────────────────────────────────────────

# Base URL template for direct HTTP calls (bearer token path)
BEDROCK_ENDPOINT = "https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/converse"


@st.cache_resource
def get_bedrock_client(region: str, access_key: str = "", secret_key: str = ""):
    """Boto3 client — used only for IAM credential auth path."""
    kwargs = {"region_name": region}
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
    return boto3.client("bedrock-runtime", **kwargs)


def _bearer_converse(region: str, bearer_token: str, model_id: str, body: dict) -> dict:
    """
    Call the Bedrock Converse API directly via HTTPS using a bearer token.
    Bypasses boto3/botocore entirely so there are no version or env-var
    timing issues.  Authorization: Bearer <token> is sent as a plain header.
    """
    url = BEDROCK_ENDPOINT.format(region=region, model_id=model_id)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {bearer_token}",
    }
    resp = requests.post(url, headers=headers, json=body, timeout=120)
    resp.raise_for_status()
    return resp.json()


def image_to_base64(pil_img: Image.Image, fmt: str = "JPEG") -> tuple[str, str]:
    """Convert PIL image to base64 string and return (b64_string, media_type)."""
    buf = io.BytesIO()
    if fmt == "JPEG":
        pil_img = pil_img.convert("RGB")
    pil_img.save(buf, format=fmt)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    media_type = "image/jpeg" if fmt == "JPEG" else "image/png"
    return b64, media_type


# ── Model Calls ───────────────────────────────────────────────────────────────

def _parse_converse_response(response: dict, t0: float) -> dict:
    """Extract text + usage from a Converse API response (same shape for both paths)."""
    elapsed = int((time.time() - t0) * 1000)
    output_text = response["output"]["message"]["content"][0]["text"]
    usage = response.get("usage", {})
    return {
        "text": output_text,
        "input_tokens": usage.get("inputTokens", "N/A"),
        "output_tokens": usage.get("outputTokens", "N/A"),
        "latency_ms": elapsed,
        "error": None,
    }


def call_vision_model(
    region: str, bearer_token: str, client, image_b64: str, media_type: str, prompt: str
) -> dict:
    """
    Call Nemotron Nano 12B VL via Bedrock Converse API.
    Uses direct HTTPS + Bearer token when bearer_token is set; boto3 otherwise.
    """
    img_bytes = base64.b64decode(image_b64)
    img_fmt = media_type.split("/")[1]  # "jpeg" or "png"

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"image": {"format": img_fmt, "source": {"bytes": img_bytes}}},
                    {"text": prompt},
                ],
            }
        ],
        "inferenceConfig": {"maxTokens": 1024, "temperature": 0.1},
    }

    t0 = time.time()
    try:
        if bearer_token:
            # Direct HTTP — Authorization: Bearer <token>
            # Image bytes must be base64-encoded in JSON for the HTTP path
            body["messages"][0]["content"][0]["image"]["source"]["bytes"] = image_b64
            response = _bearer_converse(region, bearer_token, VL_MODEL_ID, body)
        else:
            # boto3 Converse — passes raw bytes natively
            response = client.converse(modelId=VL_MODEL_ID, **{k: v for k, v in body.items() if k != "inferenceConfig"},
                                       inferenceConfig=body["inferenceConfig"])
        return _parse_converse_response(response, t0)
    except requests.HTTPError as e:
        return {"text": "", "error": f"HTTP {e.response.status_code}: {e.response.text}", "latency_ms": int((time.time() - t0) * 1000)}
    except (ClientError, NoCredentialsError) as e:
        return {"text": "", "error": str(e), "latency_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"text": "", "error": f"Unexpected error: {e}", "latency_ms": 0}


def call_super_model(
    region: str, bearer_token: str, client, vision_analysis: str, context: str
) -> dict:
    """
    Call Nemotron 3 Super 120B via Bedrock Converse API.
    Uses direct HTTPS + Bearer token when bearer_token is set; boto3 otherwise.
    """
    system_prompt = (
        "You are a senior manufacturing quality engineer and defect analysis specialist. "
        "You receive visual inspection reports from an AI vision system and provide: "
        "(1) A structured defect severity assessment, "
        "(2) Root cause hypothesis, "
        "(3) Recommended corrective actions, "
        "(4) A final PASS / FAIL / NEEDS-REVIEW disposition, "
        "(5) Priority level (Critical / High / Medium / Low). "
        "Be concise, precise, and actionable."
    )
    user_message = (
        f"Context: {context}\n\n"
        f"Vision Analysis Report:\n{vision_analysis}\n\n"
        "Based on this visual inspection report, provide your full quality engineering assessment."
    )

    body = {
        "system": [{"text": system_prompt}],
        "messages": [{"role": "user", "content": [{"text": user_message}]}],
        "inferenceConfig": {"maxTokens": 1024, "temperature": 0.2},
    }

    t0 = time.time()
    try:
        if bearer_token:
            response = _bearer_converse(region, bearer_token, SUPER_MODEL_ID, body)
        else:
            response = client.converse(
                modelId=SUPER_MODEL_ID,
                system=body["system"],
                messages=body["messages"],
                inferenceConfig=body["inferenceConfig"],
            )
        return _parse_converse_response(response, t0)
    except requests.HTTPError as e:
        return {"text": "", "error": f"HTTP {e.response.status_code}: {e.response.text}", "latency_ms": int((time.time() - t0) * 1000)}
    except (ClientError, NoCredentialsError) as e:
        return {"text": "", "error": str(e), "latency_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"text": "", "error": f"Unexpected error: {e}", "latency_ms": 0}


# ── UI Helpers ────────────────────────────────────────────────────────────────

def render_result_card(title: str, icon: str, result: dict, color: str = "#1e3a5f"):
    if result.get("error"):
        st.error(f"**{icon} {title} – Error:** {result['error']}")
        return

    with st.container():
        st.markdown(
            f"""<div style="background:{color};border-radius:10px;padding:16px 20px;margin-bottom:12px;">
            <h4 style="color:#fff;margin:0 0 4px 0;">{icon} {title}</h4>
            <span style="color:#aac8ff;font-size:0.8em;">
              ⏱ {result['latency_ms']} ms &nbsp;|&nbsp;
              📥 {result.get('input_tokens','–')} tokens in &nbsp;|&nbsp;
              📤 {result.get('output_tokens','–')} tokens out
            </span></div>""",
            unsafe_allow_html=True,
        )
        st.markdown(result["text"])


def disposition_badge(text: str):
    """Scan text for PASS/FAIL/NEEDS-REVIEW and render a badge."""
    text_upper = text.upper()
    if "FAIL" in text_upper and "PASS" not in text_upper:
        st.error("🔴 **Disposition: FAIL** — Part rejected")
    elif "NEEDS-REVIEW" in text_upper or "NEEDS REVIEW" in text_upper:
        st.warning("🟡 **Disposition: NEEDS REVIEW** — Manual inspection required")
    elif "PASS" in text_upper:
        st.success("🟢 **Disposition: PASS** — Part accepted")
    else:
        st.info("⚪ **Disposition: Unknown** — Review results below")


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/sco/2/21/Nvidia_logo.svg",
        width=120,
    )
    st.sidebar.title("⚙️ Configuration")

    # ── AWS Credentials ──────────────────────────────────────────────────────
    st.sidebar.markdown("### 🔑 AWS Credentials")

    bearer_token = st.sidebar.text_input(
        "Bedrock API Key",
        type="password",
        placeholder="baak-…",
        help=(
            "Your Amazon Bedrock API key (bearer token). "
            "Generate one in the AWS Console under Bedrock → API keys. "
            "Set as AWS_BEARER_TOKEN_BEDROCK env var, or enter it here."
        ),
    )

    # Status indicator only — actual env var is set inside get_bedrock_client()
    # so it is guaranteed to be set before boto3.client() is constructed.
    if bearer_token:
        st.sidebar.success("✅ Bedrock API key provided.")
    elif os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        bearer_token = os.environ["AWS_BEARER_TOKEN_BEDROCK"]
        st.sidebar.success("✅ Bedrock API key loaded from environment.")
    else:
        st.sidebar.info("ℹ️ No API key — falling back to IAM credentials.")

    st.sidebar.markdown("**Or use IAM credentials instead:**")
    access_key = st.sidebar.text_input(
        "AWS Access Key ID",
        type="password",
        placeholder="AKIA… (optional if API key set)",
        help="Your AWS IAM Access Key ID. Not needed if a Bedrock API key is provided.",
    )
    secret_key = st.sidebar.text_input(
        "AWS Secret Access Key",
        type="password",
        placeholder="(optional if API key set)",
        help="Your AWS IAM Secret Access Key.",
    )

    st.sidebar.markdown("---")

    region = st.sidebar.selectbox(
        "AWS Region",
        ["us-west-2", "us-east-1", "eu-west-1", "ap-southeast-1"],
        index=0,
        help="Must match the region where you have Bedrock model access enabled.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Models")
    st.sidebar.markdown(
        f"""
**Vision Analysis**
`{VL_MODEL_ID}`
*Nemotron Nano 12B v2 VL – Multimodal image understanding*

**Deep Reasoning**
`{SUPER_MODEL_ID}`
*Nemotron 3 Super 120B – Complex reasoning & recommendations*
"""
    )

    st.sidebar.markdown("---")
    use_super = st.sidebar.toggle(
        "Enable Super Model Analysis",
        value=True,
        help="After vision analysis, pass results to Nemotron 3 Super for deeper reasoning and recommendations.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 About")
    st.sidebar.markdown(
        """
This demo shows a **two-stage AI pipeline** for manufacturing defect detection:

1. 🔭 **Stage 1** – Nemotron Nano 12B VL analyzes the image visually
2. 🧠 **Stage 2** – Nemotron 3 Super 120B reasons over the findings and provides engineering recommendations

Both models run on **AWS Bedrock** as fully managed serverless endpoints.
"""
    )

    return region, bearer_token, access_key, secret_key, use_super


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    region, bearer_token, access_key, secret_key, use_super = render_sidebar()

    # ── Header
    st.markdown(
        """
        <h1 style="margin-bottom:0;">🔬 Manufacturing Defect Detection</h1>
        <p style="color:#888;font-size:1.05em;margin-top:4px;">
            Powered by <strong>NVIDIA Nemotron</strong> on <strong>AWS Bedrock</strong>
        </p>
        <hr style="margin:12px 0 20px 0;">
        """,
        unsafe_allow_html=True,
    )

    # ── Image Source Selection
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("📸 Select Image")
        source_tab, upload_tab = st.tabs(["Sample Images", "Upload Your Own"])

        selected_image: Image.Image | None = None
        selected_label = "Custom Upload"

        with source_tab:
            choice = st.selectbox("Choose a sample:", list(SAMPLE_IMAGES.keys()))
            img_path = os.path.join(SAMPLE_DIR, SAMPLE_IMAGES[choice])
            if os.path.exists(img_path):
                selected_image = Image.open(img_path)
                selected_label = choice
                st.image(selected_image, caption=choice, use_container_width=True)
            else:
                st.warning("Sample image not found. Run `generate_sample_images.py` first.")

        with upload_tab:
            uploaded = st.file_uploader(
                "Upload a manufacturing image",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                help="Upload a photo of a part, weld, PCB, or surface to inspect.",
            )
            if uploaded:
                selected_image = Image.open(uploaded)
                selected_label = "Custom Upload"
                st.image(selected_image, caption=uploaded.name, use_container_width=True)

    with col_right:
        st.subheader("🧩 Inspection Prompt")
        default_prompt = DEFECT_PROMPTS.get(selected_label, DEFECT_PROMPTS["Custom Upload"])
        custom_prompt = st.text_area(
            "Customize the inspection prompt (optional):",
            value=default_prompt,
            height=180,
        )

        st.markdown("---")
        analyze_btn = st.button(
            "🚀 Run Defect Analysis",
            type="primary",
            use_container_width=True,
            disabled=(selected_image is None),
        )

    # ── Analysis
    if analyze_btn and selected_image is not None:
        st.markdown("---")
        st.subheader("📊 Analysis Results")

        try:
            client = get_bedrock_client(region, access_key, secret_key)
        except Exception as e:
            st.error(f"Failed to create Bedrock client: {e}")
            return

        # Prepare image
        img_b64, media_type = image_to_base64(selected_image)

        # ── Stage 1: Vision Model
        with st.spinner(f"🔭 Stage 1 — Nemotron Nano 12B VL is analyzing the image…"):
            vl_result = call_vision_model(region, bearer_token, client, img_b64, media_type, custom_prompt)

        render_result_card(
            "Nemotron Nano 12B VL — Visual Inspection",
            "🔭",
            vl_result,
            color="#1a3a5c",
        )

        if vl_result.get("error"):
            st.stop()

        # ── Stage 2: Super Model
        if use_super:
            with st.spinner(f"🧠 Stage 2 — Nemotron 3 Super 120B is reasoning over findings…"):
                super_result = call_super_model(
                    region, bearer_token, client, vl_result["text"], context=selected_label
                )

            render_result_card(
                "Nemotron 3 Super 120B — Engineering Assessment",
                "🧠",
                super_result,
                color="#2d1f4e",
            )

            if not super_result.get("error"):
                st.markdown("---")
                disposition_badge(super_result["text"])

                # Token summary
                total_in = sum(
                    r.get("input_tokens", 0)
                    for r in [vl_result, super_result]
                    if isinstance(r.get("input_tokens"), int)
                )
                total_out = sum(
                    r.get("output_tokens", 0)
                    for r in [vl_result, super_result]
                    if isinstance(r.get("output_tokens"), int)
                )
                total_ms = vl_result["latency_ms"] + super_result["latency_ms"]
                st.caption(
                    f"📈 **Pipeline summary** — Total latency: {total_ms} ms | "
                    f"Tokens in: {total_in} | Tokens out: {total_out}"
                )
        else:
            st.markdown("---")
            disposition_badge(vl_result["text"])

    elif not analyze_btn:
        st.info(
            "👆 Select or upload an image above, then click **Run Defect Analysis** to start.",
            icon="ℹ️",
        )


if __name__ == "__main__":
    main()
