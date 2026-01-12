import os
import io
import base64

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from openai import OpenAI

from utils import crop_headshot, basic_polish, to_png_bytes

# -------------------------------------------------
# Environment
# -------------------------------------------------
load_dotenv()  # expects OPENAI_API_KEY in .env

# -------------------------------------------------
# Page config + light styling
# -------------------------------------------------
st.set_page_config(page_title="Selfie → Pro Headshot", page_icon="📸", layout="centered")

st.markdown(
    """

<style>
/* Image "cards" */
.stImage { background:#f6f7f9; padding:8px; border-radius:12px; }
/* Slightly tighten vertical spacing */
.block-container { padding-top: 2.2rem; }

/* Eye-catching Demo button */
.demo-cta div[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #7C3AED, #3B82F6);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.75em 1em;
    font-weight: 700;
    box-shadow: 0 8px 22px rgba(0,0,0,0.10);
}
.demo-cta div[data-testid="stButton"] > button:hover {
    filter: brightness(1.08);
    transform: translateY(-1px);
}
.demo-cta div[data-testid="stButton"] > button:active {
    transform: translateY(0px);
}
/* Readable subtitle (inherits theme text color) */
.subtitle {
    margin-top: -0.5rem;
    margin-bottom: 1rem;
    font-size: 0.95rem;
    opacity: 0.85;
    color: inherit;
}

/* Reset demo button (subtle but clear) */
.reset-demo div[data-testid="stButton"] > button {
    border-radius: 14px;
    font-weight: 700;
    padding: 0.65em 1em;
}
</style>


""",
    unsafe_allow_html=True,
)

st.title("📸 Selfie → Professional Headshot")
st.markdown("<div class='subtitle'>Upload a selfie, preview the crop, and generate a studio-style professional headshot.</div>", unsafe_allow_html=True)

with st.expander("Privacy", expanded=False):
    st.write("• Your photos are only used to create your headshot and are never saved.")
    st.write("• Images are processed in memory and discarded after your session ends.")
    st.write("• We don’t collect, store, or reuse your photos for any purpose.")


# -------------------------------------------------
# Sidebar controls (cleaner UX)
# -------------------------------------------------
st.sidebar.header("Settings")
bg_style = st.sidebar.selectbox("Background", ["Neutral gray", "Office blur", "Pure white"])
outfit = st.sidebar.selectbox("Outfit style", ["Business casual", "Formal suit", "Smart casual"])
use_ai = st.sidebar.toggle("Use AI studio headshot", value=True)
show_prompt = st.sidebar.toggle("Show prompt", value=False)

prompt = f"""
Turn this selfie into a professional studio headshot.
- Clean, realistic, natural skin texture (no plastic look)
- Professional lighting (soft key light, subtle rim light)
- {outfit} attire (realistic fabric)
- Background: {bg_style}
- Keep identity and facial features consistent
- Head-and-shoulders framing, 4:5 portrait
- No watermark, no text
""".strip()

if show_prompt:
    st.text_area("Generation instructions", prompt, height=140)

# -------------------------------------------------
# OpenAI image edit call
# -------------------------------------------------
def call_headshot_openai(image_bytes: bytes, prompt_text: str) -> bytes:
    """Generate a headshot via OpenAI Images Edit (gpt-image-1)."""
    client = OpenAI()  # uses OPENAI_API_KEY from env

    image_file = io.BytesIO(image_bytes)
    image_file.name = "selfie.png"

    result = client.images.edit(
        model="gpt-image-1.5",
        image=image_file,
        prompt=prompt_text,
        size="1024x1536",
        output_format="png",
    )

    b64 = result.data[0].b64_json
    return base64.b64decode(b64)

# -------------------------------------------------
# Main UI
# -------------------------------------------------
col_up, col_demo = st.columns([2, 1])

with col_up:
    uploaded = st.file_uploader("Upload a selfie (JPG/PNG)", type=["jpg", "jpeg", "png"])

with col_demo:
    st.markdown('<div class="demo-cta">', unsafe_allow_html=True)
    demo_clicked = st.button("✨ Try a demo photo", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    if demo_clicked:
        st.session_state["use_demo"] = True
        # switching inputs should also clear previous AI result
        st.session_state.pop("ai_image", None)
        st.rerun()

    if st.session_state.get("use_demo") and not uploaded:
        st.markdown('<div class="reset-demo">', unsafe_allow_html=True)
        reset_clicked = st.button("↩ Reset demo", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if reset_clicked:
            st.session_state.pop("use_demo", None)
            st.session_state.pop("ai_image", None)
            st.rerun()



if uploaded or st.session_state.get("use_demo"):
    if st.session_state.get("use_demo") and not uploaded:
        raw = Image.open("demo.jpg").convert("RGB")
    else:
        raw = Image.open(uploaded).convert("RGB")
    cropped = crop_headshot(raw, out_size=1024)
    polished = basic_polish(cropped)
    if st.session_state.get("use_demo") and not uploaded:
        st.info("Using demo photo (demo.jpg). Upload your own selfie anytime to replace it.")


    # Action row (above previews)
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        gen_clicked = st.button(
            "Generate professional headshot",
            type="primary",
            disabled=not use_ai,
            use_container_width=True,
        )
    with col_btn2:
        st.download_button(
            "Download crop + polish",
            data=to_png_bytes(polished),
            file_name="headshot_polished.png",
            mime="image/png",
            use_container_width=True,
        )

    if gen_clicked and use_ai:
        with st.spinner("Generating..."):
            try:
                out_bytes = call_headshot_openai(to_png_bytes(polished), prompt)
                st.session_state["ai_image"] = out_bytes
            except Exception as e:
                st.error(f"Generation failed: {e}")

    st.subheader("Preview")

    # 3-column pipeline (small → small → big)
    col1, col2, col3 = st.columns([1, 1, 1.5])

    with col1:
        st.caption("Input")
        st.image(raw, use_container_width=True)

    with col2:
        st.caption("Crop + Polish")
        st.image(polished, use_container_width=True)

    with col3:
        st.caption("AI Headshot")
        if "ai_image" in st.session_state:
            st.image(st.session_state["ai_image"], use_container_width=True)
            st.download_button(
                "Download AI headshot",
                data=st.session_state["ai_image"],
                file_name="headshot_ai.png",
                mime="image/png",
                use_container_width=True,
            )
        else:
            st.info("Click **Generate** to create your headshot.")
else:
    st.info("Upload a selfie to see the preview pipeline.")
