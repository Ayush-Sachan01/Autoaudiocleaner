"""
Streamlit interface for the noise-reduction + silence-removal pipeline.

Setup
-----
1. Save your original script next to this file as  cleaner.py
2. pip install streamlit
3. streamlit run app.py
"""

import logging
import os
import re
import shutil
import tempfile
import traceback

import streamlit as st

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Clean audio", page_icon="🎙️", layout="centered")

INK = "#16222A"
MUTED = "#5B6B73"
PAPER = "#EFF2F1"
SURFACE = "#FFFFFF"
LINE = "#D5DBDD"
ACCENT = "#3340D9"
ACCENT_SOFT = "#E6E8FB"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Public+Sans:wght@400;500;600&display=swap');

    html, body, .stApp, [class*="css"] {{
        font-family: 'Public Sans', system-ui, sans-serif;
        color: {INK};
    }}
    .stApp {{ background: {PAPER}; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    #MainMenu, footer {{ visibility: hidden; }}
    .block-container {{ max-width: 760px; padding-top: 2.5rem; padding-bottom: 5rem; }}

    h1, h2, h3, h4 {{
        font-family: 'Bricolage Grotesque', 'Public Sans', sans-serif !important;
        letter-spacing: -0.02em;
        color: {INK};
    }}
    h3 {{ font-size: 1.25rem !important; margin-top: 1.6rem; }}

    .hero {{ margin-bottom: 1.6rem; }}
    .hero svg {{ display: block; margin-bottom: 1.1rem; }}
    .hero h1 {{
        font-size: 3.4rem; line-height: 1; font-weight: 700;
        margin: 0 0 0.7rem 0; padding: 0;
    }}
    .hero p {{ color: {MUTED}; font-size: 1.05rem; max-width: 34rem; margin: 0; }}

    /* Bordered groups */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {SURFACE};
        border-color: {LINE};
        border-radius: 14px;
    }}

    /* File drop zone */
    [data-testid="stFileUploaderDropzone"] {{
        background: {SURFACE};
        border: 1.5px dashed {ACCENT};
        border-radius: 14px;
        padding: 1.6rem;
    }}
    [data-testid="stFileUploaderDropzone"]:hover {{ background: {ACCENT_SOFT}; }}

    /* Buttons */
    .stButton > button[kind="primary"] {{
        background: {ACCENT}; border: none; color: white;
        border-radius: 10px; padding: 0.75rem 1rem; font-weight: 600; font-size: 1rem;
    }}
    .stButton > button[kind="primary"]:hover {{ background: #2631B8; }}
    .stButton > button[kind="primary"]:disabled {{ background: {LINE}; color: {MUTED}; }}
    .stDownloadButton > button {{
        border-radius: 10px; border: 1.5px solid {ACCENT}; color: {ACCENT};
        font-weight: 600; padding: 0.65rem 1rem; background: {SURFACE};
    }}
    .stDownloadButton > button:hover {{ background: {ACCENT_SOFT}; color: {ACCENT}; border-color: {ACCENT}; }}

    /* Metrics */
    [data-testid="stMetricValue"] {{
        font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; color: {INK};
    }}
    [data-testid="stMetricLabel"] {{ color: {MUTED}; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-highlight"] {{ background: {ACCENT}; }}
    .stTabs [aria-selected="true"] {{ color: {ACCENT}; }}

    .small-note {{ color: {MUTED}; font-size: 0.9rem; margin-top: -0.4rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def waveform_mark() -> str:
    """Small brand mark: a row of bars shaped like a spoken phrase."""
    heights = [6, 10, 18, 30, 22, 36, 14, 8, 8, 20, 34, 40, 26, 12, 6, 6, 16, 28, 18, 8]
    bar_w, gap, box_h = 6, 5, 44
    width = len(heights) * (bar_w + gap) - gap
    bars = "".join(
        f'<rect x="{i * (bar_w + gap)}" y="{(box_h - h) / 2}" width="{bar_w}" '
        f'height="{h}" rx="3" fill="{ACCENT if h > 10 else LINE}"/>'
        for i, h in enumerate(heights)
    )
    return (
        f'<svg width="{width}" height="{box_h}" viewBox="0 0 {width} {box_h}" '
        f'aria-hidden="true">{bars}</svg>'
    )


st.markdown(
    f"""
    <div class="hero">
        {waveform_mark()}
        <h1>Clean audio</h1>
        <p>Remove background noise and cut out silence from voice recordings and videos.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}

NOISE_OPTIONS = {
    "deepfilter": "DeepFilterNet",
    "spectral": "Spectral gating",
    "none": "Off",
}
NOISE_HELP = {
    "deepfilter": "Best quality. Neural network, much faster with a GPU.",
    "spectral": "Fast and light. Good for steady hiss or hum.",
    "none": "Keep the original sound.",
}

VAD_OPTIONS = {
    "silero": "Silero",
    "whisper": "Whisper",
    "energy": "Loudness",
    "none": "Off",
}
VAD_HELP = {
    "silero": "Recommended. Detects speech by its acoustic pattern.",
    "whisper": "Follows spoken words. Set to Indonesian in the pipeline.",
    "energy": "Simple volume threshold. Fastest, least precise.",
    "none": "Keep all silence.",
}

DEFAULT_FFMPEG = shutil.which("ffmpeg") or r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the audio engine…")
def load_pipeline():
    import cleaner  # your original script, saved as cleaner.py

    return cleaner


class LiveLogHandler(logging.Handler):
    """Collects pipeline log lines and mirrors the latest ones into the page."""

    def __init__(self, placeholder):
        super().__init__(level=logging.INFO)
        self.placeholder = placeholder
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if set(msg.strip()) == {"="}:
            return
        self.lines.append(msg.lstrip("> ").strip())
        self.placeholder.code("\n".join(self.lines[-8:]), language=None)

    def stat(self, label: str):
        text = "\n".join(self.lines)
        match = re.search(rf"{label}\s*:\s*([\d.]+)", text)
        return float(match.group(1)) if match else None


def fmt_time(seconds):
    if seconds is None:
        return "–"
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:04.1f}" if m else f"{s:.1f} s"


def run_pipeline(upload, noise, vad, min_silence, pad, ffmpeg_path, out_audio_ext, log_slot):
    cleaner = load_pipeline()

    suffix = os.path.splitext(upload.name)[1].lower()
    stem = os.path.splitext(upload.name)[0]
    is_video = suffix in VIDEO_EXTS
    out_ext = ".mp4" if is_video else out_audio_ext

    workdir = tempfile.mkdtemp(prefix="cleaner_")
    in_path = os.path.join(workdir, f"input{suffix}")
    out_path = os.path.join(workdir, f"output{out_ext}")

    with open(in_path, "wb") as f:
        f.write(upload.getbuffer())

    handler = LiveLogHandler(log_slot)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        strategy = cleaner.StrategyFactory.create_noise_reduction(noise)
        vad_strategy = cleaner.StrategyFactory.create_vad(vad, min_silence_gap=min_silence, pad=pad)
        media = cleaner.FFmpegMediaManager(ffmpeg_path)
        cleaner.AudioProcessor(strategy, media, vad_strategy).execute(in_path, out_path)

        def read(path, mode="rb", **kw):
            if not os.path.exists(path):
                return None
            with open(path, mode, **kw) as f:
                return f.read()

        return {
            "name": f"{stem}_clean{out_ext}",
            "is_video": is_video,
            "mime": "video/mp4" if is_video else ("audio/wav" if out_ext == ".wav" else "audio/mp4"),
            "bytes": read(out_path),
            "plot": read(out_path + ".plot.png"),
            "manifest": read(out_path + ".manifest.txt", "r", encoding="utf-8"),
            "original": handler.stat("Original Duration"),
            "final": handler.stat("Final Duration"),
            "removed": handler.stat("Total Reduction"),
        }
    finally:
        root.removeHandler(handler)
        shutil.rmtree(workdir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 1. File
# --------------------------------------------------------------------------- #
st.markdown("### Your file")
upload = st.file_uploader(
    "Drop an audio or video file",
    type=[e.lstrip(".") for e in sorted(VIDEO_EXTS | AUDIO_EXTS)],
    label_visibility="collapsed",
)

is_video = False
if upload:
    ext = os.path.splitext(upload.name)[1].lower()
    is_video = ext in VIDEO_EXTS
    size_mb = upload.size / 1_048_576
    st.markdown(
        f'<p class="small-note">{upload.name} · {size_mb:.1f} MB</p>', unsafe_allow_html=True
    )
    (st.video if is_video else st.audio)(upload)

# --------------------------------------------------------------------------- #
# 2. Options
# --------------------------------------------------------------------------- #
st.markdown("### What to do")
with st.container(border=True):
    left, right = st.columns(2, gap="large")
    with left:
        noise = st.radio(
            "Reduce noise",
            list(NOISE_OPTIONS),
            format_func=NOISE_OPTIONS.get,
            key="noise",
        )
        st.caption(NOISE_HELP[noise])
    with right:
        vad = st.radio(
            "Remove silence",
            list(VAD_OPTIONS),
            format_func=VAD_OPTIONS.get,
            key="vad",
        )
        st.caption(VAD_HELP[vad])

    if vad in ("silero", "whisper"):
        st.divider()
        c1, c2 = st.columns(2, gap="large")
        min_silence = c1.slider(
            "Shortest pause to cut",
            0.1, 2.0, 0.5, 0.05, format="%.2f s",
            help="Pauses shorter than this stay in.",
        )
        pad = c2.slider(
            "Breathing room",
            0.0, 0.6, 0.2, 0.05, format="%.2f s",
            help="Extra audio kept around each phrase so words don't sound clipped.",
        )
    else:
        min_silence, pad = 0.5, 0.2

    if not is_video:
        st.divider()
        out_choice = st.radio(
            "Save as",
            ["WAV (lossless)", "M4A (smaller)"],
            horizontal=True,
        )
        out_audio_ext = ".wav" if out_choice.startswith("WAV") else ".m4a"
    else:
        out_audio_ext = ".wav"
        st.divider()
        st.caption("Video is saved as MP4. The picture is copied as-is, only the sound changes.")

with st.expander("Advanced"):
    ffmpeg_path = st.text_input("FFmpeg location", value=DEFAULT_FFMPEG)

# --------------------------------------------------------------------------- #
# 3. Run
# --------------------------------------------------------------------------- #
nothing_to_do = noise == "none" and vad == "none"
if upload and nothing_to_do:
    st.info("Pick a noise reduction or silence removal method to get started.")

go = st.button(
    "Clean audio",
    type="primary",
    use_container_width=True,
    disabled=(upload is None or nothing_to_do),
)

if go:
    with st.status("Cleaning your file…", expanded=True) as status:
        log_slot = st.empty()
        try:
            st.session_state["result"] = run_pipeline(
                upload, noise, vad, min_silence, pad, ffmpeg_path, out_audio_ext, log_slot
            )
            status.update(label="Done", state="complete", expanded=False)
        except ModuleNotFoundError as e:
            st.session_state.pop("result", None)
            status.update(label="Something is missing", state="error")
            st.error(f"Missing module: `{e.name}`. Install it with `pip install {e.name}`.")
        except Exception as e:
            st.session_state.pop("result", None)
            status.update(label="Couldn't finish", state="error")
            st.error(f"{e}")
            with st.expander("Technical details"):
                st.code(traceback.format_exc(), language=None)

# --------------------------------------------------------------------------- #
# 4. Result
# --------------------------------------------------------------------------- #
result = st.session_state.get("result")
if result and result.get("bytes"):
    st.markdown("### Result")

    if result["original"] is not None and result["final"] is not None:
        m1, m2, m3 = st.columns(3)
        m1.metric("Original", fmt_time(result["original"]))
        m2.metric("Cleaned", fmt_time(result["final"]))
        m3.metric("Removed", fmt_time(result["removed"]))

    if result["is_video"]:
        st.video(result["bytes"])
    else:
        st.audio(result["bytes"], format=result["mime"])

    st.download_button(
        "Download cleaned file",
        data=result["bytes"],
        file_name=result["name"],
        mime=result["mime"],
        use_container_width=True,
    )

    if result.get("plot") or result.get("manifest"):
        tab_plot, tab_text = st.tabs(["Speech timeline", "Detected speech"])
        with tab_plot:
            if result.get("plot"):
                st.image(result["plot"], use_container_width=True)
                st.caption("Green sections were kept. Dark sections were cut.")
        with tab_text:
            if result.get("manifest"):
                st.text_area(
                    "Detected speech",
                    result["manifest"],
                    height=280,
                    label_visibility="collapsed",
                )