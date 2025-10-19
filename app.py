# streamlit_app.py

import os
import io
import glob
import textwrap
from pathlib import Path
from datetime import datetime
import zipfile
import tempfile
import shutil
# --- Hard-disable GPU/XLA & stabilize TF on headless hosts ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # avoids rare segfaults on some CPUs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import streamlit as st
from streamlit_drawable_canvas import st_canvas  # optional sketch pad

# ---- Streamlit compatibility helper (old/new APIs) ----
import inspect as _inspect
_IMAGE_SIG = _inspect.signature(st.image)
_HAS_USE_CONTAINER = "use_container_width" in _IMAGE_SIG.parameters
_HAS_USE_COLUMN = "use_column_width" in _IMAGE_SIG.parameters

def show_image(img, caption=None):
    """Call st.image with the right keyword based on installed Streamlit."""
    if _HAS_USE_CONTAINER:
        return st.image(img, caption=caption, use_container_width=True)
    if _HAS_USE_COLUMN:
        return st.image(img, caption=caption, use_column_width=True)
    return st.image(img, caption=caption)

# Try headless OpenCV first; fall back to PIL if not available
try:
    import cv2  # headless wheel should satisfy this
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

# Optional folder picker (nice UX). Falls back to text input if not installed.
try:
    from streamlit_extras.st_directory_picker import st_directory_picker
    HAS_DIR_PICKER = True
except Exception:
    HAS_DIR_PICKER = False

# CV/TF (Keras 3 safe load for Lambda layers)
import tensorflow as tf
# Make sure TensorFlow never tries to see GPUs
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass
import plotly.express as px
try:
    from keras import config as keras_config  # Keras 3
    keras_config.enable_unsafe_deserialization()
except Exception:
    pass

# Lightweight local CTC + config tools to avoid importing mltu (which pulls cv2/libGL)
import yaml
from types import SimpleNamespace

# -----------------------------
# App constants / defaults
# -----------------------------
# Keep model next to this file for Streamlit Cloud reliability
APP_DIR = Path(__file__).parent
DEFAULT_H5_PATH = str(APP_DIR / "model.h5")
DEFAULT_CFG_PATH = str(APP_DIR / "configs.yaml")

APP_TITLE = "🕵️‍♀️ CAPTCHA  OCR — Visual CAPTCHA Solver"
APP_TAGLINE = "🔀   End-to-end CAPTCHA recognition with CNN + BiLSTM + CTC — demo, batch testing, and diagnostics"

# -----------------------------
# Minimal CSS for modern look
# -----------------------------
MODERN_CSS = """
<style>
/* Compact, modern typographic rhythm */
:root { --radius: 16px; }
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0f172a,#111827); }
[data-testid="stSidebar"] * { color: #e5e7eb !important; }
.sidebar-head { font-weight: 700; font-size: 1.0rem; margin: 0.5rem 0 0.25rem; color: #a5b4fc; }
.kpi { padding: 14px 16px; border-radius: var(--radius); border: 1px solid rgba(148,163,184,.25); background: rgba(2,6,23,.65); color: #e5e7eb; }
.kpi .label { font-size: .8rem; opacity: .8; }
.kpi .value { font-size: 1.4rem; font-weight: 700; }
.card { border: 1px solid rgba(148,163,184,.25); border-radius: var(--radius); padding: 16px; }
.figure-img { border-radius: 12px; border: 1px solid rgba(148,163,184,.2); }
footer { visibility: hidden; }
/* Boost visibility of canvas toolbar icons (undo/redo/delete) */
.stCanvasToolbar button, .stCanvasToolbar svg { filter: drop-shadow(0 0 2px rgba(0,0,0,.6)); transform: scale(1.1); }
.stCanvasToolbar { gap: 8px; }
/* Bigger fonts and KPIs */
.kpi.big { padding: 18px 20px; }
.kpi.big .label { font-size: 0.95rem; }
.kpi.big .value { font-size: 1.8rem; }
h1 { font-size: 2.1rem; }
h2 { font-size: 1.5rem; }
.stMarkdown, .stDataFrame { font-size: 1.05rem; }
.element-container p, .element-container code { font-size: 1.0rem; }
/* Wider KPI look */
.kpi.full { width: 100%; }
.muted { color: #94a3b8; font-size: 0.9rem; }
.label-chip { display:inline-block; padding:6px 10px; border-radius:10px; background:#0a2e1f; color:#34d399; font-weight:700; letter-spacing:0.5px; }
.pred-chip { display:inline-block; padding:6px 10px; border-radius:10px; background:#0a1f2e; color:#60a5fa; font-weight:700; letter-spacing:0.5px; }
.cer-chip { display:inline-block; padding:4px 8px; border-radius:8px; background:#111827; color:#e5e7eb; }
/* Distinct hero KPI row */
.kpi.hero {
  background: linear-gradient(135deg, rgba(59,130,246,.18), rgba(16,185,129,.16));
  border-color: rgba(148,163,184,.35);
  box-shadow: 0 6px 18px rgba(0,0,0,.25);
}
.kpi.section { background: rgba(2,6,23,.65); }
/* Example table (aligned rows) */
.ex-table { width: 100%; border-collapse: separate; border-spacing: 0 10px; }
.ex-table td {
  padding: 10px 12px;
  border: 1px solid rgba(148,163,184,.25);
  background: rgba(2,6,23,.65);
  vertical-align: middle;
}
.ex-col-gt { width: 22%; }
.ex-col-pred { width: 22%; }
.ex-col-cer { width: 12%; }
.ex-col-path { width: 44%; color:#94a3b8; font-size:.9rem; }
/* Larger tabs */
button[role="tab"] { font-size: 1.05rem !important; padding: 0.6rem 1rem !important; }
button[role="tab"][aria-selected="true"] { font-weight: 700 !important; }
</style>
"""

# -----------------------------
# Minimal helpers (replace mltu usage)
# -----------------------------
def _load_yaml_cfg(path: str) -> SimpleNamespace:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Ensure required keys exist with sensible defaults
    data.setdefault("vocab", "0123456789abcdefghijklmnopqrstuvwxyz")
    data.setdefault("height", 50)
    data.setdefault("width", 200)
    # Support either string or list vocab
    vocab = data.get("vocab")
    if isinstance(vocab, list):
        vocab = "".join(vocab)
    data["vocab"] = vocab
    return SimpleNamespace(**data)

def _levenshtein(a: str, b: str) -> int:
    # classic DP edit distance
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost     # substitution
            )
            prev = tmp
    return dp[m]

def get_cer(pred: str, gt: str) -> float:
    gt = str(gt)
    pred = str(pred)
    if len(gt) == 0:
        return float(len(pred))
    return _levenshtein(pred, gt) / float(len(gt))

def ctc_decoder(logits: np.ndarray, vocab: str):
    """
    Greedy CTC decode.
    Assumes `logits` shape (N, T, V) with blank = last index (len(vocab)).
    Returns a list of strings of length N.
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected (N,T,V) logits, got shape {logits.shape}")
    N, T, V = logits.shape
    blank = V - 1  # convention: blank at last index
    # argmax over classes
    labels = np.argmax(logits, axis=-1)  # (N, T)
    results = []
    for n in range(N):
        prev = None
        chars = []
        for t in range(T):
            idx = int(labels[n, t])
            if idx == blank:
                prev = idx
                continue
            if prev == idx:
                # collapse repeats
                prev = idx
                continue
            prev = idx
            if 0 <= idx < len(vocab):
                chars.append(vocab[idx])
        results.append("".join(chars))
    return results

# -----------------------------
# Utility functions
# -----------------------------

# Session keys for UI controls
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "last_run" not in st.session_state:
    st.session_state.last_run = pd.DataFrame(columns=["image_path","label","pred","exact_match","cer"])
if "zip_key" not in st.session_state:
    st.session_state.zip_key = 0
if "uploaded_zip_dir" not in st.session_state:
    st.session_state.uploaded_zip_dir = None
# -----------------------------
# Sample CAPTCHA state
# -----------------------------
if "label_map" not in st.session_state:
    st.session_state.label_map = {}
if "sample_dir" not in st.session_state:
    st.session_state.sample_dir = None
if "sample_files" not in st.session_state:
    st.session_state.sample_files = []
# -----------------------------
# Tab state flags (for controlling dashboard tab)
# -----------------------------
if "show_dashboard" not in st.session_state:
    st.session_state.show_dashboard = False
@st.cache_resource(show_spinner=False)
def load_model_and_cfg(h5_path: str, cfg_path: str):
    """Load model & config on-demand. If configs.yaml is missing/empty, create a minimal one.
    Avoid scanning large dataset folders to prevent long startup times on Streamlit Cloud.
    """
    # Ensure minimal config exists without scanning Datasets/
    if not os.path.exists(cfg_path) or os.path.getsize(cfg_path) == 0:
        vocab = "0123456789abcdefghijklmnopqrstuvwxyz"
        minimal_cfg = {
            "model_path": str(Path(h5_path).parent),
            "vocab": vocab,
            "height": 50,
            "width": 200,
            "max_text_length": 5,
            "batch_size": 64,
            "learning_rate": 0.001,
            "train_epochs": 70,
            "train_workers": 4,
        }
        os.makedirs(Path(cfg_path).parent, exist_ok=True)
        import yaml
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(minimal_cfg, f, sort_keys=False, allow_unicode=True)

    cfg = _load_yaml_cfg(cfg_path)
    # Load TF model (works with Keras 3 + Lambda layers)
    model = tf.keras.models.load_model(h5_path, compile=False, safe_mode=False)
    return model, cfg

def preprocess_bgr(img_bgr: np.ndarray, W=200, H=50) -> np.ndarray:
    """Resize only. DO NOT /255 here (first Lambda layer handles it).
    Uses OpenCV if available; otherwise falls back to PIL to avoid libGL issues on headless hosts.
    """
    if 'HAS_CV2' in globals() and HAS_CV2:
        img = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    else:
        # PIL expects RGB; convert BGR->RGB, resize, then back to BGR
        pil_im = Image.fromarray(img_bgr[:, :, ::-1])
        pil_im = pil_im.resize((W, H), Image.BILINEAR)
        img = np.array(pil_im)[:, :, ::-1].astype(np.float32)
    x = np.expand_dims(img, 0)  # (1, H, W, 3)
    return x

def predict_one(model, cfg, image_bgr: np.ndarray) -> str:
    x = preprocess_bgr(image_bgr, 200, 50)
    logits = model.predict(x, verbose=0)  # (1, T, |V|)
    # Some exports yield (N, |V|, T)
    if (logits.ndim == 3 and logits.shape[1] == len(cfg.vocab) and logits.shape[2] < logits.shape[1]):
        logits = np.transpose(logits, (0, 2, 1))
    return ctc_decoder(logits, cfg.vocab)[0]

from PIL import Image
import io
import numpy as np
from pathlib import Path

def _pil_to_bgr(pil_im: Image.Image) -> np.ndarray:
    rgb = np.array(pil_im.convert("RGB"))
    return rgb[:, :, ::-1]  # RGB -> BGR

def read_file_as_bgr(file) -> np.ndarray:
    """
    Reads an image from path / UploadedFile / bytes and returns a BGR np.ndarray.
    Works on headless servers even if OpenCV GUI is unavailable.
    """
    if HAS_CV2:
        if isinstance(file, (str, Path)):
            img = cv2.imread(str(file).replace("\\", "/"))
        else:
            try:
                data = file.getvalue()
            except Exception:
                data = file.read()
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not read image.")
        return img

    # Fallback: PIL only (headless-safe)
    if isinstance(file, (str, Path)):
        im = Image.open(str(file))
    else:
        try:
            data = file.getvalue()
        except Exception:
            data = file.read()
        im = Image.open(io.BytesIO(data))
    return _pil_to_bgr(im)

def stem(path_or_label: str) -> str:
    base = os.path.basename(str(path_or_label))
    name, _ = os.path.splitext(base)
    return name

def gallery(df: pd.DataFrame, n=8):
    rows_html = []
    for _, row in df.head(n).iterrows():
        rows_html.append(
            "<tr>"
            f"<td class='ex-col-gt'><strong>GT:</strong> <span class='label-chip'>{row['label']}</span></td>"
            f"<td class='ex-col-pred'><strong>Pred:</strong> <span class='pred-chip'>{row['pred']}</span></td>"
            f"<td class='ex-col-cer'><span class='cer-chip'>CER: {row['cer']:.3f}</span></td>"
            f"<td class='ex-col-path'>{row['image_path']}</td>"
            "</tr>"
        )
    html = "<table class='ex-table'>" + "".join(rows_html) + "</table>"
    st.markdown(html, unsafe_allow_html=True)

# -----------------------------
# Sample CAPTCHA generation
# -----------------------------
def generate_sample_captchas_from_folder(folder="imgs"):
    """
    Load sample CAPTCHA images from a local folder instead of generating them.
    Each filename is treated as its true label (filename stem).
    """
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        st.error(f"Sample folder '{folder}' not found.")
        st.session_state.sample_files = []
        return []

    # Clear any previous sample session
    if st.session_state.sample_dir:
        try:
            shutil.rmtree(st.session_state.sample_dir, ignore_errors=True)
        except Exception:
            pass
    st.session_state.sample_dir = tempfile.mkdtemp(prefix="captcha_samples_")

    image_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    out = []
    for img_path in folder_path.iterdir():
        if img_path.suffix.lower() in image_exts:
            label = stem(img_path.name)
            temp_path = os.path.join(st.session_state.sample_dir, img_path.name)
            shutil.copy(img_path, temp_path)
            out.append((temp_path, label))

    st.session_state.sample_files = out
    return out

# -----------------------------
# Model paths (inline, no sidebar)
# -----------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🔎", layout="wide")
st.markdown(MODERN_CSS, unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
left, right = st.columns([1.1, 1])
with left:
    st.title(APP_TITLE)
    st.write(APP_TAGLINE)
    st.markdown(
    "- Upload a few images or a whole folder/ZIP\n"
    "- Get accuracy, CER, and an interactive confusion heatmap\n"
    "- Review correct vs. misreads with per-sample details",
)
with right:
    st.empty()

st.divider()

# -----------------------------
# Tabs: Upload (always) + Dashboard (after first run)
# -----------------------------

if not st.session_state.show_dashboard:
    tab_upload = st.tabs(["Upload"])[0]
    tab_dash = None
else:
    tab_upload, tab_dash = st.tabs(["📂 Upload", "📊 Dashboard"])

# If we've just finished a run, force-switch focus to the Dashboard tab.
if st.session_state.get("focus_dashboard", False):
    st.markdown(
        """
        <script>
        (function () {
          // Avoid running multiple times per render
          if (window.__dashFocusRan) return;
          window.__dashFocusRan = true;

          function findTabs() {
            // Streamlit tabs may render as role="tab" or via BaseWeb
            const roleTabs = Array.from(document.querySelectorAll('button[role="tab"]'));
            const basewebTabs = Array.from(document.querySelectorAll('[data-baseweb="tab"] button, [data-baseweb="tabs"] button'));
            return roleTabs.concat(basewebTabs);
          }

          function clickDashboard() {
            const tabs = findTabs();
            if (!tabs.length) return false;
            const target = tabs.find((b) => {
              const t = (b.innerText || b.textContent || "").toLowerCase().replace(/\\s+/g,' ').trim();
              return t.includes('dashboard'); // matches "📊 Dashboard" too
            });
            if (target) {
              target.click();
              return true;
            }
            return false;
          }

          // Try immediately, then repeatedly for a while, and also on DOM mutations
          let attempts = 0;
          const maxAttempts = 150; // ~15 seconds
          const timer = setInterval(() => {
            attempts += 1;
            if (clickDashboard() || attempts >= maxAttempts) {
              clearInterval(timer);
            }
          }, 100);

          const mo = new MutationObserver((_muts, obs) => {
            if (clickDashboard()) {
              obs.disconnect();
            }
          });
          mo.observe(document.body, { childList: true, subtree: true });

          // Also try on next animation frames for early mount cases
          let rafTries = 0;
          (function rafTry(){
            if (clickDashboard() || rafTries > 30) return;
            rafTries += 1;
            requestAnimationFrame(rafTry);
          })();
        })();
        </script>
        """,
        unsafe_allow_html=True
    )
    st.session_state["focus_dashboard"] = False

# -----------------------------
# Data Input (in body, not sidebar)
# -----------------------------
with tab_upload:
    st.subheader("Add a Captch. Let's See How Solvable It Is!")
    upload = st.file_uploader(
        "Upload CAPTCHA images (png/jpg)",
        type=["png","jpg","jpeg","bmp","webp"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )
    zip_file = st.file_uploader(
        "Or upload a folder (.zip)",
        type=["zip"],
        key=f"zip_{st.session_state.zip_key}"
    )
    if zip_file is not None:
        # Clear previous extracted folder if any
        if st.session_state.uploaded_zip_dir:
            try:
                shutil.rmtree(st.session_state.uploaded_zip_dir, ignore_errors=True)
            except Exception:
                pass
            st.session_state.uploaded_zip_dir = None
        # Extract the uploaded ZIP to a temp directory
        tmpdir = tempfile.mkdtemp(prefix="captcha_zip_")
        try:
            with zipfile.ZipFile(zip_file) as zf:
                zf.extractall(tmpdir)
            st.session_state.uploaded_zip_dir = tmpdir
            st.caption(f"Folder uploaded and extracted to a temporary directory.")
        except Exception as e:
            st.error(f"Could not extract ZIP: {e}")
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass
            st.session_state.uploaded_zip_dir = None

    # --- Sample CAPTCHA controls ---
    st.markdown("###### No CAPTCHA images?")
    c_s1, c_s2, c_s3 = st.columns([1,1,4])
    with c_s1:
        if st.button("🎁 Load sample set", use_container_width=True):
            generate_sample_captchas_from_folder("imgs")
            if st.session_state.sample_files:
                st.success("Sample images loaded from imgs folder.")
    with c_s2:
        if st.button("🗑️ Clear samples", use_container_width=True):
            if st.session_state.sample_dir:
                try: shutil.rmtree(st.session_state.sample_dir, ignore_errors=True)
                except Exception: pass
            st.session_state.sample_dir = None
            st.session_state.sample_files = []
            st.rerun()
    if st.session_state.sample_files:
        prev_cols = st.columns(min(6, len(st.session_state.sample_files)))
        for i, (p, lbl) in enumerate(st.session_state.sample_files[:6]):
            with prev_cols[i % len(prev_cols)]:
                show_image(p, caption=lbl)

    col_dir, col_gt, col_canvas = st.columns([1.2, 1, 1])
    with col_dir:
        if 'last_dir' not in st.session_state:
            st.session_state.last_dir = ""
        if 'last_glob' not in st.session_state:
            st.session_state.last_glob = ""

        if HAS_DIR_PICKER:
            selected_dir = st_directory_picker("Or select a folder")

    with col_gt:
        add_ground_truth = st.checkbox("My file/s name is the captcha true label/s", value=True, help="If checked, labels are derived from filename stems or from a CSV mapping.")

    # Manual labeling UI for uploaded files when filename isn't GT
    if not add_ground_truth and upload:
        st.markdown("##### Enter labels for uploaded files")
        for f in upload:
            cols = st.columns([1, 2])
            with cols[0]:
                show_image(f, caption=f.name)
            with cols[1]:
                key = f"label_{st.session_state.uploader_key}_{f.name}"
                default_val = st.session_state.label_map.get(f.name, "")
                label_val = st.text_input("Label", value=default_val, placeholder="e.g., 4n3f7", key=key)
                # persist to session map
                st.session_state.label_map[f.name] = label_val.strip()
        st.caption("Tip: leave blank to exclude from accuracy/CER (still shows prediction).")

    # Control row: Run, Clear files, Reset session
    col_run, col_clear, col_reset = st.columns([1,1,1])
    with col_run:
        run_btn = st.button("▶ Run Predictions", use_container_width=True)
    with col_clear:
        if st.button("🧹 Clear all files", use_container_width=True):
            # Bump uploader keys to reset widgets
            st.session_state.uploader_key += 1
            st.session_state.zip_key += 1
            # Clear directory picker/text
            if 'selected_dir' in locals():
                selected_dir = ""
            # Remove any previously extracted ZIP contents
            if st.session_state.get("uploaded_zip_dir"):
                try:
                    shutil.rmtree(st.session_state.uploaded_zip_dir, ignore_errors=True)
                except Exception:
                    pass
                st.session_state.uploaded_zip_dir = None
            # Clear any typed labels
            st.session_state.label_map = {}
            # Clear sample captchas
            if st.session_state.get("sample_dir"):
                try: shutil.rmtree(st.session_state.sample_dir, ignore_errors=True)
                except Exception: pass
            st.session_state.sample_dir = None
            st.session_state.sample_files = []
            st.rerun()
    with col_reset:
        if st.button("🔁 Reset session", use_container_width=True):
            # Clean up temporary folders first
            try:
                if st.session_state.get("uploaded_zip_dir"):
                    shutil.rmtree(st.session_state.uploaded_zip_dir, ignore_errors=True)
            except Exception:
                pass
            try:
                if st.session_state.get("sample_dir"):
                    shutil.rmtree(st.session_state.sample_dir, ignore_errors=True)
            except Exception:
                pass
            # Now clear *all* session state (history, last_run, flags, labels, etc.)
            st.session_state.clear()
            st.rerun()

# -----------------------------
# Prediction run
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["image_path","label","pred","exact_match","cer"])

# Collect inputs (model is loaded lazily on Run)
files = []
if upload:
    files.extend(upload)
exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.webp")
if 'selected_dir' in locals() and selected_dir:
    for ext in exts:
        files.extend(glob.glob(os.path.join(selected_dir, ext)))
if st.session_state.get("uploaded_zip_dir"):
    for ext in exts:
        files.extend(glob.glob(os.path.join(st.session_state.uploaded_zip_dir, "**", ext), recursive=True))
if st.session_state.sample_files:
    files.extend([p for (p, _lbl) in st.session_state.sample_files])

if run_btn and len(files) == 0:
    st.warning("Please upload images, select a folder/ZIP, or click **Load sample set** above.")
elif run_btn and len(files) > 0:
    # Fail fast if model/config not found
    if not os.path.exists(DEFAULT_H5_PATH):
        st.error(f"Model not found at {DEFAULT_H5_PATH}. Place 'model.h5' next to app.py.")
        st.stop()
    if (not os.path.exists(DEFAULT_CFG_PATH)) or os.path.getsize(DEFAULT_CFG_PATH) == 0:
        st.info("configs.yaml missing — creating a minimal config…")
    with st.spinner("Loading model…"):
        model, cfg = load_model_and_cfg(DEFAULT_H5_PATH, DEFAULT_CFG_PATH)
        st.session_state["cfg"] = cfg  # keep for downstream tabs

    with st.spinner("Running predictions…"):
        rows = []
        for f in files:
            if isinstance(f, tuple):
                name, bgr = f
                img_bgr = bgr
                label = sketch_label if 'sketch_label' in locals() else ""
                path_repr = name
            else:
                img_bgr = read_file_as_bgr(f)
                path_repr = getattr(f, "name", str(f))
                sample_lookup = dict(st.session_state.sample_files) if st.session_state.sample_files else {}
                if str(path_repr) in sample_lookup:
                    label = sample_lookup[str(path_repr)]
                elif add_ground_truth:
                    label = stem(path_repr)
                else:
                    label = st.session_state.label_map.get(path_repr, "")

            pred = predict_one(model, st.session_state["cfg"], img_bgr)
            cer = get_cer(pred, label) if label else np.nan
            exact = int(pred == label) if label else np.nan

            rows.append({
                "image_path": path_repr,
                "label": label,
                "pred": pred,
                "exact_match": exact,
                "cer": cer,
            })

        batch_df = pd.DataFrame(rows)
        st.session_state.history = pd.concat([st.session_state.history, batch_df], ignore_index=True)
        st.session_state.last_run = batch_df
        st.session_state.show_dashboard = True
        st.session_state.focus_dashboard = True
        st.rerun()

#
# -----------------------------
# Dashboard
# -----------------------------
if tab_dash is not None:
    with tab_dash:

        # This-run banner
        if not st.session_state.last_run.empty:
            lr = st.session_state.last_run
            total_lr = len(lr)
            if lr["exact_match"].notna().any():
                correct_lr = int(lr["exact_match"].dropna().sum())
                incorrect_lr = int(lr["exact_match"].dropna().shape[0] - correct_lr)
            else:
                correct_lr, incorrect_lr = 0, 0
            cA, cB, cC = st.columns(3)
            with cA:
                st.markdown(f"<div class='kpi big full hero'><div class='label'>On the Current run • Number of Files</div><div class='value'>{total_lr}</div></div>", unsafe_allow_html=True)
            with cB:
                st.markdown(f"<div class='kpi big full hero'><div class='label'>On the Current run • Correct CAPTCHAs</div><div class='value'>{correct_lr}</div></div>", unsafe_allow_html=True)
            with cC:
                st.markdown(f"<div class='kpi big full hero'><div class='label'>On the Current run • Incorrect CAPTCHAs</div><div class='value'>{incorrect_lr}</div></div>", unsafe_allow_html=True)
        hist = st.session_state.history.copy()
        if hist.empty:
            st.info("No results yet. Upload images (or select a folder) and click **Run Predictions**.")
        else:
            tab_overview, tab_confusions, tab_examples = st.tabs(["Overview", "Confusions", "Examples"])
            with tab_overview:
                # KPIs
                total = len(hist)
                if hist["exact_match"].notna().any():
                    acc = float(hist["exact_match"].dropna().mean()) if len(hist.dropna(subset=["exact_match"])) else 0.0
                    avg_cer = float(hist["cer"].dropna().mean()) if len(hist.dropna(subset=["cer"])) else np.nan
                else:
                    acc = np.nan
                    avg_cer = np.nan
                k1, k2, k3 = st.columns([1,1,1])
                with k1:
                    st.markdown(f"<div class='kpi big section'><div class='label'>Total CAPTCHA's Samples</div><div class='value'>{total}</div></div>", unsafe_allow_html=True)
                with k2:
                    v = f"{acc*100:.2f}%" if not np.isnan(acc) else "—"
                    st.markdown(f"<div class='kpi big section'><div class='label'>Total Accuracy (Exact Match)</div><div class='value'>{v}</div></div>", unsafe_allow_html=True)
                with k3:
                    v = f"{avg_cer:.4f}" if not np.isnan(avg_cer) else "—"
                    st.markdown(f"<div class='kpi big section'><div class='label'>Average Error Rate</div><div class='value'>{v}</div></div>", unsafe_allow_html=True)
                def _row_style(s):
                    # color whole row by exact_match
                    try:
                        v = int(s.get("exact_match"))
                    except Exception:
                        v = None
                    if v == 1:
                        return ["background-color: rgba(16,185,129,0.10)"] * len(s)
                    elif v == 0:
                        return ["background-color: rgba(239,68,68,0.10)"] * len(s)
                    else:
                        return [""] * len(s)
                st.markdown("#### Results Table")
                try:
                    styled = hist.style.apply(_row_style, axis=1)
                    st.dataframe(styled, use_container_width=True)
                except Exception:
                    # fallback if Styler not supported
                    st.dataframe(hist, use_container_width=True)
            with tab_confusions:
                # (keep existing confusion matrix computation; show the expander content directly here)
                cfg = st.session_state.get("cfg")
                vocab = list(cfg.vocab) if cfg else list("0123456789abcdefghijklmnopqrstuvwxyz")
                if hist["exact_match"].notna().any() and hist["label"].astype(bool).any():
                    import difflib
                    from collections import Counter
                    char_to_idx = {c:i for i,c in enumerate(vocab)}
                    y_true_chars, y_pred_chars = [], []
                    subs_counter = Counter()
                    for _, row in hist.dropna(subset=["label"]).iterrows():
                        gt, pr = str(row["label"]), str(row["pred"])
                        sm = difflib.SequenceMatcher(a=gt, b=pr)
                        for tag, i1, i2, j1, j2 in sm.get_opcodes():
                            if tag == "equal":
                                for k in range(i2 - i1):
                                    y_true_chars.append(gt[i1 + k]); y_pred_chars.append(pr[j1 + k])
                            elif tag == "replace":
                                for k in range(max(i2 - i1, j2 - j1)):
                                    g = gt[i1 + k] if i1 + k < i2 else None
                                    p = pr[j1 + k] if j1 + k < j2 else None
                                    if g is not None and p is not None:
                                        y_true_chars.append(g); y_pred_chars.append(p)
                                        subs_counter[(g,p)] += 1
                            elif tag == "delete":
                                pass
                            elif tag == "insert":
                                pass
                    if len(y_true_chars) > 0:
                        obs_chars = sorted(set(y_true_chars + y_pred_chars), key=lambda c: (c not in vocab, c))
                        idx = {c:i for i,c in enumerate(obs_chars)}
                        cm = np.zeros((len(obs_chars), len(obs_chars)), dtype=int)
                        for g, p in zip(y_true_chars, y_pred_chars):
                            if g in idx and p in idx:
                                cm[idx[g], idx[p]] += 1
                        mat = cm.astype(float)
                        row_sums = mat.sum(axis=1, keepdims=True) + 1e-9
                        mat = mat / row_sums
                        fig = px.imshow(
                            mat,
                            x=obs_chars,
                            y=obs_chars,
                            aspect="auto",
                            labels=dict(x="Predicted", y="Ground truth", color="freq"),
                            origin="upper",
                        )
                        fig.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=520, title="Normalized confusion (GT rows → Pred cols)")
                        st.plotly_chart(fig, use_container_width=True)
                        if subs_counter:
                            st.write("Top character substitutions:")
                            sub_df = pd.DataFrame(
                                [{"gt": g, "pred": p, "count": c} for (g,p), c in subs_counter.most_common(20)]
                            )
                            st.dataframe(sub_df, use_container_width=True)
                    else:
                        st.caption("No character-level matches to compute confusion matrix.")
                else:
                    st.caption("Provide ground-truth (filename stems) to enable confusion analysis.")
            with tab_examples:
                st.markdown("#### ✅ Correct predictions")
                if hist["exact_match"].notna().any():
                    good = hist[hist["exact_match"] == 1].copy()
                    if good.empty:
                        st.caption("No exact matches yet.")
                    else:
                        gallery(good, n=12)
                else:
                    st.caption("Provide ground-truth (filename stems) to enable correctness checks.")
                st.markdown("#### ❌ Misreads")
                if hist["exact_match"].notna().any():
                    bad = hist[hist["exact_match"] == 0].sort_values("cer", ascending=False)
                    if bad.empty:
                        st.caption("No mistakes recorded yet.")
                    else:
                        gallery(bad, n=12)
                else:
                    st.caption("Provide ground-truth (filename stems) to enable error analysis.")
        if not st.session_state.history.empty:
            st.divider()
            with st.expander("📘 Guide: Reading this dashboard", expanded=False):
                st.markdown(
                    """
### What the metrics mean
- **Accuracy (exact match):** fraction of images where the predicted string exactly equals the ground truth.
- **Average CER:** mean character error rate (edit distance ÷ length). Lower is better.
- **Confusion heatmap:** rows = ground truth, columns = predictions. Brighter cells indicate more frequent confusions.
- **Examples:** quick visual audit of correct vs incorrect predictions.

### Notes (CV/portfolio)
- Keras `.h5` model (CNN + BiLSTM + CTC) with `Lambda(x/255)` as first layer.
- Preprocessing: resize to **200×50**; no extra normalization in-app.
- Decoding: greedy CTC using `vocab` from `configs.yaml`.
- The app supports batch evaluation and per-character diagnostics for OCR QA.
                    """,
                    unsafe_allow_html=False
                )