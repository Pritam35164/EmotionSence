# app.py
import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
import joblib
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import librosa
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
MODEL_FILE = "generalized_emotion_model.joblib"
SESSION_LOG = "session_log.csv"
RECORD_SECONDS_DEFAULT = 3
SAMPLE_RATE = 22050  # Must match training SR

st.set_page_config(page_title="🎙️ Live Voice Emotion (Record/Upload)", layout="centered")

# ----------------------------
# Load model bundle
# ----------------------------
@st.cache_resource
def load_model(bundle_path=MODEL_FILE):
    if not os.path.exists(bundle_path):
        return None, None
    bundle = joblib.load(bundle_path)
    model = bundle.get("model") if isinstance(bundle, dict) else bundle
    label_encoder = bundle.get("label_encoder") if isinstance(bundle, dict) else None
    return model, label_encoder

model, label_encoder = load_model()
if model is None or label_encoder is None:
    st.error(f"Model bundle not found or invalid. Expected '{MODEL_FILE}' containing {{'model','label_encoder'}}.")
    st.stop()

# ----------------------------
# Feature extraction (same as training)
# ----------------------------
def extract_features(file_path, sr=SAMPLE_RATE, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y)

    # Core MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Delta and Delta-Delta
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    delta_mean = np.mean(delta_mfcc, axis=1)
    delta2_mean = np.mean(delta2_mfcc, axis=1)

    # Combine all
    features = np.hstack([mfcc_mean, delta_mean, delta2_mean])
    return features.reshape(1, -1)

# ----------------------------
# Recording helper
# ----------------------------
def record_audio(duration=RECORD_SECONDS_DEFAULT, sr=SAMPLE_RATE):
    st.info(f"Recording for {duration} seconds... (make sure your microphone is enabled)")
    try:
        recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
    except Exception as e:
        st.error("Recording failed: " + str(e))
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, recording.flatten(), sr)
    return tmp.name

# ----------------------------
# Session-state initialization
# ----------------------------
if "history_df" not in st.session_state:
    if os.path.exists(SESSION_LOG):
        try:
            st.session_state["history_df"] = pd.read_csv(SESSION_LOG)
        except Exception:
            st.session_state["history_df"] = pd.DataFrame()
    else:
        st.session_state["history_df"] = pd.DataFrame()

if "current_session" not in st.session_state:
    st.session_state["current_session"] = []

# ----------------------------
# UI Layout
# ----------------------------
st.title("🎙️ Human Emotion Detection — Record or Upload")
st.markdown("""
You can either **record a short audio clip** using your microphone or **upload a .wav file**. 
The app extracts features, predicts emotion using the trained model, shows probability bars,
and logs each session to `session_log.csv`.
""")

# ----------------------------
# Input: Record or Upload
# ----------------------------
st.subheader("Input Audio")
col1, col2 = st.columns(2)

input_audio_path = None

with col1:
    duration = st.number_input("Record duration (seconds)", min_value=1, max_value=10, value=RECORD_SECONDS_DEFAULT, step=1)
    if st.button("🔴 Record"):
        audio_path = record_audio(duration=duration, sr=SAMPLE_RATE)
        if audio_path:
            st.success("Recording saved.")
            st.audio(audio_path)
            input_audio_path = audio_path

with col2:
    uploaded_file = st.file_uploader("Or upload a .wav file", type=["wav"])
    if uploaded_file is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(uploaded_file.read())
        tmp.flush()
        st.success("File uploaded successfully.")
        st.audio(tmp.name)
        input_audio_path = tmp.name

# ----------------------------
# Predict if we have an audio path
# ----------------------------
def predict_and_log(audio_path):
    try:
        feats = extract_features(audio_path)
        probs = model.predict_proba(feats)[0]
        labels = label_encoder.classes_
        top_idx = np.argmax(probs)
        top_label = label_encoder.inverse_transform([top_idx])[0]

        # Show prediction
        st.markdown(f"### Predicted emotion: **{top_label}**  —  confidence: **{probs[top_idx]:.2f}**")

        # Probability bar chart
        prob_df = pd.DataFrame({"emotion": labels, "probability": probs}).sort_values("probability", ascending=False)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(prob_df["emotion"][::-1], prob_df["probability"][::-1])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Emotion probabilities")
        plt.tight_layout()
        st.pyplot(fig)

        # Log session
        timestamp = datetime.now().isoformat()
        log_row = {"timestamp": timestamp, "predicted": top_label}
        for i, lab in enumerate(labels):
            log_row[f"prob_{lab}"] = float(probs[i])

        # Update CSV
        if os.path.exists(SESSION_LOG):
            try:
                existing = pd.read_csv(SESSION_LOG)
                existing = pd.concat([existing, pd.DataFrame([log_row])], ignore_index=True)
                existing.to_csv(SESSION_LOG, index=False)
            except Exception:
                pd.DataFrame([log_row]).to_csv(SESSION_LOG, index=False)
        else:
            pd.DataFrame([log_row]).to_csv(SESSION_LOG, index=False)

        # Update session state
        st.session_state["history_df"] = pd.concat(
            [st.session_state.get("history_df", pd.DataFrame()), pd.DataFrame([log_row])],
            ignore_index=True
        )
        st.session_state["current_session"].append(log_row)

    except Exception as e:
        st.error("Prediction failed: " + str(e))
    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass

if input_audio_path:
    predict_and_log(input_audio_path)

# ----------------------------
# Trend graph
# ----------------------------
st.markdown("---")
st.subheader("📈 Emotion trend (session history)")

def plot_trend_from_df(df, title="Emotion counts (recent)"):
    if df is None or df.empty:
        st.info("No session history to plot yet. Record or upload some samples.")
        return

    df = df.copy()
    df["predicted"] = df["predicted"].astype(str)

    N = 50  # last N sessions
    last = df.tail(N)

    emotions = label_encoder.classes_
    counts_over_time = pd.DataFrame(0, index=range(len(last)), columns=emotions)

    for i, emo in enumerate(last["predicted"]):
        if emo in emotions:
            counts_over_time.loc[i, emo] = 1

    cum_counts = counts_over_time.cumsum()

    fig, ax = plt.subplots(figsize=(8, 4))
    for emo in emotions:
        ax.plot(cum_counts.index, cum_counts[emo], label=emo, marker="o")
    ax.set_xlabel("Recent session index (most recent on right)")
    ax.set_ylabel("Cumulative count")
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    st.pyplot(fig)

plot_trend_from_df(st.session_state.get("history_df", pd.DataFrame()), 
                   title="Cumulative emotion counts (last 50 sessions)")

# ----------------------------
# Recent sessions table
# ----------------------------
st.markdown("---")
st.write("Recent sessions (last 10):")
if not st.session_state.get("history_df", pd.DataFrame()).empty:
    st.dataframe(
        st.session_state["history_df"].tail(10).sort_values("timestamp", ascending=False)
    )
else:
    st.info("No sessions logged yet.")

# ----------------------------
# Notes
# ----------------------------
st.markdown("""
---
**Notes**
- You can **record** or **upload a `.wav` file`**. 
- The app extracts features, predicts emotion, shows probability bars, and logs sessions.
- Recommended installation:
    pip install streamlit sounddevice soundfile librosa numpy pandas scikit-learn joblib matplotlib
- Run locally:
    streamlit run app.py
""")
