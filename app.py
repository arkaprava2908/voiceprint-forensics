import gc
import hashlib
import io
import os
import subprocess
import tempfile
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from speechbrain.inference.speaker import SpeakerRecognition


# ============================================================
# VoicePrint Forensics — v7
# Reference-based speaker verification + synthetic-speech analysis
# ============================================================

st.set_page_config(
    page_title="VoicePrint Forensics",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SR = 16000
MAX_UPLOAD_MB = 50
MIN_SPEAKER_SECONDS = 1.0
MIN_DETECTOR_SECONDS = 2.5

SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEEPFAKE_MODEL = "garystafford/wav2vec2-deepfake-voice-detector"

WINDOW_SECONDS = 8.0
HOP_SECONDS = 4.0


# -----------------------------
# Page style
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #111827 0%, #172554 100%);
        border: 1px solid rgba(255,255,255,.10);
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2.3rem; }
    .hero p { margin: .45rem 0 0; color: #cbd5e1; }

    .section-title {
        font-size: 1.35rem;
        font-weight: 750;
        margin: 1.2rem 0 .65rem;
    }

    .card {
        padding: 1rem 1.1rem;
        border-radius: 15px;
        border: 1px solid rgba(148,163,184,.20);
        background: rgba(15,23,42,.48);
        min-height: 130px;
    }
    .card-title { font-weight: 700; font-size: 1.05rem; }
    .muted { color: #94a3b8; font-size: .88rem; }
    .big-number { font-size: 2rem; font-weight: 800; margin-top: .35rem; }

    .result-good {
        padding: 1rem 1.1rem;
        border-radius: 14px;
        background: rgba(16,185,129,.13);
        border: 1px solid rgba(16,185,129,.35);
    }
    .result-warn {
        padding: 1rem 1.1rem;
        border-radius: 14px;
        background: rgba(245,158,11,.13);
        border: 1px solid rgba(245,158,11,.35);
    }
    .result-bad {
        padding: 1rem 1.1rem;
        border-radius: 14px;
        background: rgba(239,68,68,.13);
        border: 1px solid rgba(239,68,68,.35);
    }
    .result-neutral {
        padding: 1rem 1.1rem;
        border-radius: 14px;
        background: rgba(59,130,246,.12);
        border: 1px solid rgba(59,130,246,.30);
    }

    .small-note {
        color: #94a3b8;
        font-size: .82rem;
        line-height: 1.45;
    }

    [data-testid="stMetricValue"] { font-size: 1.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
def friendly_error(exc: Exception) -> str:
    msg = str(exc).strip()
    if not msg:
        return "An unexpected error occurred."
    if "symlink" in msg.lower() or "winerror 1314" in msg.lower():
        return "The model cache needs a filesystem permission that is unavailable. Use Python 3.11/3.12 locally or keep model downloads in the default cache."
    if "out of memory" in msg.lower() or "cuda out of memory" in msg.lower():
        return "The model needs more RAM than is currently available. Try a fresh app restart and avoid running both large models simultaneously."
    if "ffmpeg" in msg.lower():
        return "The audio decoder could not read this format. Try WAV/MP3, or install FFmpeg in the deployment environment."
    if "tokenizer" in msg.lower() and "garystafford" in msg.lower():
        return "The specialist model was loaded with an audio feature extractor, not AutoProcessor. If this message appears, the deployed app is probably still using an older file."
    if "model_type" in msg.lower() and "config.json" in msg.lower():
        return "The selected Hugging Face checkpoint is incompatible with the installed Transformers version."
    return msg[:500]


def show_technical_error(title: str, exc: Exception) -> None:
    st.error(title)
    st.caption(f"{friendly_error(exc)}")
    with st.expander("Technical details"):
        st.code(repr(exc))


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    if audio.ndim == 2:
        # soundfile can return (frames, channels) or (channels, frames).
        if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 8:
            audio = audio.mean(axis=0)
        else:
            audio = audio.mean(axis=1)
    audio = np.ravel(audio).astype(np.float32)

    if audio.size == 0:
        raise ValueError("The file contains no audio samples.")

    peak = float(np.max(np.abs(audio)))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("The recording contains no usable non-silent audio.")

    if peak > 1.0:
        audio = audio / peak

    return audio


def decode_with_ffmpeg(data: bytes) -> Tuple[np.ndarray, int]:
    """Fallback decoder for formats SoundFile cannot read (m4a/aac, etc.)."""
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "wav", "-ac", "1", "-ar", str(SR), "pipe:1",
    ]
    result = subprocess.run(
        command,
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="ignore")[-500:]
        raise ValueError(f"FFmpeg could not decode the audio. {detail}")

    audio, sr = sf.read(io.BytesIO(result.stdout), always_2d=False)
    return normalize_audio(audio), int(sr)


def load_audio(data: bytes) -> Tuple[np.ndarray, int]:
    if not data:
        raise ValueError("The uploaded file is empty.")

    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(
            f"File is too large. Maximum allowed size is {MAX_UPLOAD_MB} MB."
        )

    try:
        audio, sr = sf.read(io.BytesIO(data), always_2d=False)
        audio = normalize_audio(audio)
        sr = int(sr)
    except Exception:
        audio, sr = decode_with_ffmpeg(data)

    if sr <= 0:
        raise ValueError("Invalid sample rate in the recording.")

    if sr != SR:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=SR,
        )

    audio = normalize_audio(audio)

    if len(audio) < int(0.25 * SR):
        raise ValueError("The recording is too short to contain useful speech.")

    return np.asarray(audio, dtype=np.float32), SR


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audio_stats(audio: np.ndarray, sr: int) -> Dict[str, float]:
    rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
    frame_rms = librosa.feature.rms(y=audio)[0]
    return {
        "duration": len(audio) / sr,
        "rms": rms,
        "peak": float(np.max(np.abs(audio))),
        "centroid": float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])),
        "rolloff": float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)[0])),
        "silence": float(np.mean(frame_rms < max(1e-5, rms * 0.12))),
    }


def make_detector_windows(audio: np.ndarray) -> List[np.ndarray]:
    n = len(audio)
    window = int(WINDOW_SECONDS * SR)
    hop = int(HOP_SECONDS * SR)

    if n < int(MIN_DETECTOR_SECONDS * SR):
        return []

    if n <= window:
        return [np.pad(audio, (0, window - n), mode="constant").astype(np.float32)]

    windows = []
    for start in range(0, n, hop):
        end = start + window
        chunk = audio[start:end]
        if len(chunk) < int(MIN_DETECTOR_SECONDS * SR):
            break
        if len(chunk) < window:
            chunk = np.pad(chunk, (0, window - len(chunk)), mode="constant")
        windows.append(chunk.astype(np.float32))
        if end >= n:
            break
    return windows


# -----------------------------
# Models
# -----------------------------
def load_deepfake_detector():
    feature_extractor = AutoFeatureExtractor.from_pretrained(DEEPFAKE_MODEL)
    model = AutoModelForAudioClassification.from_pretrained(
        DEEPFAKE_MODEL,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return feature_extractor, model


def fake_index(model) -> int:
    label2id = getattr(model.config, "label2id", {}) or {}
    id2label = getattr(model.config, "id2label", {}) or {}

    for label, idx in label2id.items():
        if any(x in str(label).lower() for x in ["fake", "synthetic", "spoof"]):
            return int(idx)
    for idx, label in id2label.items():
        if any(x in str(label).lower() for x in ["fake", "synthetic", "spoof"]):
            return int(idx)
    return 1


def detector_score(audio: np.ndarray, feature_extractor, model) -> float:
    audio = audio.astype(np.float32)
    audio = audio - np.mean(audio)
    std = float(np.std(audio))
    if std > 1e-7:
        audio = audio / std

    inputs = feature_extractor(
        audio,
        sampling_rate=SR,
        return_tensors="pt",
        padding=True,
    )
    with torch.inference_mode():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
    return float(probs[fake_index(model)].cpu().item())


def run_deepfake_detection(audio: np.ndarray, feature_extractor, model, label: str) -> Dict[str, object]:
    windows = make_detector_windows(audio)
    if not windows:
        return {
            "status": "too_short",
            "scores": [],
            "mean": None,
            "median": None,
            "maximum": None,
            "verdict": "INSUFFICIENT AUDIO",
        }

    scores = []
    progress = st.progress(0, text=f"Analyzing {label}...")
    try:
        for i, window in enumerate(windows):
            scores.append(detector_score(window, feature_extractor, model))
            progress.progress((i + 1) / len(windows), text=f"{label}: window {i + 1}/{len(windows)}")
    finally:
        progress.empty()

    scores = np.asarray(scores, dtype=np.float32)
    mean_score = float(np.mean(scores))
    median_score = float(np.median(scores))
    max_score = float(np.max(scores))

    verdict = (
        "MODEL CLASSIFIES AS SYNTHETIC"
        if mean_score >= 0.50
        else "MODEL CLASSIFIES AS REAL / BELOW FAKE THRESHOLD"
    )

    return {
        "status": "ok",
        "scores": scores,
        "mean": mean_score,
        "median": median_score,
        "maximum": max_score,
        "verdict": verdict,
    }


def verify_speaker(reference_audio: np.ndarray, test_audio: np.ndarray):
    if len(reference_audio) < int(MIN_SPEAKER_SECONDS * SR):
        raise ValueError("Reference recording is too short for speaker verification.")
    if len(test_audio) < int(MIN_SPEAKER_SECONDS * SR):
        raise ValueError("Test recording is too short for speaker verification.")

    ref_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    test_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    ref_path, test_path = ref_file.name, test_file.name
    ref_file.close()
    test_file.close()

    try:
        sf.write(ref_path, reference_audio, SR)
        sf.write(test_path, test_audio, SR)

        verifier = SpeakerRecognition.from_hparams(
            source=SPEAKER_MODEL,
            savedir=None,
        )
        score, _ = verifier.verify_files(ref_path, test_path)
        score = float(score.squeeze().detach().cpu().item())

        if score >= 0.50:
            verdict = "LIKELY SAME SPEAKER"
        elif score < 0.25:
            verdict = "LIKELY DIFFERENT SPEAKERS"
        else:
            verdict = "INCONCLUSIVE — SCORE IN OVERLAP RANGE"

        del verifier
        gc.collect()
        return score, verdict
    finally:
        for path in [ref_path, test_path]:
            try:
                os.remove(path)
            except OSError:
                pass


def acoustic_similarity(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    def vec(x):
        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20)
        contrast = librosa.feature.spectral_contrast(y=x, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(x)
        rms = librosa.feature.rms(y=x)
        pieces = []
        for f in [mfcc, contrast, zcr, rms]:
            pieces.extend(np.mean(f, axis=1))
            pieces.extend(np.std(f, axis=1))
        return np.asarray(pieces, dtype=np.float32)

    x, y = vec(a), vec(b)
    denom = (np.linalg.norm(x) * np.linalg.norm(y)) + 1e-12
    return float(np.dot(x, y) / denom)


def compare_detector_scores(reference_result, test_result):
    ref = float(reference_result["mean"])
    test = float(test_result["mean"])
    delta = test - ref
    ratio = test / max(ref, 0.001)

    if test >= 0.50:
        interpretation = "SYNTHETIC EVIDENCE"
    elif delta >= 0.10 and test >= 0.10:
        interpretation = "ELEVATED SYNTHETIC INDICATOR — INCONCLUSIVE"
    else:
        interpretation = "NO STRONG SYNTHETIC INDICATOR"

    return {"delta": delta, "ratio": ratio, "interpretation": interpretation}


def result_box(kind: str, title: str, text: str):
    css = {
        "good": "result-good",
        "warn": "result-warn",
        "bad": "result-bad",
        "neutral": "result-neutral",
    }[kind]
    st.markdown(
        f'<div class="{css}"><b>{title}</b><br><span class="muted">{text}</span></div>',
        unsafe_allow_html=True,
    )


# -----------------------------
# UI
# -----------------------------
st.markdown(
    '<div class="hero"><h1>🎙️ VoicePrint Forensics</h1><p>Reference-based speaker verification and AI/synthetic-speech analysis.</p></div>',
    unsafe_allow_html=True,
)

st.info(
    "This is an assistive forensic tool, not a courtroom-grade authenticity test. "
    "A low AI score does not prove that audio is human, and a speaker score is not an authenticity percentage."
)

with st.expander("🧠 What the app checks"):
    st.markdown(
        """
        **1. AI / synthetic detection** — looks for patterns associated with synthetic speech.

        **2. Speaker verification** — compares speaker characteristics using ECAPA-TDNN.

        **3. Acoustic diagnostics** — compares general spectral/audio characteristics only.

        These are separate measurements. The app does **not** turn them into a fake 'authenticity percentage'.
        """
    )

st.markdown('<div class="section-title">1. Upload the two recordings</div>', unsafe_allow_html=True)

left, right = st.columns(2, gap="large")

with left:
    st.markdown('<div class="card"><div class="card-title">🟢 Reference audio</div><div class="muted">Known/genuine voice used as the comparison reference.</div></div>', unsafe_allow_html=True)
    reference_file = st.file_uploader(
        "Choose reference audio",
        type=["wav", "mp3", "flac", "m4a", "ogg", "aac"],
        key="reference",
        help=f"Maximum file size: {MAX_UPLOAD_MB} MB.",
    )

with right:
    st.markdown('<div class="card"><div class="card-title">🔴 Test audio</div><div class="muted">Recording you want to examine.</div></div>', unsafe_allow_html=True)
    test_file = st.file_uploader(
        "Choose test audio",
        type=["wav", "mp3", "flac", "m4a", "ogg", "aac"],
        key="test",
        help=f"Maximum file size: {MAX_UPLOAD_MB} MB.",
    )

if reference_file and test_file:
    reference_bytes = reference_file.getvalue()
    test_bytes = test_file.getvalue()

    if sha256(reference_bytes) == sha256(test_bytes):
        st.error("The two files are exactly identical. Please upload a separate reference and test recording.")
        st.stop()

    # Decode independently so one bad file does not hide which file failed.
    try:
        reference_audio, _ = load_audio(reference_bytes)
    except Exception as exc:
        show_technical_error("Could not decode the reference recording.", exc)
        st.stop()

    try:
        test_audio, _ = load_audio(test_bytes)
    except Exception as exc:
        show_technical_error("Could not decode the test recording.", exc)
        st.stop()

    reference_stats = audio_stats(reference_audio, SR)
    test_stats = audio_stats(test_audio, SR)

    with st.expander("🎧 Preview recordings"):
        p1, p2 = st.columns(2)
        with p1:
            st.caption("Reference")
            st.audio(reference_bytes)
        with p2:
            st.caption("Test")
            st.audio(test_bytes)

    with st.expander("📋 File and audio details"):
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Reference**")
            st.write(f"Duration: {reference_stats['duration']:.2f} s")
            st.write(f"Sample rate after processing: {SR} Hz")
            st.write(f"SHA-256: `{sha256(reference_bytes)[:16]}…`")
        with d2:
            st.markdown("**Test**")
            st.write(f"Duration: {test_stats['duration']:.2f} s")
            st.write(f"Sample rate after processing: {SR} Hz")
            st.write(f"SHA-256: `{sha256(test_bytes)[:16]}…`")

    st.markdown('<div class="section-title">2. Basic audio quality check</div>', unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Reference length", f"{reference_stats['duration']:.1f} s")
    q2.metric("Test length", f"{test_stats['duration']:.1f} s")
    q3.metric("Reference RMS", f"{reference_stats['rms']:.3f}")
    q4.metric("Test RMS", f"{test_stats['rms']:.3f}")

    if reference_stats["duration"] < MIN_DETECTOR_SECONDS or test_stats["duration"] < MIN_DETECTOR_SECONDS:
        st.warning(f"One recording is shorter than {MIN_DETECTOR_SECONDS:.1f} seconds. The synthetic detector may return 'insufficient audio'.")

    if st.button("🔎 Run forensic analysis", type="primary", use_container_width=True):
        st.markdown('<div class="section-title">3. AI / synthetic-speech detection</div>', unsafe_allow_html=True)

        reference_result = None
        test_result = None
        try:
            with st.spinner("Loading specialist synthetic-speech model…"):
                feature_extractor, deepfake_model = load_deepfake_detector()

            reference_result = run_deepfake_detection(reference_audio, feature_extractor, deepfake_model, "Reference")
            test_result = run_deepfake_detection(test_audio, feature_extractor, deepfake_model, "Test")

            del feature_extractor
            del deepfake_model
            gc.collect()

            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("### 🟢 Reference")
                if reference_result["mean"] is not None:
                    st.metric("Mean AI/fake score", f"{reference_result['mean']:.1%}")
                    st.caption(f"Median {reference_result['median']:.1%} • Maximum {reference_result['maximum']:.1%}")
                else:
                    st.warning("Insufficient audio for this detector.")
                st.caption(reference_result["verdict"])

            with c2:
                st.markdown("### 🔴 Test")
                if test_result["mean"] is not None:
                    st.metric("Mean AI/fake score", f"{test_result['mean']:.1%}")
                    st.caption(f"Median {test_result['median']:.1%} • Maximum {test_result['maximum']:.1%}")
                else:
                    st.warning("Insufficient audio for this detector.")
                st.caption(test_result["verdict"])

        except Exception as exc:
            show_technical_error("The specialist synthetic-speech detector could not run. The rest of the analysis can still continue.", exc)

        st.markdown('<div class="section-title">4. Speaker verification</div>', unsafe_allow_html=True)
        speaker_score = None
        speaker_verdict = "UNAVAILABLE"
        try:
            with st.spinner("Comparing speaker characteristics with ECAPA-TDNN…"):
                speaker_score, speaker_verdict = verify_speaker(reference_audio, test_audio)

            if speaker_verdict == "LIKELY SAME SPEAKER":
                result_box("good", "LIKELY SAME SPEAKER", "The score is above the app's conservative operating band.")
            elif speaker_verdict == "LIKELY DIFFERENT SPEAKERS":
                result_box("bad", "LIKELY DIFFERENT SPEAKERS", "The score is below the lower separation band.")
            else:
                result_box("warn", "INCONCLUSIVE", "The score falls in the overlap/uncertain range.")

            st.metric("ECAPA cosine similarity", f"{speaker_score:.4f}")
            st.caption("This is speaker similarity, not a probability and not an authenticity score.")
        except Exception as exc:
            show_technical_error("Speaker verification failed. This does not invalidate the other tests.", exc)

        st.markdown('<div class="section-title">5. Acoustic diagnostics</div>', unsafe_allow_html=True)
        try:
            similarity = acoustic_similarity(reference_audio, test_audio, SR)
            st.metric("Diagnostic acoustic similarity", f"{similarity:.4f}")
            st.caption("Diagnostic only. It is deliberately excluded from the AI-vs-real decision and does not prove speaker identity.")
        except Exception as exc:
            similarity = None
            show_technical_error("Acoustic diagnostics failed.", exc)

        st.markdown('<div class="section-title">6. Evidence interpretation</div>', unsafe_allow_html=True)

        relative = None
        if reference_result and test_result and reference_result["mean"] is not None and test_result["mean"] is not None:
            relative = compare_detector_scores(reference_result, test_result)

            m1, m2, m3 = st.columns(3)
            m1.metric("Reference AI/fake", f"{reference_result['mean']:.1%}")
            m2.metric("Test AI/fake", f"{test_result['mean']:.1%}")
            m3.metric("Test − reference", f"{relative['delta']:+.1%}")

            if relative["interpretation"] == "SYNTHETIC EVIDENCE":
                result_box("bad", "SYNTHETIC EVIDENCE DETECTED", "The test score crosses the model's documented 50% standalone threshold.")
            elif relative["interpretation"].startswith("ELEVATED"):
                result_box("warn", "ELEVATED SYNTHETIC INDICATOR — INCONCLUSIVE", "The test is substantially higher than the supplied reference, but the model remains below its standalone synthetic threshold.")
            else:
                result_box("neutral", "NO STRONG SYNTHETIC INDICATOR", "This model did not find strong synthetic evidence. That does not prove the recording is human.")
        else:
            result_box("neutral", "INCONCLUSIVE", "The specialist detector did not produce usable scores, so synthetic-speech evidence cannot be interpreted.")

        st.markdown('<div class="section-title">7. Final forensic summary</div>', unsafe_allow_html=True)
        summary = []
        summary.append(f"**Speaker comparison:** {speaker_verdict}")
        if test_result is not None:
            summary.append(f"**Synthetic detector:** {test_result['verdict']}")
        if relative is not None:
            summary.append(f"**Reference-calibrated change:** {relative['delta']:+.1%}")
        if similarity is not None:
            summary.append(f"**Acoustic similarity:** {similarity:.4f} (diagnostic only)")
        for line in summary:
            st.write(line)

        if relative and speaker_verdict == "LIKELY SAME SPEAKER" and relative["interpretation"] == "SYNTHETIC EVIDENCE":
            result_box("bad", "HIGH-INTEREST CASE", "The recording is consistent with the reference speaker and the synthetic detector also crosses its standalone threshold. Further independent verification is recommended.")
        elif relative and relative["interpretation"].startswith("ELEVATED"):
            result_box("warn", "CURRENT CONCLUSION: ELEVATED / INCONCLUSIVE", "There is a notable synthetic-speech signal relative to the reference, but the evidence is not strong enough to call the recording a confirmed deepfake.")
        else:
            result_box("neutral", "CURRENT CONCLUSION: INCONCLUSIVE / NO STRONG SYNTHETIC SIGNAL", "The available model evidence is insufficient for a definitive authenticity decision.")

        with st.expander("⚠️ Important limitations"):
            st.markdown(
                """
                - The synthetic detector can miss unseen voice generators, voice-conversion systems, re-recorded audio, noise, and codec artifacts.
                - ECAPA speaker similarity depends on recording quality, speech content, duration, channel conditions, and model calibration.
                - Acoustic similarity is not speaker identity and is not proof of AI generation.
                - Thresholds shown here are conservative operating rules for this application, not universal forensic standards.
                - For a high-stakes decision, use multiple independent methods and human/expert review.
                """
            )

else:
    st.info("Upload both a reference recording and a test recording to begin.")

st.divider()
st.caption("VoicePrint Forensics v7 • ECAPA-TDNN + Wav2Vec2 synthetic-speech detector • Evidence, not certainty")
