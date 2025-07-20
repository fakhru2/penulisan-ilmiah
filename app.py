import streamlit as st
import torch
import torch.nn.functional as F
import pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

# ==== Setup device ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Model paths ====
sentiment_model_path = "fakhrusyi/sentimen"
emotion_model_path = "fakhrusyi/emosi"

# ==== Load tokenizer (boleh salah satu) ====
tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)

# ==== Load models ====
def safe_load_model(model_name):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model dari Hugging Face: {model_name}\n\n**Error:** {str(e)}")
        return None

sentiment_model = safe_load_model(sentiment_model_path)
emotion_model = safe_load_model(emotion_model_path)

# ==== Download dan load LabelEncoder dari Hugging Face ====
try:
    sentiment_encoder_path = hf_hub_download(repo_id="fakhrusyi/sentimen", filename="sentiment_encoder.pkl")
    emotion_encoder_path = hf_hub_download(repo_id="fakhrusyi/emosi", filename="emotion_encoder.pkl")

    with open(sentiment_encoder_path, "rb") as f:
        sentiment_le = pickle.load(f)
    with open(emotion_encoder_path, "rb") as f:
        emotion_le = pickle.load(f)

    sentiment_label_mapping = dict(zip(sentiment_le.transform(sentiment_le.classes_), sentiment_le.classes_))
    emotion_label_mapping = dict(zip(emotion_le.transform(emotion_le.classes_), emotion_le.classes_))
except Exception as e:
    st.error(f"❌ Gagal memuat LabelEncoder dari Hugging Face: {e}")
    sentiment_label_mapping = {}
    emotion_label_mapping = {}

# ==== Fungsi prediksi ====
def predict_sentiment_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Sentimen
    sent_label = sent_conf = sent_probs = None
    if sentiment_model:
        with torch.no_grad():
            sent_out = sentiment_model(**inputs)
        sent_probs = F.softmax(sent_out.logits, dim=1)[0]
        sent_idx = sent_probs.argmax().item()
        sent_label = sentiment_label_mapping.get(sent_idx, "N/A")
        sent_conf = sent_probs[sent_idx].item()
        sent_probs = {sentiment_label_mapping[i]: float(sent_probs[i]) for i in range(len(sentiment_label_mapping))}

    # Emosi
    emo_label = emo_conf = emo_probs = None
    if emotion_model:
        with torch.no_grad():
            emo_out = emotion_model(**inputs)
        emo_probs = F.softmax(emo_out.logits, dim=1)[0]
        emo_idx = emo_probs.argmax().item()
        emo_label = emotion_label_mapping.get(emo_idx, "N/A")
        emo_conf = emo_probs[emo_idx].item()
        emo_probs = {emotion_label_mapping[i]: float(emo_probs[i]) for i in range(len(emotion_label_mapping))}

    return sent_label, sent_conf, sent_probs, emo_label, emo_conf, emo_probs

# ==== Streamlit Layout ====
st.set_page_config(page_title="Dashboard Analisis Sentimen & Emosi", layout="centered")
st.title("📊 Analisis Sentimen & Emosi Komentar")
st.write("Masukkan komentar di bawah untuk memprediksi sentimen dan emosi menggunakan model BERT yang sudah dilatih.")

user_input = st.text_area("📝 Komentar:", height=150)

if st.button("Analisis"):
    if not user_input.strip():
        st.warning("Silakan masukkan komentar terlebih dahulu.")
    else:
        sent_label, sent_conf, sent_probs, emo_label, emo_conf, emo_probs = predict_sentiment_emotion(user_input)

        st.markdown("### 🎯 Hasil Prediksi")

        if sent_label is not None:
            st.write(f"**Sentimen**: `{sent_label}` (confidence: {sent_conf:.2f})")
            st.markdown("#### 📊 Probabilitas Sentimen")
            st.bar_chart(pd.DataFrame.from_dict(sent_probs, orient='index', columns=["Probabilitas"]))
        else:
            st.warning("Model sentimen tidak tersedia.")

        if emo_label is not None:
            st.write(f"**Emosi**: `{emo_label}` (confidence: {emo_conf:.2f})")
            st.markdown("#### 🎭 Probabilitas Emosi")
            st.bar_chart(pd.DataFrame.from_dict(emo_probs, orient='index', columns=["Probabilitas"]))
        else:
            st.warning("Model emosi tidak tersedia.")
