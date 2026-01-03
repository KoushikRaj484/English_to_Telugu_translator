import re
import streamlit as st
import numpy as np
import tensorflow as tf
import sentencepiece as spm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

# ================= CONFIG =================
MAX_LEN = 80   # 🔥 MUST MATCH TRAINING
BOS_ID = 1
EOS_ID = 2

# ================= UI =====================
st.set_page_config(
    page_title="English to Telugu Translator",
    page_icon="🌐"
)

st.title("🌐 English → Telugu Translator")
st.subheader("Translate English text into Telugu using Transformer")

st.write(
    "This application uses a **6-Encoder + 6-Decoder Transformer model** "
    "trained on **14.5 lakh sentence pairs**."
)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model_and_tokenizers():
    model = keras.models.load_model(
        "translator_21lks_5.keras",
        compile=False
    )

    sp_eng = spm.SentencePieceProcessor(
        model_file="spm_eng (2).model"
    )
    sp_tel = spm.SentencePieceProcessor(
    
        model_file="spm_tel (2).model"
    )

    # 🔥 Extract MAX_LEN from model
    max_len = model.input_shape[0][1]

    return model, sp_eng, sp_tel, max_len


model, sp_eng, sp_tel, MAX_LEN = load_model_and_tokenizers()


# ================= LAYOUT =================
col1, col2 = st.columns(2)

with col1:
    st.header("📝 Enter English Text")
    input_text = st.text_area(
        "Type your English sentence here:",
        height=150,
        placeholder="How are you today?"
    )

    translate_button = st.button("🚀 Translate")

with col2:
    st.header("📘 Telugu Translation")
    output_box = st.empty()

# ================= UTILS =================
def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\u0C00-\u0C7F\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_eng(text):
    return sp_eng.encode(text, out_type=int)


def translate(sentence):
    sentence = clean(sentence)

    # Encoder input
    enc_seq = encode_eng(sentence)
    enc_seq = pad_sequences(
        [enc_seq],
        maxlen=MAX_LEN,
        padding="post"
    )

    # Decoder starts with BOS
    dec_tokens = [BOS_ID]

    for _ in range(MAX_LEN):
        dec_input = pad_sequences(
            [dec_tokens],
            maxlen=MAX_LEN,
            padding="post"
        )

        preds = model.predict(
            [enc_seq, dec_input],
            verbose=0
        )

        next_token = int(
            np.argmax(preds[0, len(dec_tokens)-1])
        )

        if next_token == EOS_ID:
            break

        dec_tokens.append(next_token)

    return sp_tel.decode(dec_tokens[1:])

# ================= RUN =================
if translate_button:
    if not input_text.strip():
        st.warning("⚠️ Please enter some English text.")
    else:
        with st.spinner("🔄 Translating..."):
            input_text = input_text.split(".")
            result = ""
            for i in input_text:
                i = i.strip()
                if i:
                    result += translate(i) + ". "

        output_box.success(result)
      


# llms_translator_14.5lks_5.keras
# translator_4lks_15.keras

# translator_21lks_1.keras
# translator_21lks_2.keras