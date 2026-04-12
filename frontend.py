import streamlit as st
import joblib
import re
import string
import numpy as np

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Hate Speech Detection AI",
    page_icon="🛡️",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS (Professional UI)
# -------------------------------------------------
st.markdown("""
<style>

.main-title {
    font-size:40px;
    font-weight:bold;
    text-align:center;
    color:#4CAF50;
}

.card {
    padding:20px;
    border-radius:15px;
    background-color:#f4f6f7;
    box-shadow:0px 4px 10px rgba(0,0,0,0.1);
}

.safe {color:green; font-size:28px; font-weight:bold;}
.offensive {color:orange; font-size:28px; font-weight:bold;}
.hate {color:red; font-size:28px; font-weight:bold;}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = joblib.load("hate_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -------------------------------------------------
# CLEAN FUNCTION
# -------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown('<p class="main-title">🛡️ Hate Speech Detection AI</p>',
            unsafe_allow_html=True)

st.write(
"""
This AI system analyzes user text and detects **Hate Speech**, 
**Offensive Language**, or **Safe Content** using Machine Learning.
"""
)

st.divider()

# -------------------------------------------------
# LAYOUT
# -------------------------------------------------
col1, col2 = st.columns([2,1])

with col1:

    user_input = st.text_area(
        "Enter Social Media Text",
        height=200,
        placeholder="Type message here..."
    )

    analyze = st.button("🚀 Analyze Text")

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if analyze:

    if user_input.strip() == "":
        st.warning("Please enter text first.")
    else:

        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)

        confidence = np.max(probability)*100

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 0:
            st.markdown(
                f'<p class="hate">❌ Hate Speech Detected</p>',
                unsafe_allow_html=True
            )

        elif prediction == 1:
            st.markdown(
                f'<p class="offensive">⚠ Offensive Language</p>',
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                f'<p class="safe">✅ Safe Content</p>',
                unsafe_allow_html=True
            )

        st.metric("Confidence Score", f"{confidence:.2f}%")

        st.progress(int(confidence))

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with col2:

    st.markdown("### 🧠 Model Information")

    st.info("""
    **Algorithm:** Logistic Regression  
    **Vectorization:** TF-IDF  
    **Task:** Text Classification  
    """)

    st.markdown("### 👨‍💻 Developer")

    st.success("""
    **Ramneet Singh**  
    Enrollment No: 01613215723  
    SSMDA Project
    """)

st.divider()

st.caption("© 2026 Hate Speech Detection System")