import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Prediksi Kualitas Kopi",
    page_icon="☕",
    layout="centered"
)

model = joblib.load("model_klasifikasi_kualitas_kopi.joblib")

st.title("☕Klasifikasi Kualitas Kopi")
st.markdown("Aplikasi machine learning untuk memprediksi kualitas kopi dengan kategori Baik, Sedang, atau Buruk berdasarkan fitur seperti aroma, warna, keasaman, dan ukuran biji.")

kafein = st.number_input("Kadar Kafein", 50, 200, 90, 1)
keasaman = st.number_input("Tingkat Keasaman", 2.0, 10.0, 5.0, 0.1)
jenis = st.selectbox("Jenis Proses", ["Natural", "Washed", "Honey"])

if st.button("Prediksi"):
    # Data baru dari input
    data_baru = pd.DataFrame([[kafein, keasaman, jenis]],
                             columns=["Kadar Kafein", "Tingkat Keasaman", "Jenis Proses"])
    
    # Prediksi
    prediksi = model.predict(data_baru)[0]
    probabilitas = model.predict_proba(data_baru)[0]
    presentase = max(probabilitas)

    st.success(f"Model memprediksi **{prediksi}** dengan keyakinan {presentase*100:.2f}%")
    st.balloons()

    # Tampilkan distribusi probabilitas
    st.subheader("Distribusi Probabilitas Prediksi")
    prob_df = pd.DataFrame({
        "Kualitas Kopi": model.classes_,
        "Probabilitas": probabilitas
    })
    st.bar_chart(prob_df.set_index("Kualitas Kopi"))

st.divider()
st.caption("Dibuat oleh Raditya Fauzi Pratama")


