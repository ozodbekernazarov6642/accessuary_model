from fastai.vision.all import *
import streamlit as st
import plotly.express as px
import pathlib

# WindowsPath ni Linuxda ishlashga moslashtirish
pathlib.WindowsPath = pathlib.PosixPath

# Klass nomlarini tarjima qilish
product = {
    "Watch": "Soat",
    "Umbrella": "Soyabon",
    "Crown": "Toj",
    "Glasses": "Ko'zoynak"
}

# Sarlavha va tavsif
st.title("Inson o'zi uchun qo‘llaydigan aksessuarlarni klassifikatsiya qilish")
st.markdown(
    """
    Klassifikatsiya qilinadigan mahsulotlar:
    <b><i>Ko'zoynak</i></b>, <b><i>Toj</i></b>, <b><i>Soat</i></b>, <b><i>Soyabon</i></b>
    """,
    unsafe_allow_html=True
)

# Fayl yuklash
file = st.file_uploader("Rasm yuklang", type=['webp', 'png', 'jpg', 'jpeg'])

if file:
    # Modelni yuklash
    model = load_learner("accessory_model.pkl")

    # Rasmdan PILImage yaratish
    img = PILImage.create(file)

    # Bashorat qilish
    pred, pred_id, probs = model.predict(img)

    # Ehtimollik tekshirish
    if probs[pred_id] * 100 < 90:
        st.image(img, width=200, caption="Rasmingiz ushbu toifalarga to‘g‘ri kelmaydi")
    else:
        st.image(img, width=200, caption=f"Bashorat: {product[pred]}")
        st.success(f"Bashorat: {product[pred]}")
        st.info(f"Ehtimollik: {probs[pred_id] * 100:.2f}%")

        # Ehtimolliklarni grafikda ko‘rsatish
        fig = px.bar(
            x=[product[c] for c in product.keys()],
            y=probs * 100,
            labels={'x': "Klasslar", "y": "Ehtimollik (%)"},
            title="Klassifikatsiya ehtimolliklari"
        )
        st.plotly_chart(fig)
