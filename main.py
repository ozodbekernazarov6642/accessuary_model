from fastai.vision.all import *
import streamlit as st
import pathlib
import platform
import plotly.express as px

temp = pathlib.PosixPath
pathlib.WindowsPath = pathlib.PosixPath

product = {
    "Watch": "Soat",
    "Umbrella": "Soyabon",
    "Crown": "Toj",
    "Glasses": "Ko'zoynak"
}

st.title("Inson o'zi uchun qo'llaydigan aksessuarlarni klassifikatsiya qilish")
st.markdown(
    "Klassifikatsiya qilinadigan mahsulotlar: <b><i>Ko'zoynak</i></b>, <b><i>Toj</i></b>, <b><i>Soat</i></b>, "
    "<b><i>Soyabon</i></b>",
    unsafe_allow_html=True
)

file = st.file_uploader("Rasm yuklang", type=['webp', 'png', 'jpg'])
if file:
    model = load_learner("accessory_model.pkl")
    img = PILImage.create(file)
    pred, pred_id, probs = model.predict(img)
    if probs[pred_id] * 100 < 90:
        st.image(img, width=200, caption="Rasmingiz bu toifalarga to'g'ri kelmaydi")
    else:
        st.success(f"Bashorat: {product[pred]}")
        st.info(f"Ehtimollik: {probs[pred_id] * 100:.2f}%")
        fig = px.bar(
            x=["Toj", "Ko'zoynak", "Soyabon", "Soat"],
            y=probs * 100,
            labels={"x": "Klasslar", "y": "Ehtimollik"}
        )
        st.plotly_chart(fig)
