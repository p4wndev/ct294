import streamlit as st
import streamlit_shadcn_ui as ui
import pandas as pd
import joblib
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
import time
# Tải các mô hình từ file .pkl
models = {
    "Naive Bayes": joblib.load("models/NB.pkl"),
    # "Random Forest": joblib.load("models/RF.pkl"),
    "Decision Tree": joblib.load("models/DT.pkl"),
    "K-Nearest Neighbors": joblib.load("models/kNN.pkl")
}
df = pd.read_csv('nearest-earth-objects(1910-2024).csv')

# Hàm linear normalization
def linear_nomalize(val, col):
    min_val = df[col].min()
    max_val = df[col].max()
    return (val - min_val) / (max_val - min_val)

# Tiêu đề ứng dụng
st.title("AstraeusDetector 🚀")
ui.badges(badge_list=[("CT294", "default"), ("Applied machine learning", "destructive"), ("Group 03", "outline")], class_name="flex gap-2", key="badges1")
with st.form("my_form"):
    col = st.columns(3)
    with col[0]:
        absolute_magnitude = linear_nomalize(st.number_input("Absolute Magnitude", min_value=0.0), 'absolute_magnitude')
        estimated_diameter_min = st.number_input("Estimated Diameter Min", min_value=0.0)
    with col[1]:
        relative_velocity = linear_nomalize(st.number_input("Relative Velocity", min_value=0.0), 'relative_velocity')
        estimated_diameter_max = st.number_input("Estimated Diameter Max", min_value=0.0)
    with col[2]:
        miss_distance = linear_nomalize(st.number_input("Miss Distance", min_value=0.0), 'miss_distance')
        model_choice = st.selectbox("Choose model:", list(models.keys()))
    with st.popover("Documents"):
        st.markdown("""
                    - **Absolute Magnitude**: The absolute brightness of the asteroid, representing how bright it appears in space.

                    - **Relative Velocity**: The velocity of the asteroid relative to Earth, measured in km/h.

                    - **Miss Distance**: The closest distance between the asteroid and Earth in its orbit.

                    - **Estimated Diameter Min**: The estimated minimum diameter of the asteroid, measured in km.

                    - **Estimated Diameter Max**: The estimated maximum diameter of the asteroid, measured in km.
                    """)
        
    # Chọn mô hình dự đoán
    submitted = st.form_submit_button("Predict")

# Nút để thực hiện dự đoán
if submitted:
    # Tạo dataframe từ thông số đầu vào
    input_data = pd.DataFrame({
        "absolute_magnitude": [absolute_magnitude],
        "estimated_diameter_min": [estimated_diameter_min],
        "estimated_diameter_max": [estimated_diameter_max],
        "relative_velocity": [relative_velocity],
        "miss_distance": [miss_distance]
    })

    # Lấy mô hình được chọn
    model = models[model_choice]

    # Kiểm tra mô hình có hỗ trợ predict_proba hay không
    if hasattr(model, "predict_proba"):
        # Dự đoán xác suất
        start = time.time()
        prediction_proba = model.predict_proba(input_data)
        exetime = (time.time() - start) * 1000 
        confidence = prediction_proba[0][1]
        
        # Hiển thị kết quả dự đoán với xác suất
        if confidence > 0.5:
            col = st.columns(2)
            with col[0]:
                ui.metric_card(title="Prediction Result", content="Dangerous", description=f"Confidence {confidence:.2f} - Execution time {exetime:.2f}ms", key="card1")
        else:
            col = st.columns(2)
            with col[0]:
                ui.metric_card(title="Prediction Result", content="Not Dangerous", description=f"Confidence {1 - confidence:.2f} - Execution time {exetime:.2f}ms", key="card1")

    else:
        st.error(f"Mô hình {model_choice} không hỗ trợ tính xác suất dự đoán.")
   
        
def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(textDecoration="none", **style))(text)

def link2(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)
def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
    """

    style_div = styles(
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        text_align="center",
        height="100px",
        # opacity=0.9
    )

    style_hr = styles(
    )

    body = p()
    foot = div(style=style_div)(hr(style=style_hr), body)

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)
        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

def footer():
    myargs = [
        " Developed with ❤️ by",
        br(),
        link("https://fb.com/is.819", "Mai Ha Ngoc Hai"),
        "ㅤ",
        link("https://www.facebook.com/profile.php?id=100053355381563", "Ly Tri Khai"),
        "ㅤ",
        link("https://fb.com/trimin.tram", "Tram Tri Min"),
        "ㅤ",
        link("https://fb.com/tuitenphuan", "Thai Phu An"),
        br(),
        "Dataset:ㅤ",
        link2("https://www.kaggle.com/datasets/ivansher/nasa-nearest-earth-objects-1910-2024", "NEO 1910-2024"),
        "ㅤNotebook:ㅤ",
        link2("https://colab.research.google.com/drive/1klCWjixBdBRwBZT_kYxjrjTbb4k9bLPY?usp=sharing", "NEO Hazard Prediction"),
    ]
    layout(*myargs)

footer()