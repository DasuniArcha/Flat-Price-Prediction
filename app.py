import streamlit as st
import pandas as pd
import pickle
import base64
from pathlib import Path


# PAGE CONFIG
st.set_page_config(
    page_title="Flat Price Prediction",
    page_icon="🏢",
    layout="wide"
)


# BACKGROUND + GLOBAL CSS
def set_bg_image(image_path: str):
    img_file = Path(image_path)
    if not img_file.exists():
        return  

    b64 = base64.b64encode(img_file.read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(10,10,20,0.55), rgba(10,10,20,0.55)),
                url("data:image/jpg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;

        }}
    

        /* Hide default Streamlit header/footer */
        header[data-testid="stHeader"] {{
            background: transparent;
        }}
        footer {{
            visibility: hidden;
        }}

        /* Main container spacing */
        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 4rem;
        }}

        /* Navbar */
        .nav {{
            position: sticky;
            top: 0;
            z-index: 999;
            backdrop-filter: blur(12px);
            background: rgba(20, 20, 35, 0.55);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 14px 18px;
            margin-bottom: 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .nav .brand {{
            display: flex;
            gap: 10px;
            align-items: center;
            color: #fff;
            font-weight: 700;
            font-size: 28px;
            letter-spacing: 0.2px;
        }}
        .nav .links {{
            display: flex;
            gap: 16px;
            align-items: center;
            font-size: 14px;
        }}
        .nav .links a {{
            color: rgba(255,255,255,0.85);
            text-decoration: none;
            padding: 6px 10px;
            border-radius: 10px;
            border: 1px solid transparent;
        }}
        .nav .links a:hover {{
            border-color: rgba(255,255,255,0.15);
            background: rgba(255,255,255,0.06);
        }}


        /* Cards */
        .card {{
            background: rgba(255,255,255,0.25);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
            font-weight: 800;
        }}
        .card h3 {{
            color: #ffffff;
            margin: 0 0 10px 0;
            font-size: 25px;
            font-weight: 800;
        }}
        .label-title {{
            color: white;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 4px;
        }}
        .muted {{
            color: rgba(255,255,255,0.78);
            font-size: 13px;
            margin-top: 2px;
        }}

        
        
        /* Success box */
        /* Center ONLY the result box */
        .result-wrapper {{
            display: flex;
            justify-content: center;
            margin-top: 16px;
            width: 100%;
        }}

        /* Keep result box size controlled */
        .result {{
            background: linear-gradient(135deg, rgba(37,99,235,0.85), rgba(30,64,175,0.85));
            border-radius: 18px;
            padding: 16px 36px;
            color: white;
            font-size: 18px;
            font-weight: 700;
            text-align: center;
            max-width: 600px;
        }}

        .center-box {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 14px;
            margin-top: 10px;
        }}

        div.stButton > button {{
            width: 190px;
            height: 44px;
            font-size: 15px;
            font-weight: 700;
            border-radius: 16px;
        }}

        /* Make inputs look nicer */
        div[data-testid="stNumberInput"], div[data-testid="stTextInput"], div[data-testid="stSelectbox"] {{
            background: rgba(255,255,255,0.02);
            border-radius: 14px;
            padding: 6px;
        }}

        /* Footer */
        .custom-footer {{
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0;
            padding: 12px 18px;
            background: rgba(20, 20, 35, 0.65);
            backdrop-filter: blur(10px);
            border-top: 1px solid rgba(255,255,255,0.10);
            color: rgba(255,255,255,0.75);
            font-size: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .custom-footer b {{
            color: rgba(255,255,255,0.92);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_bg_image("img/bg.jpg")


# NAVBAR 
st.markdown(
    """
    <div class="nav">
        <div class="brand">🏢 Flat Price Predictor</div>
        <div class="links">
            <a href="#predict">Predict</a>
            <a href="#about">About</a>
            <a href="#help">Help</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# LOAD MODEL & ENCODERS
@st.cache_resource
def load_model():
    with open("model_catboost.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_model()

# # 🔑 EXACT feature order used during training
# FEATURE_ORDER = list(model.feature_names_in_)
FEATURE_ORDER = [
    "district_name",
    "rooms_count",
    "total_area",
    "kitchen_area",
    "bath_area",
    "other_area",
    "extra_area",
    "extra_area_count",
    "floor",
    "floor_max",
    "ceil_height",
    "bath_count",
    "flat_age",
    "gas",
    "hot_water",
    "central_heating",
    "extra_area_type_name"
]


# PREPROCESSING
def preprocess(df):
    binary_map = {'yes': 1, 'no': 0}

    for col in ['gas', 'hot_water', 'central_heating']:
        df[col] = df[col].astype(str).str.lower().map(binary_map).fillna(0)

    df['flat_age'] = 2023 - df['year']

    for col in ['district_name', 'extra_area_type_name']:
        encoder = encoders[col]
        df[col] = df[col].astype(str).map(
            lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
        )

    return df

def clear_form():
    keys_to_clear = [
        "district_name",
        "extra_area_type_name",
        "gas_select",
        "hot_water_select",
        "central_heating_select",
        "total_area",
        "kitchen_area",
        "bath_area",
        "other_area",
        "extra_area",
        "extra_area_count",
        "rooms_count",
        "bath_count",
        "floor",
        "floor_max",
        "ceil_height",
        "year"
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    st.rerun()



#UI
st.markdown('<a id="predict"></a>', unsafe_allow_html=True)
st.markdown('<div class="card"><h3>🔮 Predict Flat Price</h3><div class="muted">Enter flat details below. Prediction will use your trained model.pkl + encoders.pkl.</div></div>', unsafe_allow_html=True)
st.write("")  


c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown('<div class="card"><h3>📍 Location</h3>', unsafe_allow_html=True)
    st.markdown('<div class="label-title">District Name</div>', unsafe_allow_html=True)
    district = st.text_input("", placeholder="e.g., Nevskij", key="district_name", label_visibility="collapsed")

    st.markdown('<div class="label-title">Extra Area Type</div>', unsafe_allow_html=True)
    extra_area_type = st.text_input("", placeholder="e.g., balcony", key="extra_area_type_name", label_visibility="collapsed")



    st.markdown('<div class="card"><h3>🔥 Amenities</h3>', unsafe_allow_html=True)
    st.markdown('<div class="label-title">Gas</div>', unsafe_allow_html=True)
    gas = st.selectbox(
        "",
        ["Yes", "No"],
        key="gas_select",
        label_visibility="collapsed"
    )

    st.markdown('<div class="label-title">Hot Water</div>', unsafe_allow_html=True)
    hot_water = st.selectbox(
        "",
        ["Yes", "No"],
        key="hot_water_select",
        label_visibility="collapsed"
    )

    st.markdown('<div class="label-title">Central Heating</div>', unsafe_allow_html=True)
    central_heating = st.selectbox(
        "",
        ["Yes", "No"],
        key="central_heating_select",
        label_visibility="collapsed"
    )


    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card"><h3>📐 Areas</h3>', unsafe_allow_html=True)
    st.markdown('<div class="label-title">Total Area (sq.m)</div>', unsafe_allow_html=True)
    total_area = st.number_input("", min_value=0.0, step=0.1, key="total_area", label_visibility="collapsed")

    st.markdown('<div class="label-title">Kitchen Area (sq.m)</div>', unsafe_allow_html=True)
    kitchen_area = st.number_input("", min_value=0.0, step=0.1, key="kitchen_area", label_visibility="collapsed")

    st.markdown('<div class="label-title">Bath Area (sq.m)</div>', unsafe_allow_html=True)
    bath_area = st.number_input("", min_value=0.0, step=0.1, key="bath_area", label_visibility="collapsed")

    st.markdown('<div class="label-title">Other Area (sq.m)</div>', unsafe_allow_html=True)
    other_area = st.number_input("", min_value=0.0, step=0.1, key="other_area", label_visibility="collapsed")

    st.markdown('<div class="label-title">Extra Area (sq.m)</div>', unsafe_allow_html=True)
    extra_area = st.number_input("", min_value=0.0,step=0.1, key="extra_area", label_visibility="collapsed")

    st.markdown('<div class="label-title">Extra Area Count</div>', unsafe_allow_html=True)
    extra_area_count = st.number_input("", min_value=0, max_value=10, step=1, key="extra_area_count", label_visibility="collapsed")


    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="card"><h3>🏗 Building</h3>', unsafe_allow_html=True)
    st.markdown('<div class="label-title">Rooms Count</div>', unsafe_allow_html=True)
    rooms = st.number_input("", min_value=0, max_value=10,  step=1, key="rooms_count", label_visibility="collapsed")

    st.markdown('<div class="label-title">Bathroom Count</div>', unsafe_allow_html=True)
    bath_count = st.number_input("", min_value=0, max_value=10, value=1, step=1, key="bath_count", label_visibility="collapsed")


    st.markdown('<div class="label-title">Floor</div>', unsafe_allow_html=True)
    floor = st.number_input("", min_value=0, max_value=100, step=1, key="floor", label_visibility="collapsed")

    st.markdown('<div class="label-title">Max Floor</div>', unsafe_allow_html=True)
    floor_max = st.number_input("", min_value=0, max_value=100,step=1, key="floor_max", label_visibility="collapsed")

    st.markdown('<div class="label-title">Ceiling Height (m)</div>', unsafe_allow_html=True)
    ceil_height = st.number_input("", min_value=0.0, step=0.01, key="ceil_height", label_visibility="collapsed")

    st.markdown('<div class="label-title">Year Built</div>', unsafe_allow_html=True)
    year = st.number_input("", min_value=1900, max_value=2024, step=1, key="year", label_visibility="collapsed")


    

    st.markdown('</div>', unsafe_allow_html=True)

st.write("")


# PREDICTION
predict_clicked = False

predict_col1, predict_col2, predict_col3 = st.columns([3, 2, 2])

with predict_col2:
    predict_clicked = st.button("💰 Predict Price")
    

if predict_clicked:
    input_data = pd.DataFrame([{
        'district_name': district,
        'rooms_count': rooms,
        'total_area': total_area,
        'kitchen_area': kitchen_area,
        'bath_area': bath_area,
        'other_area': other_area,
        'extra_area': extra_area,
        'extra_area_count': extra_area_count,
        'floor': floor,
        'floor_max': floor_max,
        'ceil_height': ceil_height,
        'bath_count': bath_count,
        'year': year,
        'gas': gas,
        'hot_water': hot_water,
        'central_heating': central_heating,
        'extra_area_type_name': extra_area_type
    }])

    input_data = preprocess(input_data)
    input_data = input_data[FEATURE_ORDER]

    prediction = model.predict(input_data)[0]

    # FULL-WIDTH CENTERED RESULT
    st.markdown(
        f"""
        <div class="result-wrapper">
            <div class="result">
                Estimated Price: Rs. {prediction:,.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# FOOTER
st.markdown(
    """
    <div class="custom-footer">
        <div><b>Flat Price Prediction</b> • Streamlit App</div>
        <div>© 2026 • st20284507</div>
    </div>
    """,
    unsafe_allow_html=True
)




