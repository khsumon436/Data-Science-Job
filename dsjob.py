import streamlit as st
import pandas as pd
import joblib

# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Data Science Job Prediction",
    page_icon="🤖",
    layout="wide"
)


# =========================
# Load Model
# =========================

@st.cache_resource
def load_model():
    model = joblib.load("Data Science.pkl")
    return model


model = load_model()


# =========================
# Title
# =========================

st.title("🤖 Data Science Job Chance Prediction")

st.write(
    "Enter your details below to predict whether a candidate is likely to look for a new job."
)


# =========================
# Display Image
# =========================

st.image(
    "datascience job img.webp",
    use_container_width=True
)


st.divider()


# =========================
# Input Section
# =========================

col1, col2, col3 = st.columns(3)


with col1:

    city = st.selectbox(
        "Select City",
        [
            "city_103",
            "city_21",
            "city_16",
            "city_114",
            "city_160",
            "city_129",
            "city_111",
            "city_121",
            "city_140",
            "city_171"
        ]
    )

    city_development_index = st.number_input(
        "City Development Index",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.001
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Other"
        ]
    )


with col2:

    relevent_experience = st.selectbox(
        "Relevant Experience",
        [
            "Has relevent experience",
            "No relevent experience"
        ]
    )

    enrolled_university = st.selectbox(
        "University Enrollment",
        [
            "no_enrollment",
            "Full time course",
            "Part time course"
        ]
    )

    education_level = st.selectbox(
        "Education Level",
        [
            "Graduate",
            "Masters",
            "High School",
            "Phd",
            "Primary School"
        ]
    )

    major_discipline = st.selectbox(
        "Major Discipline",
        [
            "STEM",
            "Humanities",
            "Other",
            "Business Degree",
            "Arts",
            "No Major"
        ]
    )


with col3:

    experience = st.number_input(
        "Years of Experience",
        min_value=0,
        max_value=20,
        value=5
    )

    company_size = st.selectbox(
        "Company Size",
        [
            "<10",
            "10/49",
            "50-99",
            "100-500",
            "500-999",
            "1000-4999",
            "5000-9999",
            "10000+"
        ]
    )

    company_type = st.selectbox(
        "Company Type",
        [
            "Pvt Ltd",
            "Funded Startup",
            "Public Sector",
            "Early Stage Startup",
            "NGO",
            "Other"
        ]
    )

    training_hours = st.number_input(
        "Training Hours",
        min_value=0,
        max_value=300,
        value=30
    )


st.divider()


# =========================
# Prediction Button
# =========================

if st.button("🔮 Predict Job Change", use_container_width=True):

    input_data = pd.DataFrame({
        "city": [city],
        "city_development_index": [city_development_index],
        "gender": [gender],
        "relevent_experience": [relevent_experience],
        "enrolled_university": [enrolled_university],
        "education_level": [education_level],
        "major_discipline": [major_discipline],
        "experience": [experience],
        "company_size": [company_size],
        "company_type": [company_type],
        "training_hours": [training_hours]
    })

    try:

        prediction = model.predict(input_data)

        st.divider()

        if prediction[0] == 1:

            st.success(
                "🎯 Prediction: The candidate is likely to look for a new job."
            )

        else:

            st.info(
                "✅ Prediction: The candidate is not likely to look for a new job."
            )


    except Exception as e:

        st.error("❌ Prediction Error!")

        st.write(e)

        st.write("Input Data:")

        st.dataframe(input_data)