import streamlit as st
import matplotlib.pyplot as plt 
from scripts.data import getdata
import seaborn as sns
import pandas as pd
import numpy as np

# Set the page configuration to wide mode
st.set_page_config(page_title="Welfare App", layout="wide")

# Fetch the data from the provided source
diabetes, stroke, hypertension = getdata('data')

# Create tabs
tabs = st.tabs(["Dashboard", "Form"])

# Dashboard tab
with tabs[0]:
    st.header("Dashboard")
    st.write('Welcome to your Data Dashboard')

    # Display the diabetes data as a table
    if diabetes is not None:
        st.subheader('Diabetes Data')
        st.table(diabetes.head())
    else:
        st.write("Diabetes data not available.")
        
    # Display the number chart
    st.metric(label="Mean Age", value=diabetes['Age'].mean())

    st.subheader('Summary Statistics')
    st.write(diabetes.describe())

    fig, ax = plt.subplots()
    sns.heatmap(diabetes.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader('BMI Distribution')
    fig, ax = plt.subplots()
    sns.histplot(diabetes['BMI'], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader('Smoker Distribution')
    smoker_counts = diabetes['Smoker'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(smoker_counts, labels=['No', 'Yes'], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

# Form tab
with tabs[1]:
    st.header("Form")
    with st.form(key='diabetes_form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.selectbox('Age Category', options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], format_func=lambda x: f'{x*5+13} years' if x != 13 else '80 or older')
            sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
            high_chol = st.selectbox('High Cholesterol', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
            chol_check = st.selectbox('Cholesterol Check in 5 years', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
            bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, step=0.1)
            smoker = st.selectbox('Smoker', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
        with col2:
            heart_disease_or_attack = st.selectbox('Heart Disease or Attack', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
            phys_activity = st.selectbox('Physical Activity', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
            fruits = st.selectbox('Consume Fruit 1+ times/day', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
            veggies = st.selectbox('Consume Veggies 1+ times/day', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
            hvy_alcohol_consump = st.selectbox('Heavy Alcohol Consumption', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
            gen_hlth = st.selectbox('General Health', options=[1, 2, 3, 4, 5], format_func=lambda x: ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'][x-1])
        with col3:
            ment_hlth = st.number_input('Days of Poor Mental Health', min_value=0, max_value=30, step=1)
            phys_hlth = st.number_input('Physical Illness or Injury Days', min_value=0, max_value=30, step=1)
            diff_walk = st.selectbox('Difficulty Walking/Stairs', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
            stroke = st.selectbox('Stroke', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
            high_bp = st.selectbox('High Blood Pressure', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
            diabetes = st.selectbox('Diabetes', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

        # Form submission button
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        st.write('Form submitted with the following data:')
        st.write({
            'Age': age,
            'Sex': sex,
            'HighChol': high_chol,
            'CholCheck': chol_check,
            'BMI': bmi,
            'Smoker': smoker,
            'HeartDiseaseorAttack': heart_disease_or_attack,
            'PhysActivity': phys_activity,
            'Fruits': fruits,
            'Veggies': veggies,
            'HvyAlcoholConsump': hvy_alcohol_consump,
            'GenHlth': gen_hlth,
            'MentHlth': ment_hlth,
            'PhysHlth': phys_hlth,
            'DiffWalk': diff_walk,
            'Stroke': stroke,
            'HighBP': high_bp,
            'Diabetes': diabetes,
        })
