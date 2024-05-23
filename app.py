import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go 
import joblib
import time
from scripts.data import *
from scripts.model import *
import seaborn as sns


st.set_page_config(
    page_title="Welfare App",
    page_icon="img/icon.png",  # You can use emojis or the path to an image file
    layout="wide"
)

# Add the hero image at the top of the page
#st.image("img/img1.png", use_column_width=True)

# Fetch the data from the provided source
diabetes = pd.read_csv('data/data.csv')

# Create tabs
tabs = st.tabs(["Dashboard", "Simulator","Model Deployments"])

# Dashboard tab
with tabs[0]:
    st.header("Dashboard")
    st.write('')

    # Display the diabetes data as a table
    if diabetes is not None:
        st.subheader('Metabolism Data')
        st.table(diabetes.head())
    else:
        st.write("Diabetes data not available.")
        
    col1,col2 = st.columns(2)
    # Display the number chart
    col1.metric(label="Number of individuals", value=len(diabetes))
    
    # Number of cluster
    col2.metric(label="Metabolism Cluster", value=4)

    feature = st.selectbox('Explain Category', options=['cluster'] + [i for i in diabetes.columns])

    # Define custom color scales based on icon logo (example colors)
    custom_colors = ['#1f78b4', '#33a02c', '#b2df8a', '#a6cee3']

    # Create an interactive scatter plot with specific colormaps
    fig = px.scatter(diabetes, x='umap_x', y='umap_y', color=feature,
                    color_discrete_sequence=custom_colors, color_continuous_scale=custom_colors)

    annot = False
    marker_x = 0
    marker_y = 0

    if annot:
        # Add a marker annotation at position x=11, y=10
        annotation_trace = go.Scatter(
            x=[marker_x],
            y=[marker_y],
            mode='markers+text',
            marker=dict(size=15, color='black', symbol='x'),
            text=['Your Metabolism'],
            textfont=dict(size=16, color='black'),
            textposition='top center',
            name='Your Metabolism'
        )

        # Add the annotation trace after the main plot data
        fig.add_trace(annotation_trace)

    fig.update_layout(height=600)
    # Display the plot in Streamlit
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.write('Cluster Description')

    st.markdown('''| Index | Category             | Description                                                                                  |
|-------|----------------------|----------------------------------------------------------------------------------------------|
| 0     | Young Energy         | Characterized by young individuals with high energy, a high consumption of alcohol and tobacco. |
| 1     | Vitality and Balance | Composed of healthy young adults who regularly engage in physical activity.                   |
| 2     | Balanced Wisdom      | Groups older individuals with high cholesterol levels and moderate health.                    |
| 3     | Vigilant Serenity    | Gathers older individuals with high cholesterol levels, heart diseases, and strokes.          |
 ''')

    st.divider()
    st.markdown('''| Cluster | Main Characteristics                                                                         |
|---------|----------------------------------------------------------------------------------------------|
| 0       | Active young adults, smokers, and high alcohol consumption                                   |
| 1       | Healthy young adults, few smokers and alcohol consumers, high physical activity              |
| 2       | Older individuals, high cholesterol levels, average health, moderate physical activity       |
| 3       | Older individuals, high cholesterol levels, heart diseases and strokes, moderate physical activity |
''')
    st.divider()
    st.markdown('''| VAR/Cluster                                                  | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|--------------------------------------------------------------|-----------|-----------|-----------|-----------|
| Average age                                                  | 33.42     | **24.83** | 33.29     | 28.61     |
| Proportion of women                                          | 39.09%    | 46.16%    | **53.02%**| 42.56%    |
| Proportion of people with high cholesterol                   | **61.63%**| 38.98%    | 53.79%    | 60.14%    |
| Proportion of people who have checked their cholesterol level| 97.72%    | 96.11%    | 98.04%    | 98.60%    |
| Average Body Mass Index (BMI)                                | 24.83     | 24.83     | **33.29** | 28.61     |
| Proportion of smokers                                        | **53.51%**| 40.96%    | 44.86%    | **53.69%**|
| Proportion of people with heart disease or heart attack      | 19.45%    | **7.01%** | 11.60%    | **24.62%**|
| Proportion of physical activity                              | 60.58%    | **82.97%**| 72.47%    | 60.34%    |
| Proportion of daily fruit consumption                        | 55.54%    | **65.62%**| 59.61%    | 63.01%    |
| Proportion of daily vegetable consumption                    | 75.04%    | **83.05%**| 78.57%    | 75.96%    |
| Proportion of excessive alcohol consumption                  | 4.12%     | **5.87%** | 3.58%     | 3.13%     |
| Average general health assessment (GenHlth)                  | 3.40      | **2.16**  | 2.63      | 3.43      |
| Average number of days of poor mental health (MentHlth)      | **13.66** | 0.59      | 0.60      | 1.81      |
| Average number of days of poor physical health (PhysHlth)    | **9.37**  | 0.27      | 0.26      | **16.80** |
| Proportion of people with difficulty walking or climbing stairs (DiffWalk) | 43.57% | **5.35%**  | 14.79%    | **46.90%**|
| Proportion of people who have had a stroke                   | 8.99%     | **2.75%** | 3.82%     | **11.15%**|
| Proportion of people with high blood pressure (HighBP)       | **66.76%**| 35.21%    | 62.04%    | **67.07%**|
| Proportion of people with diabetes                           | **62.84%**| 27.12%    | 56.01%    | 60.18%    |
''')

# Form tab
with tabs[1]:
    st.header("Simulator")
    left, right = st.columns([3, 1])
    with left:
        with st.form(key='diabetes_form'):
            col1, col2, col3, col4 = st.columns(4)
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
            with col4: 
                model = st.selectbox('Choose Your Model', options=['Default', 'Personalized'])
                # Form submission button
                submit_button = st.form_submit_button(label='Predict')

        st.divider()

    with right:
        if submit_button:
            if model == 'Personalized':
                st.error('No model deployed yet :(!')
            else:
                with st.expander('Form submitted with the following data:'):
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
                with st.spinner('Model is running'):
                    # Load the model
                    loaded_model = joblib.load('meta_machina.pkl')
                    # Create input array for the model
                    input_data = [[age, sex, high_chol, chol_check, bmi, smoker, heart_disease_or_attack, phys_activity, fruits, veggies, hvy_alcohol_consump, gen_hlth, ment_hlth, phys_hlth, diff_walk, stroke, high_bp, diabetes]]
                    # Perform the prediction
                    prediction = loaded_model.predict(input_data)
                    cluster_label = {
                        0: 'Young Energy',
                        1: 'Vitality and Balance',
                        2: 'Balanced Wisdom',
                        3: 'Vigilant Serenity'
                    }
                    # Create a reverse mapping
                    reverse_label = {v: k for k, v in cluster_label.items()}
                    st.success(f'Prediction result: {cluster_label[prediction[0]]}')
                    marker_x = 11
                    marker_y = 11
                    annot= True
                    st.write('dodo')

with tabs[2]:
    st.header('Meta Detector Model : Devellop train and deploy')
    st.write('How to deploy')
    st.divider()
    st.header('Projector Model : Devellop train and deploy')