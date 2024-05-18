# WelfareDataModelisation

My study centered on an extensive database comprising no less than 71,000 individuals, allowing for comprehensive multi-dimensional analysis.

The central aim of this thesis was to gain an in-depth understanding of the complex interactions between human metabolism and well-being, utilizing advanced modeling techniques and artificial intelligence. Through a detailed exploration of this massive dataset, I sought to identify trends, correlations, and key factors influencing individuals' well-being.

This project has been an exciting endeavor, merging biology, data science, and AI to offer fresh and valuable insights into how our metabolism may impact our well-being. The predictive models I developed have potential applications in the fields of health, disease prevention, and improving quality of life.

I invite you to delve further into this project in my portfolio, where you will discover the details of my methodology, the fascinating results I achieved, and the potential implications of this research for our understanding of human metabolism and well-being. I hope this exploration inspires you as much as it has inspired me.


## Description des données 

**DATA SOURCE** : https://www.kaggle.com/datasets/prosperchuks/health-dataset/discussion?resource=download&select=hypertension_data.csv

**Stroke**

- sex : patient's gender (1: male; 0: female)

- age :  patient's age (in years)

- hypertension : patient has ever had hypertension (1) or not (0)

- heart_disease : patient has ever had heart_disease(1) or not (0)

- ever_married : patient married (1) or not (0)

- work_type : patient job type: 0 - Never_worked, 1 - children, 2 - Govt_job, 3 - Self-employed, 4 - Private

- Residence_type : patient area: 1 - Urban, 0 - Rural

- avg_glucose_level : patient average blood sugar level

- bmi : Body Mass Index

- smoking_status : 1 - smokes, 0 - never smoked

- stroke : Whether the patient has stroke (1) or not (0)


**Hypertension**

- age : patient's age (in years)

- sex : patient's gender (1: male; 0: female)

- cp : Chest pain type: 0: asymptomatic 1: typical angina 2: atypical angina 3: non-anginal pain

- trestbps : Resting blood pressure (in mm Hg)

- chol : Serum cholestoral in mg/dl

- fbs : if the patient's fasting blood sugar > 120 mg/dl (1: yes; 0: no)

- restecg Resting ECG results: 0: normal 1: ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) 2: probable or definite left ventricular hypertrophy by Estes' criteria

- thalach : Maximum heart rate achieved.

- exang : Exercise induced angina (1: yes; 0: no)

- oldpeak : ST depression induced by exercise relative to rest.

- slope : The slope of the peak exercise ST segment: 0: upsloping 1: flat 2: downsloping

- ca : Number of major vessels (0–3) colored by flourosopy

- thal : 3: Normal; 6: Fixed defect; 7: Reversable defect

- target : Whether the patient has hypertension (1) or not (0)

# Mapping variable 
**Diabetes**

- Age 13-level age category (_AGEG5YR see codebook) 1 = 18-24 9 = 60-64 13 = 80 or older

- Sex patient's gender (1: male; 0: female).

- HighChol 0 = no high cholesterol 1 = high cholesterol

- CholCheck 0 = no cholesterol check in 5 years 1 = yes cholesterol check in 5 years

- BMI Body Mass Index

- Smoker Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes] 0 = no 1 = yes

- HeartDiseaseorAttack : coronary heart disease (CHD) or myocardial infarction (MI) 0 = no 1 = yes

- PhysActivity physical activity in past 30 days - not including job 0 = no 1 = yes

- Fruits Consume Fruit 1 or more times per day 0 = no 1 = yes

- Veggies : Consume Vegetables 1 or more times per day 0 = no 1 = yes

- HvyAlcoholConsump : (adult men >=14 drinks per week and adult women>=7 drinks per week) 0 = no 1 = yes

- GenHlth :  Would you say that in general your health is: scale 1-5 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor

- MentHlth :  days of poor mental health scale 1-30 days

- PhysHlth : physical illness or injury days in past 30 days scale 1-30

- DiffWalk :Do you have serious difficulty walking or climbing stairs? 0 = no 1 = yes

- Stroke : you ever had a stroke. 0 = no, 1 = yes

- HighBP :  0 = no high, BP 1 = high BP

- Diabetes : 0 = no diabetes, 1 = diabetes

