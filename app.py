import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu 
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title='Multi Disease Predict Streamlit',
    page_icon='🩺',
    layout='wide'
)

# loading the saved models
diabetes_model = pickle.load(open("./models/diabetes_model.sav",'rb'))
parkinsons_model = pickle.load(open("./models/parkinsons_model.sav",'rb'))
lung_cancer_model = pickle.load(open("./models/lung_cancer.sav",'rb'))

# sidebar navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System', 
                           [ 'Diabetes Prediction','Parkinson\'s Prediction', 'Stroke Prediction', 'Autism Prediction' , 
                            'Lung Cancer Prediction', 'Covid Prediction' , 'Anemia Prediction'],
                           icons=['activity','person','','gender-female', 'lungs', 'virus', 'heart'],
                           default_index=0)
    
# Covid-19 Prediction
df1=pd.read_csv("./dataFiles/Covid-19 Predictions.csv")
x1=df1.drop("Infected with Covid19",axis=1)
x1=np.array(x1)
y1=pd.DataFrame(df1["Infected with Covid19"])
y1=np.array(y1)
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.2,random_state=0)
model1=RandomForestClassifier()
model1.fit(x1_train,y1_train)

# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    # page title
    st.title('Diabetes Prediction using ML')
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
          
    # create a button for prediction
    if st.button('Diabetes Test Result'):
        if not all([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
            st.warning("Please fill in all the fields.")
        else:
            diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            
            if (diab_prediction[0] == 1):
              st.warning('The person is diabetic')
            else:
              st.success('The person is not diabetic')
        

# Parkinsons Prediction Page  
if (selected == 'Parkinson\'s Prediction'):    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP: Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP: Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP: Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP: Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP: Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP: RAP')
        
    with col2:
        PPQ = st.text_input('MDVP: PPQ')
        
    with col3:
        DDP = st.text_input('Jitter: DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP: Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP: Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer: APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer: APQ5')
        
    with col3:
        APQ = st.text_input('MDVP: APQ')
        
    with col4:
        DDA = st.text_input('Shimmer: DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        if not all([fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]):
            st.warning("Please fill in all the fields.")
        else:
            parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
            
            if (parkinsons_prediction[0] == 1):
              st.warning("The person has Parkinson's disease")
            else:
              st.success("The person does not have Parkinson's disease")
        


# Autism Prediction Page
if selected == 'Autism Prediction':
    st.title('Autism Prediction using ML')
    col1, col2, col3, col4 = st.columns(4)
    
    model1 = pickle.load(open('./models/logreg.pkl', 'rb'))

    def find_asd(res):
        if res == 1:
            return 'The person has Autism Spectrum Disorder'
        else:
            return 'The person does not have Autism Spectrum Disorder'
        
    ethnicities = ["Asian", "Black", "Hispanic", "Latino", "Middle Eastern", "Others", "Pasifika", "South Asian", "Turkish", "White-European"]
    relations = ["Health Care Professional", "Others", "Parent", "Relative", "Self"]
    genders = ["Female", "Male"]

    a1 = st.selectbox("A1 Score", [0, 1])
    a2 = st.selectbox("A2 Score", [0, 1])
    a3 = st.selectbox("A3 Score", [0, 1])
    a4 = st.selectbox("A4 Score", [0, 1])
    a5 = st.selectbox("A5 Score", [0, 1])
    a6 = st.selectbox("A6 Score", [0, 1])
    a7 = st.selectbox("A7 Score", [0, 1])
    a8 = st.selectbox("A8 Score", [0, 1])
    a9 = st.selectbox("A9 Score", [0, 1])
    a10 = st.selectbox("A10 Score", [0, 1])
    age = st.number_input("Age")
    gender = st.selectbox("Gender", genders)
    ethnicity = st.selectbox("Ethnicity", ethnicities)
    jaundice = st.selectbox("Jaundice", [0, 1])
    autism = st.selectbox("Autism", [0, 1])
    used_app_before = st.selectbox("Used App Before", [0, 1])
    result = st.number_input("Result")
    relation = st.selectbox("Relation", relations)

    if st.button("Autism Test Result"):
        gender = genders.index(gender)
        ethnicity = ethnicities.index(ethnicity)
        relation = relations.index(relation)
        test = np.array([[a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, age, gender, ethnicity, jaundice, autism, used_app_before, result, relation]])
        res1 = model1.predict(test)
        print(res1)
        result1 = find_asd(res1[0])
        st.success(" " + result1)


# Stroke Prediction Page
if selected == 'Stroke Prediction':

    st.title('Stroke Prediction using ML')
    model_stroke = pickle.load(open('./models/model_stroke.sav', 'rb'))
    col1, col2 = st.columns(2)

    with col1:
        gender_select = st.selectbox('Gender', options=['Female', 'Male'])
        gender = 1 if gender_select == 'Male' else 0
    
    with col2:
        age = st.number_input('Age', 5, 100)

    with col1:
        hypertension_select = st.selectbox('Hypertension', options=['No', 'Yes'])
        hypertension = 1 if hypertension_select == 'Yes' else 0

    with col2:
        heart_disease_select = st.selectbox('Heart Disease', options=['No', 'Yes'])
        heart_disease = 1 if heart_disease_select == 'Yes' else 0

    with col1:
        ever_married_select = st.selectbox('Ever Married', options=['No', 'Yes'])
        ever_married = 1 if ever_married_select == 'Yes' else 0

    with col2:
        residence_select = st.selectbox('Residence Type', options=['Urban', 'Rural'])
        Residence_type = 1 if residence_select == 'Urban' else 0

    with col1:
        avg_glucose_level = st.number_input('Glucose', min_value=50.0, max_value=300.0, step=0.1)
        
    with col2:
        bmi = st.number_input('Bmi',  min_value=10.0, max_value=100.0, step=0.1)

    with col1:
        smoking_status_select = st.selectbox('Smoking Status', options=['No', 'Yes', 'Unknown', 'formerly smoked'])
        smoking_status = 1 if smoking_status_select == 'Yes' else 0

    govt_job = 0
    never_worked = 0
    private = 0
    self_employed = 0
    children = 0

    with col2:
        selected_work = st.selectbox('Select Work:', options=['Government', 'Never Worked', 'Private', 'Self Employed', 'Children'])
        
    if selected_work == 'Government':
        govt_job = 1
    elif selected_work == 'Never Worked':
        never_worked = 1
    elif selected_work == 'Private':
        private = 1
    elif selected_work == 'Self Employed':
        self_employed = 1
    else:
        children = 1

    stroke_pred = None
    
    if st.button('Stroke Test Result'):
        stroke_pred = model_stroke.predict([[gender, age, hypertension, heart_disease, ever_married, Residence_type, avg_glucose_level, bmi, smoking_status, govt_job, never_worked, private, self_employed, children]])
    
    if stroke_pred is not None:
        if (stroke_pred)[0] == 1:
            stroke_diag = 'you had a stroke'
            st.error(stroke_diag)
        else:
            stroke_diag = "you didn't have a stroke"
            st.success(stroke_diag)


# Lung Cancer Prediction Page
if selected == 'Lung Cancer Prediction':
    st.title('Lung Cancer Prediction using ML')
    col1, col2, col3 = st.columns (3)
    
    with col1:
        GENDER = st.number_input('1 = Male, 2 = Female')
    with col1:
        AGE = st.number_input('Age')
    with col1:
        SMOKING = st.number_input('SMOKING ? 1 = NO, 2 = YES')
    with col1:
        YELLOW_FINGERS = st.number_input ('YELLOW FINGERS ? 1 = NO, 2 = YES')
    with col1:
        ANXIETY = st.number_input('ANXIETY ?  1 = NO, 2 = YES')
    with col2:
        PEER_PRESSURE = st.number_input ('PEER PRESSURE ? 1 = NO, 2 = YES')
    with col2:
        CHRONIC_DISEASE = st.number_input ( 'CHRONIC_DISEASE ? 1 = NO, 2 = YES')
    with col2:
        FATIGUE = st.number_input ('FATIGUE ? 1 = NO, 2 = YES')
    with col2:
        ALLERGY = st.number_input ('ALLERGY ? 1 = NO, 2 = YES')
    with col2:
        WHEEZING = st.number_input ('WHEEZING ? 1 = NO, 2 = YES')
    with col3:
        ALCOHOL_CONSUMING = st.number_input ('ALCOHOL CONSUMING ? 1 = NO, 2 = YES')
    with col3:
        COUGHING = st.number_input ('COUGHING ? 1 = NO, 2 = YES')
    with col3:
        SHORTNESS_OF_BREATH = st.number_input ('SHORTNESS OF BREATH ? 1 = NO, 2 = YES')
    with col3:
        SWALLOWING_DIFFICULTY = st.number_input ('SWALLOWING DIFFICULTY ? 1 = NO, 2 = YES')
    with col3:
        CHEST_PAIN = st.number_input ('CHEST PAIN ? 1 = NO, 2 = YES')
    
    if st.button('Lung Cancer Test Result'):
        cancer_prediction = lung_cancer_model.predict([[GENDER,AGE,SMOKING,YELLOW_FINGERS,ANXIETY,PEER_PRESSURE,CHRONIC_DISEASE,FATIGUE ,ALLERGY ,WHEEZING,ALCOHOL_CONSUMING,COUGHING,SHORTNESS_OF_BREATH,SWALLOWING_DIFFICULTY,CHEST_PAIN]])
        if (cancer_prediction [0]==1):
            st.warning('The patient does not have lung cancer.')
        else:
            st.success('The patient has lung cancer.')
        

# Covid-19 Prediction Page
if selected == 'Covid Prediction':
    st.title('Covid Prediction using ML')
    st.write("All The Values Should Be In Range Mentioned")
    drycough=st.number_input("Rate Of Dry Cough (0-20)",min_value=0,max_value=20,step=1)
    fever=st.number_input("Rate Of Fever (0-20)",min_value=0,max_value=20,step=1)
    sorethroat=st.number_input("Rate Of Sore Throat (0-20)",min_value=0,max_value=20,step=1)
    breathingprob=st.number_input("Rate Of Breathing Problem (0-20)",min_value=0,max_value=20,step=1)
    prediction1=model1.predict([[drycough,fever,sorethroat,breathingprob]])[0]

    if st.button("Covid-19 Test Result"):
        if prediction1=="Yes":
            st.warning("The Patient is Affected By Covid-19")
        elif prediction1=="No":
            st.success("The Patient is not Affected By Covid-19")

    st.image('covid-19-image.jpg',caption='Covid-19',use_column_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

# Data Collection
    df = pd.read_csv('./dataFiles/covid_19_clean_complete.csv')
    df.drop(columns=['Province/State','WHO Region'],inplace=True)
    df.rename(columns={'Country/Region':'Country'},inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])

    st.write("### Spread of Covid 19 Across the World")

# Create Density MApbox  
    fig = px.density_mapbox(df, lat="Lat", lon="Long",
                        hover_data=["Country", "Confirmed"],
                        z="Confirmed", radius=20, zoom=0,
                        range_color=[0, 1000], mapbox_style='carto-positron',
                        animation_frame='Date',
                        title="Spread of Covid-19",
                        )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig)


# Anemia Prediction Page
if selected == 'Anemia Prediction':
    st.title('Anemia Prediction using ML')
    anemia_model = pickle.load(open('./models/anemia_model.sav', 'rb'))
    Gender = st.text_input('Gender')
    Hemoglobin = st.text_input('Enter the amount of protein in red blood cells')
    MCH = st.text_input('Enter the average amount in each red blood cell')
    MCHC = st.text_input('Enter the average concentration of hemoglobin')
    MCV = st.text_input('Enter your average red blood cell')

    if st.button('Anemia Test Result'):
        ane_prediction = anemia_model.predict([[Gender, Hemoglobin, MCH, MCHC, MCV]])
        if ane_prediction[0] == 1:
            st.warning('The patient has anemia')
        else:
            st.success('The patient does not have anemia')
