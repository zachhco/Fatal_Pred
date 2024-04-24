import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.markdown("# Fatality Predictor - Classification Model")
st.sidebar.markdown("# Fatality Predictor - Classification Model")
st.sidebar.markdown("Interactive Machine Learning App 🤖")
st.sidebar.markdown("Interactive crash risk prediction tool which, given a set of user inputs, predicts the wether a fatal crash will occur.")
st.sidebar.markdown("The underlying model is a random forest regressor trained on the NHTSA's FARS dataset. The model uses a variety of\
                     features to predict the a fatal crash.")


### READ DATA ###
df = pd.read_csv('data_GEN6.csv')

# Separate target from predictors
y = df['FATAL']
X = df.drop(['FATAL', 'INJ_SEV', 'INJ_LEVEL'], axis=1)

# Train/Test Split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Create the RandomForestRegressor model
if 'model' not in st.session_state:
    st.session_state['model'] = RandomForestClassifier(n_estimators=476,
                                                       min_samples_split=2,
                                                       bootstrap=True,
                                                       max_depth=50,
                                                       n_jobs=-1,
                                                       min_samples_leaf=1,
                                                       oob_score=True,
                                                       random_state=42)

    #fit model to data
    st.session_state['model'].fit(X_train, y_train)

    #make predictions
    old_preds = st.session_state['model'].predict(X_valid)


# Preprocess inputs so that categorical variables are encoded into binary
def preprocess_inputs(alcohol_use, age, sex, model_year, height, weight, speed_limit, rush_hour, light_condition,
              restraint_use, drug_use, cold_weather, speeding, license_status, airbag_deploy, driver,
              front_seat, collision, ejection, large_size):

    # Use one-hot encoding for binary variables
    alcohol_use_one_hot = {'Yes': 1, 'No': 0}[alcohol_use]
    sex_one_hot = {'Male': 1, 'Female': 0}[sex]
    rush_hour_one_hot = {'Yes': 1, 'No': 0}[rush_hour]
    light_condition_one_hot = {'Daylight': 1, 'Night': 0}[light_condition]
    restraint_use_one_hot = {'Yes': 1, 'No': 0}[restraint_use]
    drug_use_one_hot = {'Yes': 1, 'No': 0}[drug_use]
    cold_weather_one_hot = {'Yes': 1, 'No': 0}[cold_weather]
    speeding_one_hot = {'Yes': 1, 'No': 0}[speeding]
    license_status_one_hot = {'Valid': 1, 'Invalid': 0}[license_status]
    airbag_deploy_one_hot = {'Yes': 1, 'No': 0}[airbag_deploy]
    driver_one_hot = {'Yes': 1, 'No': 0}[driver]
    front_seat_one_hot = {'Yes': 1, 'No': 0}[front_seat]
    collision_one_hot = {'Yes': 1, 'No': 0}[collision]
    ejection_one_hot = {'Yes': 1, 'No': 0}[ejection]
    large_size_one_hot = {'Yes': 1, 'No': 0}[large_size]
    
    # Combine all inputs into a single flat list
    return [alcohol_use_one_hot, age, sex_one_hot, model_year, height, weight, speed_limit,
        rush_hour_one_hot, light_condition_one_hot, restraint_use_one_hot, drug_use_one_hot,
        cold_weather_one_hot, speeding_one_hot, license_status_one_hot, airbag_deploy_one_hot,
        driver_one_hot, front_seat_one_hot, collision_one_hot, ejection_one_hot, large_size_one_hot]

# Specify col size ratios
left_col, spacer, right_col = st.columns([1, 0.5, 1])

# Create a left col for user inputs
with left_col:
    st.markdown('## <span style="color:green">User Inputs</span>', unsafe_allow_html=True)
    st.write('Enter in feature values and/or model hyperparamaters to train the model. \
             Without entering any values, the model will use the default hyperparameters and mean feature values.')
    
    # Dropdown for feature inputs
    with st.expander("Enter Feature Values"):
        restraint_use = st.selectbox('Restraint Use', ['Yes', 'No'], index=0)
        speeding = st.selectbox('Speeding', ['Yes', 'No'], index=1)
        age = st.number_input('Age', value=39, min_value=0)
        sex = st.selectbox('Sex', ['Male', 'Female'], index=0)
        height = st.number_input('Height (inches)', min_value=0, value=68)
        weight = st.number_input('Weight (lbs)', min_value=0, value=184)
        speed_limit = st.number_input('Speed Limit', min_value=0, value=50)
        rush_hour = st.selectbox('Rush Hour', ['Yes', 'No'], index=1)
        light_condition = st.selectbox('Light Condition', ['Daylight', 'Night'], index=0)
        cold_weather = st.selectbox('Cold Weather (temp < 32 degrees)', ['Yes', 'No'], index=1)
        model_year = st.number_input('Model Year', min_value=1900, value=2008)
        license_status = st.selectbox('License Status', ['Valid', 'Invalid'], index=0)
        driver = st.selectbox('Driver (driver of a car)', ['Yes', 'No'], index=0)
        front_seat = st.selectbox('Front Seat (in the front seat of a car)', ['Yes', 'No'], index=0)
        large_size = st.selectbox('Large Size', ['Yes', 'No'], index=1)
        
    with st.expander("Additional Features (Optional)"):
        airbag_deploy = st.selectbox('Airbag Deploy', ['Yes', 'No'], index=1)
        ejection = st.selectbox('Ejection (ejected from a car)', ['Yes', 'No'], index=1)
        collision = st.selectbox('Collision (collision with another car)', ['Yes', 'No'], index=1)
        alc_use = st.selectbox('Alcohol Use', ['Yes', 'No'], index=1)
        drug_use = st.selectbox('Drug Use', ['Yes', 'No'], index=1)

    # Dropdown for hyperparameter inputs
    with st.expander("Enter Hyperparameters"):
        n_estimators2 = st.slider("Enter a number for the number of estimators:", 1, 800, value=476, key='estims')
        min_samples_split2 = st.slider("Enter a number for the minimum samples split:", 2, 10, value=2, key='min_samples_split')
        #bootstrap = st.selectbox("Select whether to use bootstrap:", [True, False], key='bootstrap')
        max_depth2 = st.slider("Enter a number for the maximum depth:", 1, 100, value=50, key='max_depth')
        min_samples_leaf2 = st.slider("Enter a number for the minimum samples leaf:", 1, 10, value=1, key='min_samples_leaf')
        random_state2 = st.slider("Enter a number for the random state:", 0, 100, value=42, key='random_state')

        if st.button('Train Model'):
            # Update the model in the session state with new hyperparameters
            st.session_state['model'] = RandomForestClassifier(
                n_estimators=n_estimators2,
                min_samples_split=min_samples_split2,
                bootstrap=True,
                max_depth=max_depth2,
                n_jobs=-1,
                min_samples_leaf=min_samples_leaf2,
                oob_score=True,
                random_state=random_state2
            )
            # Fit model to data
            st.session_state['model'].fit(X_train, y_train)
            # Display the model's current hyperparameters
            # st.write("Model retrained with the following hyperparameters:")
            # st.write(f"- n_estimators: {n_estimators2}")
            # st.write(f"- min_samples_split: {min_samples_split2}")
            # st.write(f"- max_depth: {max_depth2}")
            # st.write(f"- min_samples_leaf: {min_samples_leaf2}")
            # st.write(f"- random_state: {random_state2}")
            st.success('Model retrained with the new hyperparameters.')

            inputs = preprocess_inputs(alc_use, age, sex, model_year, height, weight, speed_limit, rush_hour,
                                    light_condition, restraint_use, drug_use, cold_weather, speeding, license_status,
                                    airbag_deploy, driver, front_seat, collision, ejection, large_size)
            
        # make a reset to default button
        if st.button('Reset to Default'):
            # Update the model in the session state with new hyperparameters
            st.session_state['model'] = RandomForestClassifier(
                n_estimators=476,
                min_samples_split=2,
                bootstrap=True,
                max_depth=50,
                n_jobs=-1,
                min_samples_leaf=1,
                oob_score=True,
                random_state=42
            )
            # Fit model to data
            st.session_state['model'].fit(X_train, y_train)
            st.success('Model retrained with the default hyperparameters.')

            inputs = preprocess_inputs


if 'accuracy_scores' not in st.session_state:
    st.session_state['accuracy_scores'] = []

# Create a right col for model reslts
with right_col:
    st.markdown('## <span style="color:green">Model Results</span>', unsafe_allow_html=True)
    st.write('Push Button Below to Get Model Results')
    # Button to make prediction
    if st.button('Predict'):
        st.success('Model successfully made predictions')
        # Preprocess the inputs
        inputs = preprocess_inputs(alc_use, age, sex, model_year, height, weight, speed_limit, rush_hour,
                                light_condition, restraint_use, drug_use, cold_weather, speeding, license_status,
                                airbag_deploy, driver, front_seat, collision, ejection, large_size)
        
        # Make prediction
        prediction = st.session_state['model'].predict([inputs])[0]

        # Display the prediction
        st.markdown('The model Predicts: **{}**'.format('Death' if prediction == 1 else 'No Death'))

        # Recalculate accuracy with the updated model
        new_preds = st.session_state['model'].predict(X_valid)
        score = accuracy_score(y_valid, new_preds)
        st.markdown("The model's accuracy score is **{:.4f}**".format(score))

        # Display the corresponding image
        image_path = 'death.png' if prediction == 1 else 'life.png'
        st.image(image_path)

         # Append the new score to the accuracy list
        st.session_state['accuracy_scores'].append(score)

        # Now plot the accuracy scores
        fig, ax = plt.subplots()
        ax.plot(range(len(st.session_state['accuracy_scores'])), st.session_state['accuracy_scores'], marker='o')
        ax.set_xlabel('Run Number')
        ax.set_ylabel('Accuracy Score')
        ax.set_title('Accuracy Score Over Multiple Runs')
        st.pyplot(fig)

        # display the value for min_samples_split the the model used
        st.write("Model Hyperparameters:")
        hyperparameters = {
            'n_estimators': st.session_state['model'].n_estimators,
            'min_samples_split': st.session_state['model'].min_samples_split,
            'max_depth': st.session_state['model'].max_depth,
            'min_samples_leaf': st.session_state['model'].min_samples_leaf,
            'bootstrap': st.session_state['model'].bootstrap,
            'random_state': st.session_state['model'].random_state
        }

        df = pd.DataFrame(hyperparameters.items(), columns=['Hyperparameter', 'Value'])
        st.write(df)
