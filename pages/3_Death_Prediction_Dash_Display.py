import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.title('Death Prediction Dash Display')
st.write("The Death Prediction Dash Display is a Streamlit app that showcases a prototype idea for a car display that broadcasts an active reading of a \
          driver's probability of death in a car accident. The dashboard would hypothetically be trained on a robust random forest model, which takes live \
         data input from sensors within the car for things such as speed and road conditions.")

st.write("The dashboard below is a random forest model trained using a US DOT dataset. The model below substitutes live inputs from car sensors with user \
         inputs for feature values. When the predictions are made, the probability of death is projected onto the dashboard display. The intended product \
         will showcase a continuous probability reading, which always adjusts to driving conditions. In contrast, this app is a stationary representation \
         where predictions are only made when the button is pressed.")

#st.set_page_config(layout="wide")

### READ DATA ###
df = pd.read_csv('data_GEN6.csv')

# Separate target from predictors
y = df['FATAL']
X = df.drop(['FATAL', 'INJ_SEV', 'INJ_LEVEL'], axis=1)

# Train/Test Split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestRegressor

# Initialize the model
if 'model' not in st.session_state:
    st.session_state['model'] = RandomForestRegressor(n_estimators=476,
                                                      min_samples_split=2,
                                                      bootstrap=True,
                                                      max_depth=50,
                                                      n_jobs=-1,
                                                      min_samples_leaf=1,
                                                      random_state=42)

    # Fit model to data
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


st.markdown('## <span style="color:green">User Inputs</span>', unsafe_allow_html=True)
st.write('Enter in driver, car, and road information to generate a prediction. \
            Without entering any values, the model will use the default hyperparameters and mean feature values.')

# Dropdown for feature inputs'
with st.expander("Driving Conditions"):
    restraint_use = st.selectbox('Restraint Use', ['Yes', 'No'], index=0)
    speeding = st.selectbox('Speeding', ['Yes', 'No'], index=1)
    rush_hour = st.selectbox('Rush Hour', ['Yes', 'No'], index=1)
    light_condition = st.selectbox('Light Condition', ['Daylight', 'Night'], index=0)
    cold_weather = st.selectbox('Cold Weather (temp < 32 degrees)', ['Yes', 'No'], index=1)

st.markdown(' <span style="color:green">Other Inputs</span>', unsafe_allow_html=True)
with st.expander("Additional Features (Optional)"):
    airbag_deploy = st.selectbox('Airbag Deploy', ['Yes', 'No'], index=1)
    ejection = st.selectbox('Ejection (ejected from a car)', ['Yes', 'No'], index=1)
    collision = st.selectbox('Collision (collision with another car)', ['Yes', 'No'], index=1)
    alc_use = st.selectbox('Alcohol Use', ['Yes', 'No'], index=1)
    drug_use = st.selectbox('Drug Use', ['Yes', 'No'], index=1)
    age = st.number_input('Age', value=39, min_value=0)
    sex = st.selectbox('Sex', ['Male', 'Female'], index=0)
    height = st.number_input('Height (inches)', min_value=0, value=68)
    weight = st.number_input('Weight (lbs)', min_value=0, value=184)
    speed_limit = st.number_input('Speed Limit', min_value=0, value=50)
    model_year = st.number_input('Model Year', min_value=1900, value=2008)
    license_status = st.selectbox('License Status', ['Valid', 'Invalid'], index=0)
    driver = st.selectbox('Driver (driver of a car)', ['Yes', 'No'], index=0)
    front_seat = st.selectbox('Front Seat (in the front seat of a car)', ['Yes', 'No'], index=0)
    large_size = st.selectbox('Large Size', ['Yes', 'No'], index=1)

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
        st.session_state['model'] = RandomForestRegressor(
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
        st.session_state['model'] = RandomForestRegressor(
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

st.write('Push button to show active death probability prediction automation ')
# Button to make prediction
# Prediction button and handling
if st.button('Predict'):
    inputs = preprocess_inputs(alc_use, age, sex, model_year, height, weight, speed_limit, rush_hour,
                            light_condition, restraint_use, drug_use, cold_weather, speeding, license_status,
                            airbag_deploy, driver, front_seat, collision, ejection, large_size)

    # Make prediction
    probability_of_death = st.session_state['model'].predict([inputs])[0]  # Predict now returns a continuous value
    new_prob = probability_of_death * 100

    threshold = 0.5  # Define a threshold to decide on 'Death' vs 'No Death'
    prediction_category = 'Death' if probability_of_death >= threshold else 'No Death'
    # Display the corresponding image
    imsage_path = 'death.png' if prediction_category == 'Death' else 'life.png'

    ########
    ########

    ##  DYNAMIC CAR ANIMATION  ##

    ########
    ########
        
    # Get the selected number using a slider
    number = f"{new_prob:.0f}"

    # load the image
    if prediction_category == 'Death':
        image_path = "red.png"
    else:
        image_path = "green.png"
    image = Image.open(image_path)

    # Path to the font file
    font_path = "segoe-ui-bold.ttf"

    # Prepare to draw on the image
    draw = ImageDraw.Draw(image)

    try:
        # Try using a specific font from the given path
        font = ImageFont.truetype(font_path, size=50)
    except IOError:
        # Fallback to the default font if the specific one is not found
        font = ImageFont.load_default()

    # Position for the text (adjust as needed)
    text_position = (955, 417)

    # Draw the text on the image
    draw.text(text_position, f"{number}%", font=font, fill=(255, 255, 255))

    # Save the image to a buffer
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)

    # Display the image in Streamlit, directly using the buffer
    st.image(buffer, use_column_width=True)
