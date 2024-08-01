import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app title

st.title("Prediction of Price:")

# st.write('### Prediction of Price:')
# Load your trained model once at startup
data_loaded = joblib.load('model_with_data.pkl')
# Extract the model and dataset
model = data_loaded['model']

st.write("Model loaded successfully")
key = False
# option_diagram = "Living Area"
# Create tabs

tab1, tab2, tab3 = st.tabs(["Input Form", "Summary", "Diagram"])
# Input form in Tab 1
with tab1:
    
# Create columns
    col1, col2 = st.columns(2)

    # Input fields in Column 1
    with col1:
        st.header("Property Details")
        BedroomCount = st.number_input('Bedroom Count', min_value=1, step=1)
        LivingArea = st.number_input('Living Area (sq meters)', min_value=200, step=50)
        BathroomCount = st.number_input('Bathroom Count', min_value=1)
        # StateOfBuilding_Encoded = st.number_input('State of Building Encoded (1-3)', min_value=1, step=1, max_value=3)
        Var_Encoded = st.selectbox('Select the state of the building',
                                options=["Good", "Average", "Not good"])
        if Var_Encoded == 'Good':
            StateOfBuilding_Encoded = 1
        elif Var_Encoded == 'Average':
            StateOfBuilding_Encoded = 2
        elif Var_Encoded == 'Not good':
            StateOfBuilding_Encoded = 3
        SurfaceOfPlot = st.number_input('Surface Of Plot (sq meters)', min_value=0, step=100)
        
    with col2:
        st.header("Additional Features")
        option_pool = st.radio('Choose Swimming Pool:', ('Yes', 'No'))
        SwimmingPool = 1 if option_pool == "Yes" else 0

        option_property = st.radio('Choose Type of Property:', ('Residential', 'Commercial'))
        TypeOfProperty = 1 if option_property == "Residential" else 2

        option_terrace = st.radio('Choose Terrace:', ('Yes', 'No'))
        Terrace = 1 if option_terrace == "Yes" else 0
        
        option_region = st.radio('Choose Region:', ('Flanders', 'Wallonie', 'Brussels'))
        if option_region == "Flanders":
            Region_Encoded = 1
        elif option_region == "Wallonie":
            Region_Encoded = 2
        else:
            Region_Encoded = 3


    # Create another row of columns for Region and counts
    col3, col4 = st.columns(2)

    with col3:
        MonthlyCharges = st.number_input('Monthly Charges (currency)', min_value=100, step=50)
        RoomCount = st.number_input('Room Count', min_value=1, step=1)

    with col4:
        st.header("Type of diagram")
        option_diagram = st.radio('Choose diagram:', ('Living Area', 'Count of Bedroom'))
        
        # else:
            # Region_Encoded = 3
    # Button to Submit and show summary
    if st.button("Submit"):
        key = True
        # Show Summary in Tab 2
        # Prepare input data for prediction

        input_data = np.asarray([[BedroomCount, LivingArea, BathroomCount, StateOfBuilding_Encoded, 
                   SwimmingPool, RoomCount, TypeOfProperty, Terrace, SurfaceOfPlot, Region_Encoded,
                   MonthlyCharges]])
        prediction = model.predict(input_data)

# Summary tab initially empty
with tab2:
    if key == False:
        st.header("Input Summary")
        st.write("Please fill out the form above and click Submit to see the summary.")
    elif key:
        st.header("Input Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Bedroom Count:** {BedroomCount}")
            st.write(f"**Living Area:** {LivingArea} sq meters")
            st.write(f"**Bathroom Count:** {BathroomCount}")
            st.write(f"**State of Building Encoded:** {Var_Encoded}")
            st.write(f"**Swimming Pool:** {'Yes' if SwimmingPool == 1 else 'No'}")
            st.write(f"**Type of Property:** {'Residential' if TypeOfProperty == 1 else 'Commercial'}")
        with col2:
            st.write(f"**Terrace:** {'Yes' if Terrace == 1 else 'No'}")
            st.write(f"**Region:** {option_region}")
            st.write(f"**Room Count:** {RoomCount}")
            st.write(f"**Surface Of Plot:** {SurfaceOfPlot} sq meters")
            st.write(f"**Monthly Charges:** {MonthlyCharges} Euro")
        st.write('### Prediction:')
        st.write(f'**Predicted Value:** {int(prediction[0])} Euro')

with tab3:
    if key == False:
        st.header("Diagram Living Area & Price")
        st.write("Please fill out the form above and click Submit to see the diagram.")
    elif key:
               
        X = data_loaded['dataset']  # Your DataFrame

        # Predict prices
        predicted_prices = model.predict(X)

        # living_area = X['LivingArea']  # Replace if your DataFrame structure is different
        
        
        # Create a line plot for predictions 
        plt.figure(figsize=(10, 6))
        if option_diagram == "Living Area":
            st.header("Diagram Living Area & Price")
            df_pred = pd.DataFrame({'LivingArea': X[:, 1], 'Predicted Price': predicted_prices})
            scatter_plot = sns.lmplot(x='LivingArea', y='Predicted Price', data=df_pred, aspect=2, height=6) 
            plt.scatter(LivingArea, int(prediction[0]), color='yellow', marker= '*',s=100, zorder=5, label='Our price')
            plt.xlabel('Living Area')

        elif option_diagram == "Count of Bedroom":
            st.header("Diagram Count of Bedroom & Price")
            df_pred = pd.DataFrame({'BedroomCount': X[:, 0], 'Predicted Price': predicted_prices})
            scatter_plot = sns.lmplot(x='BedroomCount', y='Predicted Price', data=df_pred, aspect=2, height=6)
            plt.scatter(BedroomCount, int(prediction[0]), color='yellow', marker= '*', s=100, zorder=5, label='Our price')
            plt.xlabel('BedroomCount')

        plt.gca().lines[0].set_color('red')
        plt.ylabel('Predicted Price')
        plt.grid()

        # Show the plot
        st.pyplot(plt)
        
        