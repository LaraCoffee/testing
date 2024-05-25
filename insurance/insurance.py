import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
import plotly.express as px


csv_file_path = "insurance/insurance.csv"

#Read the CSV file
df = pd.read_csv(csv_file_path)




X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]  # Features
X = pd.get_dummies(X, columns=['sex', 'smoker', 'region'], drop_first=True)  # Convert categorical variables to dummy variables
y = df['charges']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the regression model
model = LinearRegression()
model.fit(X_train, y_train)


st.title('Custos com plano de saude')


st.sidebar.header('parametros')
inputs = {}
inputs['age'] = st.sidebar.slider('Age', float(X['age'].min()), float(X['age'].max()), float(X['age'].mean()))
inputs['bmi'] = st.sidebar.slider('BMI', float(X['bmi'].min()), float(X['bmi'].max()), float(X['bmi'].mean()))
inputs['children'] = st.sidebar.slider('Number of Children', float(X['children'].min()), float(X['children'].max()), float(X['children'].mean()))
inputs['sex_male'] = st.sidebar.selectbox('Sex', ['Male', 'Female'], index=0)
inputs['smoker_yes'] = st.sidebar.selectbox('Smoker', ['Yes', 'No'], index=1)
inputs['region_northwest'] = st.sidebar.selectbox('Region', ['Northwest', 'Northeast', 'Southwest', 'Southeast'], index=0)

# Convert selected values to appropriate format
inputs['sex_male'] = 1 if inputs['sex_male'] == 'Male' else 0
inputs['smoker_yes'] = 1 if inputs['smoker_yes'] == 'Yes' else 0
inputs['region_' + inputs['region_northwest'].lower()] = 1
del inputs['region_northwest']

# Ensure dummy variable columns are consistent
missing_cols = set(X_train.columns) - set(inputs.keys())
for col in missing_cols:
    inputs[col] = 0

# Reorder columns in the input data frame
inputs_df = pd.DataFrame([inputs], columns=X_train.columns)

# Predict insurance cost based on user input
prediction = model.predict(inputs_df)

# Display the prediction
st.write('previsão de custos com seguro:', round(prediction[0], 2))

# Visualizations
# Scatter plots to visualize relationship between numerical features and charges
fig1 = px.scatter(df, x='age', y='charges', trendline='ols', title='Idade vs. Custos')
fig2 = px.scatter(df, x='bmi', y='charges', trendline='ols', title='IMC vs. Custos')
fig3 = px.scatter(df, x='children', y='charges', trendline='ols', title='Numero de meninu vs. Custos')

# Display the visualizations
st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)
