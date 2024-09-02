# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score

# # Path to the local CSV file
# file_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

# # Load data
# df = pd.read_csv(file_path)
# df = df.dropna()  # Drop rows with missing values

# # Encode categorical columns
# label_encoders = {}
# for column in df.select_dtypes(include=['object']).columns:
#     le = LabelEncoder()
#     df[column] = le.fit_transform(df[column])
#     label_encoders[column] = le

# # Split data into features and target
# X = df.drop(['Churn'], axis=1)
# y = df['Churn']

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Initialize models
# logistic_model = LogisticRegression(max_iter=1000)
# random_forest_model = RandomForestClassifier()

# # Train models
# logistic_model.fit(X_train, y_train)
# random_forest_model.fit(X_train, y_train)

# # Make predictions
# logistic_preds = logistic_model.predict(X_test)
# random_forest_preds = random_forest_model.predict(X_test)

# # Compute metrics
# logistic_report = classification_report(y_test, logistic_preds, output_dict=True)
# random_forest_report = classification_report(y_test, random_forest_preds, output_dict=True)

# logistic_accuracy = accuracy_score(y_test, logistic_preds)
# random_forest_accuracy = accuracy_score(y_test, random_forest_preds)

# # Streamlit app layout
# st.title("Customer Churn Prediction - EDA Dashboard")

# # Summary Statistics
# st.header("Summary Statistics")
# summary_stats = df.describe(include='all').transpose()
# st.write(summary_stats)

# # Plot Summary Statistics
# fig_summary = px.bar(summary_stats, x=summary_stats.index, y='mean', title='Summary Statistics', labels={'mean': 'Average Value'})
# st.plotly_chart(fig_summary)

# # Correlation Heatmap for all features
# st.header("Correlation Heatmap")
# corr = df.corr()
# fig_heatmap = go.Figure(data=go.Heatmap(z=corr.values,
#                                        x=corr.columns,
#                                        y=corr.columns,
#                                        colorscale='Viridis'))
# fig_heatmap.update_layout(title='Correlation Heatmap')
# st.plotly_chart(fig_heatmap)

# # Customer Behavior Analysis: Churn Distribution
# st.header("Churn Distribution")
# churn_dist = df['Churn'].value_counts().reset_index()
# churn_dist.columns = ['Churn', 'Count']
# fig_churn = px.pie(churn_dist, names='Churn', values='Count', title='Churn Distribution')
# st.plotly_chart(fig_churn)

# # Distribution of Monthly Charges by Churn
# st.header("Monthly Charges Distribution by Churn")
# fig_charges = px.box(df, x='Churn', y='MonthlyCharges', title='Monthly Charges Distribution by Churn')
# st.plotly_chart(fig_charges)

# # Distribution of Tenure by Churn
# st.header("Tenure Distribution by Churn")
# fig_tenure = px.box(df, x='Churn', y='tenure', title='Tenure Distribution by Churn')
# st.plotly_chart(fig_tenure)

# st.header("Churn Rate by Internet Service")
# # Handle missing or unexpected values
# if 'Churn' in df.columns and 'InternetService' in df.columns:
#     churn_internet_service = df.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack().fillna(0)
#     churn_internet_service['Churn Rate'] = churn_internet_service.get(1, 0) / (churn_internet_service.get(1, 0) + churn_internet_service.get(0, 0))
#     fig_internet_service = px.bar(churn_internet_service, x=churn_internet_service.index, y='Churn Rate', title='Churn Rate by Internet Service')
#     st.plotly_chart(fig_internet_service)
# else:
#     st.write("Columns 'Churn' or 'InternetService' are missing in the dataset.")

# # Churn Rate by Contract Type
# st.header("Churn Rate by Contract Type")
# if 'Churn' in df.columns and 'Contract' in df.columns:
#     churn_contract = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack().fillna(0)
#     churn_contract['Churn Rate'] = churn_contract.get(1, 0) / (churn_contract.get(1, 0) + churn_contract.get(0, 0))
#     fig_contract = px.bar(churn_contract, x=churn_contract.index, y='Churn Rate', title='Churn Rate by Contract Type')
#     st.plotly_chart(fig_contract)
#     st.write("The bar chart shows the churn rate for different contract types. It helps in understanding if contract length influences customer retention.")

# # Monthly Charges vs. Total Charges
# st.header("Monthly Charges vs. Total Charges")
# fig_charges_total = px.scatter(df, x='MonthlyCharges', y='TotalCharges', color='Churn', title='Monthly Charges vs. Total Charges')
# st.plotly_chart(fig_charges_total)
# st.write("The scatter plot illustrates the relationship between monthly charges and total charges, colored by churn status. It helps to identify any patterns or clusters in charge-related features.")

# # Distribution of Customer Age
# st.header("Distribution of Customer Age")
# # Assuming age is not directly available, you can use tenure as a proxy if age is not available.
# fig_age, ax_age = plt.subplots()
# ax_age.hist(df['tenure'], bins=20, color='skyblue', edgecolor='black')
# ax_age.set_title('Distribution of Customer Tenure')
# ax_age.set_xlabel('Tenure')
# ax_age.set_ylabel('Frequency')
# st.pyplot(fig_age)
# st.write("The histogram shows the distribution of customer tenure, which can serve as a proxy for customer age. It provides insight into the typical duration customers stay with the company.")

# # Display model results
# st.header("Model Performance")

# log_model = LogisticRegression()
# log_model.fit(X_train, y_train)
# log_pred = log_model.predict(X_test)
# st.header("Logistic Regression Results")
# log_report = classification_report(y_test, log_pred, output_dict=True)
# st.write("**Precision, Recall, and F1-Score for Logistic Regression:**")
# st.write(pd.DataFrame(log_report).transpose())

# log_accuracy = accuracy_score(y_test, log_pred)
# st.write("**Accuracy:**", log_accuracy)

# # Random Forest
# rf_model = RandomForestClassifier()
# rf_model.fit(X_train, y_train)
# rf_pred = rf_model.predict(X_test)
# st.header("Random Forest Results")
# rf_report = classification_report(y_test, rf_pred, output_dict=True)
# st.write("**Precision, Recall, and F1-Score for Random Forest:**")
# st.write(pd.DataFrame(rf_report).transpose())

# rf_accuracy = accuracy_score(y_test, rf_pred)
# st.write("**Accuracy:**", rf_accuracy)

# importances = rf_model.feature_importances_
# indices = np.argsort(importances)[::-1]
# features = X.columns

# fig, ax = plt.subplots(figsize=(12, 6))

# # Plotting
# ax.set_title('Feature Importances')
# ax.bar(range(X.shape[1]), importances[indices], align='center')
# ax.set_xticks(range(X.shape[1]))
# ax.set_xticklabels(features[indices], rotation=90)
# ax.set_xlim([-1, X.shape[1]])

# # Display the plot with Streamlit
# st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Path to the local CSV file
file_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

# Load data
df = pd.read_csv(file_path)
df = df.dropna()  # Drop rows with missing values

# Encode categorical columns
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split data into features and target
X = df.drop(['Churn'], axis=1)
y = df['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize models
logistic_model = LogisticRegression(max_iter=1000)
random_forest_model = RandomForestClassifier()

# Train models
logistic_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

# Make predictions
logistic_preds = logistic_model.predict(X_test)
random_forest_preds = random_forest_model.predict(X_test)

# Compute metrics
logistic_report = classification_report(y_test, logistic_preds, output_dict=True)
random_forest_report = classification_report(y_test, random_forest_preds, output_dict=True)

logistic_accuracy = accuracy_score(y_test, logistic_preds)
random_forest_accuracy = accuracy_score(y_test, random_forest_preds)

# Streamlit app layout
st.title("Customer Churn Prediction - EDA Dashboard")

# Summary Statistics
st.header("Summary Statistics")
summary_stats = df.describe(include='all').transpose()
st.write(summary_stats)

# Plot Summary Statistics
fig_summary = px.bar(summary_stats, x=summary_stats.index, y='mean', title='Summary Statistics', labels={'mean': 'Average Value'}, color_discrete_sequence=['#1f77b4'])
st.plotly_chart(fig_summary)

# Correlation Heatmap for all features
st.header("Correlation Heatmap")
corr = df.corr()
fig_heatmap = go.Figure(data=go.Heatmap(z=corr.values,
                                       x=corr.columns,
                                       y=corr.columns,
                                       colorscale='Viridis'))
fig_heatmap.update_layout(title='Correlation Heatmap', title_font_size=20, xaxis_title='', yaxis_title='', xaxis=dict(tickvals=list(range(len(corr.columns))), ticktext=corr.columns), yaxis=dict(tickvals=list(range(len(corr.columns))), ticktext=corr.columns))
st.plotly_chart(fig_heatmap)

# Customer Behavior Analysis: Churn Distribution
st.header("Churn Distribution")
churn_dist = df['Churn'].value_counts().reset_index()
churn_dist.columns = ['Churn', 'Count']
fig_churn = px.pie(churn_dist, names='Churn', values='Count', title='Churn Distribution', color_discrete_sequence=['#1f77b4', '#ff7f0e'])
st.plotly_chart(fig_churn)

# Distribution of Monthly Charges by Churn
st.header("Monthly Charges Distribution by Churn")
fig_charges = px.box(df, x='Churn', y='MonthlyCharges', title='Monthly Charges Distribution by Churn', color='Churn', color_discrete_sequence=['#1f77b4', '#ff7f0e'])
st.plotly_chart(fig_charges)

# Distribution of Tenure by Churn
st.header("Tenure Distribution by Churn")
fig_tenure = px.box(df, x='Churn', y='tenure', title='Tenure Distribution by Churn', color='Churn', color_discrete_sequence=['#1f77b4', '#ff7f0e'])
st.plotly_chart(fig_tenure)

st.header("Churn Rate by Internet Service")
if 'Churn' in df.columns and 'InternetService' in df.columns:
    churn_internet_service = df.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack().fillna(0)
    churn_internet_service['Churn Rate'] = churn_internet_service.get(1, 0) / (churn_internet_service.get(1, 0) + churn_internet_service.get(0, 0))
    fig_internet_service = px.bar(churn_internet_service, x=churn_internet_service.index, y='Churn Rate', title='Churn Rate by Internet Service', color_discrete_sequence=['#1f77b4'])
    st.plotly_chart(fig_internet_service)
else:
    st.write("Columns 'Churn' or 'InternetService' are missing in the dataset.")

# Churn Rate by Contract Type
st.header("Churn Rate by Contract Type")
if 'Churn' in df.columns and 'Contract' in df.columns:
    churn_contract = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack().fillna(0)
    churn_contract['Churn Rate'] = churn_contract.get(1, 0) / (churn_contract.get(1, 0) + churn_contract.get(0, 0))
    fig_contract = px.bar(churn_contract, x=churn_contract.index, y='Churn Rate', title='Churn Rate by Contract Type', color_discrete_sequence=['#1f77b4'])
    st.plotly_chart(fig_contract)
    st.write("The bar chart shows the churn rate for different contract types. It helps in understanding if contract length influences customer retention.")

# Monthly Charges vs. Total Charges
st.header("Monthly Charges vs. Total Charges")
fig_charges_total = px.scatter(df, x='MonthlyCharges', y='TotalCharges', color='Churn', title='Monthly Charges vs. Total Charges', color_discrete_sequence=['#1f77b4', '#ff7f0e'])
st.plotly_chart(fig_charges_total)
st.write("The scatter plot illustrates the relationship between monthly charges and total charges, colored by churn status. It helps to identify any patterns or clusters in charge-related features.")

# Distribution of Customer Age
st.header("Distribution of Customer Age")
fig_age, ax_age = plt.subplots(figsize=(10, 6))
ax_age.hist(df['tenure'], bins=20, color='#1f77b4', edgecolor='black')
ax_age.set_title('Distribution of Customer Tenure')
ax_age.set_xlabel('Tenure')
ax_age.set_ylabel('Frequency')
st.pyplot(fig_age)
st.write("The histogram shows the distribution of customer tenure, which can serve as a proxy for customer age. It provides insight into the typical duration customers stay with the company.")

# Display model results
st.header("Model Performance")

# Logistic Regression Results
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
st.header("Logistic Regression Results")
log_report = classification_report(y_test, log_pred, output_dict=True)
st.write("**Precision, Recall, and F1-Score for Logistic Regression:**")
st.write(pd.DataFrame(log_report).transpose())

log_accuracy = accuracy_score(y_test, log_pred)
st.write("**Accuracy:**", log_accuracy)

# Random Forest Results
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
st.header("Random Forest Results")
rf_report = classification_report(y_test, rf_pred, output_dict=True)
st.write("**Precision, Recall, and F1-Score for Random Forest:**")
st.write(pd.DataFrame(rf_report).transpose())

rf_accuracy = accuracy_score(y_test, rf_pred)
st.write("**Accuracy:**", rf_accuracy)

# Feature Importances
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

fig_importances, ax_importances = plt.subplots(figsize=(12, 6))
ax_importances.bar(range(X.shape[1]), importances[indices], align='center', color='#1f77b4')
ax_importances.set_title('Feature Importances')
ax_importances.set_xticks(range(X.shape[1]))
ax_importances.set_xticklabels(features[indices], rotation=90)
ax_importances.set_xlim([-1, X.shape[1]])

# Display the plot with Streamlit
st.pyplot(fig_importances)
