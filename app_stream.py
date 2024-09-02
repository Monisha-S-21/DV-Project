import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

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

# Streamlit app layout
st.title("Customer Churn Prediction - EDA Dashboard")

# Summary Statistics
st.header("Summary Statistics")
summary_stats = df.describe(include='all').transpose()
st.write(summary_stats)

# Plot Summary Statistics
fig_summary = px.bar(summary_stats, x=summary_stats.index, y='mean', title='Summary Statistics', labels={'mean': 'Average Value'})
st.plotly_chart(fig_summary)

# Correlation Heatmap for all features
st.header("Correlation Heatmap")
corr = df.corr()
fig_heatmap = go.Figure(data=go.Heatmap(z=corr.values,
                                       x=corr.columns,
                                       y=corr.columns,
                                       colorscale='Viridis'))
fig_heatmap.update_layout(title='Correlation Heatmap')
st.plotly_chart(fig_heatmap)

# Customer Behavior Analysis: Churn Distribution
st.header("Churn Distribution")
churn_dist = df['Churn'].value_counts().reset_index()
churn_dist.columns = ['Churn', 'Count']
fig_churn = px.pie(churn_dist, names='Churn', values='Count', title='Churn Distribution')
st.plotly_chart(fig_churn)

# Distribution of Monthly Charges by Churn
st.header("Monthly Charges Distribution by Churn")
fig_charges = px.box(df, x='Churn', y='MonthlyCharges', title='Monthly Charges Distribution by Churn')
st.plotly_chart(fig_charges)

# Distribution of Tenure by Churn
st.header("Tenure Distribution by Churn")
fig_tenure = px.box(df, x='Churn', y='tenure', title='Tenure Distribution by Churn')
st.plotly_chart(fig_tenure)

st.header("Churn Rate by Internet Service")
# Handle missing or unexpected values
if 'Churn' in df.columns and 'InternetService' in df.columns:
    churn_internet_service = df.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack().fillna(0)
    churn_internet_service['Churn Rate'] = churn_internet_service.get(1, 0) / (churn_internet_service.get(1, 0) + churn_internet_service.get(0, 0))
    fig_internet_service = px.bar(churn_internet_service, x=churn_internet_service.index, y='Churn Rate', title='Churn Rate by Internet Service')
    st.plotly_chart(fig_internet_service)
else:
    st.write("Columns 'Churn' or 'InternetService' are missing in the dataset.")

# Churn Rate by Contract Type
st.header("Churn Rate by Contract Type")
if 'Churn' in df.columns and 'Contract' in df.columns:
    churn_contract = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack().fillna(0)
    churn_contract['Churn Rate'] = churn_contract.get(1, 0) / (churn_contract.get(1, 0) + churn_contract.get(0, 0))
    fig_contract = px.bar(churn_contract, x=churn_contract.index, y='Churn Rate', title='Churn Rate by Contract Type')
    st.plotly_chart(fig_contract)
    st.write("The bar chart shows the churn rate for different contract types. It helps in understanding if contract length influences customer retention.")

# Monthly Charges vs. Total Charges
st.header("Monthly Charges vs. Total Charges")
fig_charges_total = px.scatter(df, x='MonthlyCharges', y='TotalCharges', color='Churn', title='Monthly Charges vs. Total Charges')
st.plotly_chart(fig_charges_total)
st.write("The scatter plot illustrates the relationship between monthly charges and total charges, colored by churn status. It helps to identify any patterns or clusters in charge-related features.")

# Distribution of Customer Age
st.header("Distribution of Customer Age")
# Assuming age is not directly available, you can use tenure as a proxy if age is not available.
fig_age = px.histogram(df, x='tenure', title='Distribution of Customer Tenure')
st.plotly_chart(fig_age)
st.write("The histogram shows the distribution of customer tenure, which can serve as a proxy for customer age. It provides insight into the typical duration customers stay with the company.")