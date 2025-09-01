import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')

st.title("📊 Big Mart Sales EDA")

# Load Data
df = pd.read_csv("BigMart Sales Data.csv")
st.subheader("Dataset Preview")
st.write(df.head())

# Shape of the data
st.subheader("Dataset Shape")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Info about data
st.subheader("Dataset Info")
buffer = []
df.info(buf=buffer.append)  # Capture info as text
st.text("\n".join(buffer))

# Missing Values
st.subheader("Missing Values Count")
st.write(df.isnull().sum())

# Separate Categorical and Numerical Data
categorical_data = df.select_dtypes(include=[object])
numerical_data = df.select_dtypes(include=[np.float64, np.int64])

st.write(f"**Categorical Features:** {categorical_data.shape[1]}")
st.write(f"**Numerical Features:** {numerical_data.shape[1]}")

# Fill Missing Values
categorical_data['Outlet_Size'] = categorical_data['Outlet_Size'].fillna(
    categorical_data['Outlet_Size'].mode()[0]
)

st.subheader("Outlet Size Count")
fig, ax = plt.subplots()
sns.countplot(x='Outlet_Size', data=categorical_data, ax=ax)
st.pyplot(fig)

st.subheader("Item Fat Content (Before Cleaning)")
fig, ax = plt.subplots()
sns.countplot(x='Item_Fat_Content', data=categorical_data, ax=ax)
st.pyplot(fig)

# Clean Item_Fat_Content
categorical_data.replace({'Item_Fat_Content': {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}}, inplace=True)

st.subheader("Item Fat Content (After Cleaning)")
fig, ax = plt.subplots()
sns.countplot(x='Item_Fat_Content', data=categorical_data, ax=ax)
st.pyplot(fig)

# Outlet Identifier Count
st.subheader("Outlet Identifier Count")
fig, ax = plt.subplots()
sns.countplot(y='Outlet_Identifier', data=categorical_data, ax=ax)
st.pyplot(fig)

# Item Type Count
st.subheader("Item Type Distribution")
fig, ax = plt.subplots(figsize=(12, 8))
sns.countplot(y='Item_Type', data=categorical_data, ax=ax)
st.pyplot(fig)

# Outlet Type & Location Type
st.subheader("Outlet Type & Location Type Distribution")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(y='Outlet_Type', data=categorical_data, ax=axes[0])
sns.countplot(y='Outlet_Location_Type', data=categorical_data, ax=axes[1])
st.pyplot(fig)

# Numerical Data Summary
st.subheader("Numerical Data Summary")
st.write(numerical_data.describe())

# Histograms
st.subheader("Numerical Feature Distributions")
for col in ['Item_Weight', 'Item_Visibility', 'Item_MRP']:
    fig, ax = plt.subplots()
    numerical_data[col].hist(bins=100, ax=ax)
    ax.set_title(col)
    st.pyplot(fig)

# Outlet Establishment Year
st.subheader("Outlet Establishment Year")
fig, ax = plt.subplots()
sns.countplot(x='Outlet_Establishment_Year', data=numerical_data, ax=ax)
st.pyplot(fig)

# Item Outlet Sales by Item Type
st.subheader("Item Outlet Sales by Item Type")
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(y='Item_Type', x='Item_Outlet_Sales', data=df, ax=ax)
st.pyplot(fig)

# Item Outlet Sales by Outlet Size
st.subheader("Item Outlet Sales by Outlet Size")
fig, ax = plt.subplots(figsize=(7, 7))
sns.barplot(x='Outlet_Size', y='Item_Outlet_Sales', data=df, ax=ax)
st.pyplot(fig)

# Outlet Location Type vs Outlet Type
st.subheader("Outlet Location Type vs Outlet Type")
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='Outlet_Location_Type', y='Item_Outlet_Sales', hue='Outlet_Type', data=df, ax=ax)
ax.legend()
st.pyplot(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, ax=ax)
ax.set_title('Correlation between the columns')
st.pyplot(fig)
