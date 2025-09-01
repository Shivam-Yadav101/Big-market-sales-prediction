import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')

def main():
    st.title("Big Mart Sales Data Exploration")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload your Training.csv file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Data Preview")
        st.dataframe(df.head())

        st.subheader("Dataset Info")
        buffer = []
        df.info(buf=buffer)
        info_str = "\n".join(buffer)
        st.text(info_str)

        st.subheader("Missing Values Count")
        st.write(df.isnull().sum())

        categorical_data = df.select_dtypes(include=[object])
        numerical_data = df.select_dtypes(include=[np.float64, np.int64])

        st.write(f"Count of categorical features: {categorical_data.shape[1]}")
        st.write(f"Count of numerical features: {numerical_data.shape[1]}")

        # Fill missing 'Outlet_Size' with mode
        categorical_data['Outlet_Size'] = categorical_data['Outlet_Size'].fillna(categorical_data['Outlet_Size'].mode()[0])

        st.subheader("Categorical Feature: Outlet_Size Value Counts")
        st.write(categorical_data['Outlet_Size'].value_counts())

        # Plots
        st.subheader("Countplot: Outlet_Size")
        fig, ax = plt.subplots()
        sns.countplot(x='Outlet_Size', data=categorical_data, ax=ax)
        st.pyplot(fig)

        st.subheader("Countplot: Item_Fat_Content")
        categorical_data.replace({'Item_Fat_Content': {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}}, inplace=True)
        fig, ax = plt.subplots()
        sns.countplot(x='Item_Fat_Content', data=categorical_data, ax=ax)
        st.pyplot(fig)

        st.subheader("Countplot: Outlet_Identifier")
        fig, ax = plt.subplots(figsize=(6, 8))
        sns.countplot(y='Outlet_Identifier', data=categorical_data, ax=ax)
        st.pyplot(fig)

        st.subheader("Countplot: Item_Type")
        fig, ax = plt.subplots(figsize=(8, 12))
        sns.countplot(y='Item_Type', data=categorical_data, ax=ax)
        st.pyplot(fig)

        st.subheader("Countplots: Outlet_Type and Outlet_Location_Type")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.countplot(y='Outlet_Type', data=categorical_data, ax=axes[0])
        sns.countplot(y='Outlet_Location_Type', data=categorical_data, ax=axes[1])
        st.pyplot(fig)

        st.subheader("Numerical Data Description")
        st.write(numerical_data.describe())

        st.subheader("Histograms of Numerical Features")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        numerical_data['Item_Weight'].hist(bins=100, ax=axes[0])
        axes[0].set_title('Item_Weight')
        numerical_data['Item_Visibility'].hist(bins=100, ax=axes[1])
        axes[1].set_title('Item_Visibility')
        numerical_data['Item_MRP'].hist(bins=100, ax=axes[2])
        axes[2].set_title('Item_MRP')
        st.pyplot(fig)

        st.subheader("Countplot: Outlet_Establishment_Year")
        fig, ax = plt.subplots()
        sns.countplot(x='Outlet_Establishment_Year', data=numerical_data, ax=ax)
        st.pyplot(fig)

        st.subheader("Barplot: Item_Type vs Item_Outlet_Sales")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(y='Item_Type', x='Item_Outlet_Sales', data=df, ax=ax)
        st.pyplot(fig)

        st.subheader("Barplot: Outlet_Size vs Item_Outlet_Sales")
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.barplot(x='Outlet_Size', y='Item_Outlet_Sales', data=df, ax=ax)
        st.pyplot(fig)

        st.subheader("Barplot: Outlet_Location_Type vs Item_Outlet_Sales by Outlet_Type")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Outlet_Location_Type', y='Item_Outlet_Sales', hue='Outlet_Type', data=df, ax=ax)
        ax.legend()
        st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include='number')
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
