import streamlit as st
import pandas as pd
import plotly.express as px
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

st.set_page_config(
    page_title="AI Analytics Copilot",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Analytics Copilot")

st.write("Upload a CSV file to generate basic dataset insights.")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success(f"Uploaded file: {uploaded_file.name}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), width="stretch")

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Column Details")

    column_info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(column_info, width="stretch")

    st.subheader("Basic Statistical Summary")
    summary_df = df.describe(include="all").astype(str)
    st.dataframe(summary_df, width="stretch")

    st.subheader("Auto-Detected Numeric KPIs")

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if numeric_columns:
        selected_kpi_column = st.selectbox(
            "Select a numeric column to analyze",
            numeric_columns
        )

        total_value = df[selected_kpi_column].sum()
        average_value = df[selected_kpi_column].mean()
        min_value = df[selected_kpi_column].min()
        max_value = df[selected_kpi_column].max()

        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

        with kpi_col1:
            st.metric("Total", f"{total_value:,.2f}")

        with kpi_col2:
            st.metric("Average", f"{average_value:,.2f}")

        with kpi_col3:
            st.metric("Minimum", f"{min_value:,.2f}")

        with kpi_col4:
            st.metric("Maximum", f"{max_value:,.2f}")
    else:
        st.warning("No numeric columns found for KPI analysis.")

    st.subheader("Rows With Negative Values")

    negative_rows = df[df[selected_kpi_column] < 0]

    st.dataframe(negative_rows, width="stretch")

    st.subheader("KPI Distribution Chart")

    fig_hist = px.histogram(
        df,
        x=selected_kpi_column,
        nbins=30,
        title=f"Distribution of {selected_kpi_column}"
    )

    st.plotly_chart(fig_hist, width="stretch")

    st.subheader("Box Plot Analysis")

    fig_box = px.box(
        df,
        y=selected_kpi_column,
        title=f"Box Plot of {selected_kpi_column}"
    )

    st.subheader("Automatic Outlier Detection")

    Q1 = df[selected_kpi_column].quantile(0.25)
    Q3 = df[selected_kpi_column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[selected_kpi_column] < lower_bound) |
        (df[selected_kpi_column] > upper_bound)
        ]

    st.write(f"Lower Bound: {lower_bound:,.2f}")
    st.write(f"Upper Bound: {upper_bound:,.2f}")
    st.write(f"Outlier Rows Found: {outliers.shape[0]}")

    if not outliers.empty:
        st.dataframe(outliers.head(50), width="stretch")
    else:
        st.success("No outliers detected for this column.")

    st.plotly_chart(fig_box, width="stretch")

    st.subheader("Top 10 Highest Values")

    top_10 = df.nlargest(10, selected_kpi_column)

    st.dataframe(top_10, width="stretch")

    st.subheader("Rule-Based Business Insights")

    insights = []

    rows_count = df.shape[0]
    columns_count = df.shape[1]
    missing_values_count = int(df.isnull().sum().sum())
    outlier_count = outliers.shape[0]

    insights.append(f"The dataset contains {rows_count:,} rows and {columns_count:,} columns.")

    if missing_values_count > 0:
        insights.append(
            f"There are {missing_values_count:,} missing values in the dataset, so data cleaning may be required.")
    else:
        insights.append("There are no missing values in the dataset.")

    if outlier_count > 0:
        insights.append(
            f"The selected KPI column '{selected_kpi_column}' has {outlier_count:,} outlier rows, which may represent unusual transactions, returns, errors, or bulk activity."
        )
    else:
        insights.append(f"No major outliers were detected in '{selected_kpi_column}'.")

    if numeric_columns:
        highest_avg_column = df[numeric_columns].mean().idxmax()
        highest_avg_value = df[highest_avg_column].mean()

        insights.append(
            f"Among numeric columns, '{highest_avg_column}' has the highest average value at {highest_avg_value:,.2f}."
        )

    for insight in insights:
        st.info(insight)

    st.subheader("AI Generated Business Insights")

    if st.button("Generate AI Insights"):
        sample_data = df.head(20).to_string()

        prompt = f"""
        You are a business data analyst.

        Analyze this dataset sample and provide:
        1. Key trends
        2. Possible anomalies
        3. Business risks
        4. Actionable recommendations

        Dataset Sample:
        {sample_data}
        """

        with st.spinner("Generating AI insights..."):
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )

            ai_insights = response.choices[0].message.content

            st.write(ai_insights)

else:
    st.info("Please upload a CSV file to begin.")