import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

st.set_page_config(
    page_title="Sales Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

sns.set_theme(style="whitegrid")


@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)


@st.cache_data
def load_mysql(host, user, password, database, query):
    import mysql.connector

    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    try:
        return pd.read_sql(query, conn)
    finally:
        conn.close()


def prepare_data(df):
    df = df.copy()

    if "OrderDate" in df.columns:
        df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")
        df["Day_Name"] = df["OrderDate"].dt.day_name()
        df["Month_Name"] = df["OrderDate"].dt.month_name()
        df["Year"] = df["OrderDate"].dt.year
        df["Day"] = df["OrderDate"].dt.day

    return df


def show_dataframe_summary(df):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")


st.title("📊 Sales Data Analysis Dashboard")
st.caption("Streamlit dashboard based on the supplied EDA notebook")

# -----------------------------
# Data source
# -----------------------------
st.sidebar.header("Data Source")
source = st.sidebar.radio(
    "Choose data source",
    ["CSV Upload", "MySQL Database"]
)

df = None

if source == "CSV Upload":
    uploaded_file = st.sidebar.file_uploader(
        "Upload the sales CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df = load_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the CSV file: {e}")
            st.stop()
    else:
        st.info("Upload the CSV file from your notebook/data source to start the dashboard.")
        st.stop()

else:
    st.sidebar.subheader("MySQL Connection")

    host = st.sidebar.text_input(
        "Host",
        value=os.getenv("MYSQL_HOST", "localhost")
    )
    user = st.sidebar.text_input(
        "User",
        value=os.getenv("MYSQL_USER", "root")
    )
    password = st.sidebar.text_input(
        "Password",
        value=os.getenv("MYSQL_PASSWORD", ""),
        type="password"
    )
    database = st.sidebar.text_input(
        "Database",
        value=os.getenv("MYSQL_DATABASE", "eda_sql")
    )
    query = st.sidebar.text_area(
        "SQL Query",
        value="SELECT * FROM data"
    )

    connect = st.sidebar.button("Connect to MySQL")

    if connect:
        try:
            df = load_mysql(host, user, password, database, query)
            st.session_state["mysql_df"] = df
        except Exception as e:
            st.error(f"MySQL connection/query failed: {e}")
            st.stop()

    if df is None:
        df = st.session_state.get("mysql_df")

    if df is None:
        st.info("Enter the MySQL details and click 'Connect to MySQL'.")
        st.stop()

df = prepare_data(df)

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

filtered_df = df.copy()

if "Region" in df.columns:
    regions = sorted(df["Region"].dropna().astype(str).unique())
    selected_regions = st.sidebar.multiselect(
        "Region",
        regions,
        default=regions
    )
    if selected_regions:
        filtered_df = filtered_df[
            filtered_df["Region"].astype(str).isin(selected_regions)
        ]

if "ProductCategory" in df.columns:
    categories = sorted(
        df["ProductCategory"].dropna().astype(str).unique()
    )
    selected_categories = st.sidebar.multiselect(
        "Product Category",
        categories,
        default=categories
    )
    if selected_categories:
        filtered_df = filtered_df[
            filtered_df["ProductCategory"].astype(str).isin(selected_categories)
        ]

if "Year" in filtered_df.columns:
    years = sorted(
        pd.to_numeric(filtered_df["Year"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    if years:
        selected_years = st.sidebar.multiselect(
            "Year",
            years,
            default=years
        )
        if selected_years:
            filtered_df = filtered_df[
                pd.to_numeric(filtered_df["Year"], errors="coerce")
                .isin(selected_years)
            ]

st.sidebar.write(f"Filtered rows: {len(filtered_df):,}")

# -----------------------------
# Dashboard
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Overview",
        "Data Inspection",
        "Statistics",
        "Visual Analysis",
        "Missing Values"
    ]
)

with tab1:
    st.subheader("Overview")
    show_dataframe_summary(filtered_df)

    st.markdown("### Top 5 Rows")
    st.dataframe(filtered_df.head(), use_container_width=True)

    st.markdown("### Bottom 5 Rows")
    st.dataframe(filtered_df.tail(), use_container_width=True)

    st.markdown("### Column Data Types")
    dtype_df = pd.DataFrame({
        "Column": filtered_df.columns,
        "Data Type": filtered_df.dtypes.astype(str).values
    })
    st.dataframe(dtype_df, use_container_width=True)

with tab2:
    st.subheader("Data Inspection")

    st.markdown("### Dataset Shape")
    st.write(f"**{filtered_df.shape[0]:,} rows × {filtered_df.shape[1]:,} columns**")

    st.markdown("### Data Information")
    info_df = pd.DataFrame({
        "Column": filtered_df.columns,
        "Non-Null Count": filtered_df.notna().sum().values,
        "Null Count": filtered_df.isna().sum().values,
        "Dtype": filtered_df.dtypes.astype(str).values
    })
    st.dataframe(info_df, use_container_width=True)

    st.markdown("### Complete Dataset")
    st.dataframe(filtered_df, use_container_width=True)

with tab3:
    st.subheader("Statistical Analysis")

    numerical_columns = filtered_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = filtered_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if numerical_columns:
        st.markdown("### Numerical Data")
        st.dataframe(
            filtered_df[numerical_columns].describe().T,
            use_container_width=True
        )
    else:
        st.info("No numerical columns found.")

    if categorical_columns:
        st.markdown("### Categorical Data")
        st.dataframe(
            filtered_df[categorical_columns].describe().T,
            use_container_width=True
        )
    else:
        st.info("No categorical columns found.")

with tab4:
    st.subheader("Analysis by Visualization")

    # 1. Region-wise total SalesAmount
    if {"Region", "SalesAmount"}.issubset(filtered_df.columns):
        st.markdown("### 1. Region-wise Total Sales Amount")

        sales_by_region = (
            filtered_df.groupby("Region", dropna=False)["SalesAmount"]
            .sum()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            sales_by_region.index.astype(str),
            sales_by_region.values
        )
        ax.bar_label(bars, fmt="%.0f", padding=3)
        ax.set_xlabel("Region")
        ax.set_ylabel("Total Sales Amount")
        ax.set_title("Total Sales Amount by Region")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            sales_by_region.rename("Total Sales Amount").reset_index(),
            use_container_width=True
        )
    else:
        st.warning("Region or SalesAmount column is missing.")

    # 2. Top locations by units sold
    if {"Region", "UnitsSold"}.issubset(filtered_df.columns):
        st.markdown("### 2. Top 5 Locations by Units Sold")

        top_5 = (
            filtered_df.groupby("Region")["UnitsSold"]
            .sum()
            .nlargest(5)
        )

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            top_5.values,
            labels=top_5.index.astype(str),
            autopct="%1.1f%%",
            startangle=90
        )
        ax.set_title("Top 5 Locations by Units Sold")

        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            top_5.rename("Units Sold").reset_index(),
            use_container_width=True
        )
    else:
        st.warning("Region or UnitsSold column is missing.")

    # 3. Units sold by product category
    if {"ProductCategory", "UnitsSold"}.issubset(filtered_df.columns):
        st.markdown("### 3. Total Units Sold by Product Category")

        units_by_category = (
            filtered_df.groupby("ProductCategory")["UnitsSold"]
            .sum()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            x=units_by_category.index.astype(str),
            y=units_by_category.values,
            marker="o",
            linewidth=2,
            ax=ax
        )

        for category, value in units_by_category.items():
            ax.annotate(
                f"{value:,.0f}",
                (str(category), value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center"
            )

        ax.set_xlabel("Product Category")
        ax.set_ylabel("Total Units Sold")
        ax.set_title("Total Units Sold by Product Category")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            units_by_category.rename("Units Sold").reset_index(),
            use_container_width=True
        )
    else:
        st.warning("ProductCategory or UnitsSold column is missing.")

    # 4. Product category by year
    if {"Year", "ProductCategory"}.issubset(filtered_df.columns):
        st.markdown("### 4. Number of Product Categories by Year")

        category_by_year = (
            filtered_df.groupby(["Year", "ProductCategory"])
            .size()
            .unstack(fill_value=0)
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        category_by_year.plot(
            kind="bar",
            ax=ax
        )

        for container in ax.containers:
            ax.bar_label(container, padding=2, fontsize=8)

        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Products")
        ax.set_title("Number of Product Categories by Year")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Product Category")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(category_by_year, use_container_width=True)
    else:
        st.warning("Year or ProductCategory column is missing.")

with tab5:
    st.subheader("Missing Value Analysis")

    null_counts = filtered_df.isnull().sum().sort_values(ascending=False)
    null_table = null_counts.rename("Missing Values").to_frame()
    null_table["Missing %"] = (
        null_table["Missing Values"] / max(len(filtered_df), 1) * 100
    ).round(2)

    st.dataframe(null_table, use_container_width=True)

    columns_with_nulls = null_counts[null_counts > 0]

    if len(columns_with_nulls) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            x=columns_with_nulls.values,
            y=columns_with_nulls.index,
            ax=ax
        )
        ax.set_xlabel("Number of Missing Values")
        ax.set_ylabel("Column")
        ax.set_title("Missing Values by Column")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.success("No missing values found in the filtered dataset.")

st.divider()
st.caption("Built with Streamlit, Pandas, NumPy, Matplotlib and Seaborn.")
