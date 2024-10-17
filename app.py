import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


CORRELATION_INDICATORS = [
    "Region",
    "IncomeGroup",
    "ExternalDebtReportingStatus",
    "CurrencyUnit",
    "SystemOfTrade",
    "LatestPopulationCensus",
]


# Function to load and prepare data
def load_data():
    gdp_data = pd.read_csv("data/gdp_per_capita.csv")
    dev_data = pd.read_csv("data/development_data.csv")
    return gdp_data, dev_data


def preprocess_development_data(dev_data):
    # Convert categorical columns to numerical
    dev_data = dev_data.copy()
    for column in CORRELATION_INDICATORS:
        dev_data[column] = dev_data[column].astype("category").cat.codes
    return dev_data


# Main function for Streamlit app
def main():
    st.title("Country GDP and Development Data Analysis")

    # Load and preprocess data
    gdp_data, dev_data = load_data()
    dev_data = preprocess_development_data(dev_data)

    # Display raw data
    if st.checkbox("Show raw GDP data"):
        st.subheader("GDP Per Capita Data")
        st.write(gdp_data)

    if st.checkbox("Show raw Development data"):
        st.subheader("Country Development Data")
        st.write(dev_data)

    # Merge GDP and development data
    st.subheader("Correlation of GDP and development indicators")
    gdp_melted = gdp_data.melt(
        id_vars=["Sr.No", "Country"], var_name="Year", value_name="GDP"
    )
    df_merged = pd.merge(gdp_melted, dev_data, on="Country")

    indicators = CORRELATION_INDICATORS

    # Create a figure and axis
    plt.figure(figsize=(12, 6))

    # Calculate and plot correlation for each indicator
    for indicator in indicators:
        # Compute correlation
        correlation_df = (
            df_merged.groupby("Year")
            .apply(lambda x: x[["GDP", indicator]].corr().apply(abs).iloc[0, 1])
            .reset_index()
        )
        correlation_df.columns = ["Year", "Correlation"]

        # Plot
        plt.plot(
            correlation_df["Year"],
            correlation_df["Correlation"],
            marker="o",
            linestyle="-",
            label=f"{indicator}",
        )

    # Customize the plot
    plt.title("Correlation Between Different Indicators and GDP Over Time")
    plt.xlabel("Year")
    plt.ylabel("Correlation")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)

    # Plot histogram of GDP per capita
    st.subheader("Histogram of GDP Per Capita")
    gdp_data_melted = gdp_data.melt(
        id_vars=["Country"], var_name="Year", value_name="GDP_Per_Capita"
    )
    plt.figure(figsize=(10, 6))
    sns.histplot(gdp_data_melted["GDP_Per_Capita"], bins=30, kde=True)
    plt.title("Histogram of GDP Per Capita")
    plt.xlabel("GDP Per Capita")
    plt.ylabel("Frequency")
    st.pyplot(plt)

    # Plot GDP per capita over time
    st.subheader("GDP Per Capita Over Time")
    countries = gdp_data["Country"].unique()
    selected_country = st.selectbox("Select a country", countries)
    country_data = gdp_data[gdp_data["Country"] == selected_country]
    years = list(map(str, range(1970, 2023)))
    plt.figure(figsize=(12, 6))
    plt.plot(
        years,
        country_data[years].values.flatten(),
        marker="o",
    )
    plt.title(f"GDP Per Capita of {selected_country}")
    plt.xlabel("Year")
    plt.ylabel("GDP Per Capita")
    plt.xticks(rotation=45, fontsize=8)
    st.pyplot(plt)


if __name__ == "__main__":
    main()
