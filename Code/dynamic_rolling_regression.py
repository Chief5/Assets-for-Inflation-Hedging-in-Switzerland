import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
import os

#ADD ALL RELEVANT TICKERS HERE
gold = "GLD"
smi = "^SSMI"
sp500 = "^GSPC"

#set file location
script_dir = os.path.dirname(os.path.abspath(__file__))  # Location of the script
project_root = os.path.abspath(os.path.join(script_dir, '..'))  # One level up from the script's directory
dataset_path = os.path.join(project_root, 'Datasets', 'cpi_ch.csv') # Construct the path to the dataset



def pull_data(*tickers):
    # Create a dictionary to store the data for each ticker
    data_dict = {}
        
    # Loop through each ticker and download its data using download_ticker_data
    for ticker in tickers:
        data_dict[ticker] = download_ticker_data(ticker)
        
    # Combine all ticker data into a single DataFrame, aligning by date
    combined_data = pd.concat(data_dict, axis=1)
    return combined_data

def download_ticker_data(ticker):
        # Download maximum available data for the Gold
        asset = yf.download(ticker, interval="1mo", period="max")
        asset_adj_close = asset[['Adj Close']]
        # Reset the index to make 'Date' a regular column, if not already
        asset_adj_close = asset_adj_close.reset_index()

        # Rename columns to match the SMI format
        asset_adj_close.columns = ['Date', 'Value']

        # Convert 'Date' to datetime format and set as index (if not already a datetime type)
        asset_adj_close['Date'] = pd.to_datetime(asset_adj_close['Date'], errors='coerce')
        asset_adj_close.set_index('Date', inplace=True)

        asset_adj_close.index = asset_adj_close.index.tz_localize(None)

        # Sort by date in ascending order
        asset_adj_close = asset_adj_close.sort_index(ascending=True)
        return asset_adj_close

def calculate_rates(df):
    # Loop through each asset column (ticker name only)
    for column in df.columns.get_level_values(0).unique():
        # Calculate Year-over-Year (YoY) return rate
        df[f'{column}_Return_Rate_YoY'] = df[(column, 'Value')].pct_change(12) * 100  # 12-month (YoY) change
            
        # Calculate Month-over-Month (MoM) return rate
        df[f'{column}_Return_Rate_MoM'] = df[(column, 'Value')].pct_change() * 100  # 1-month (MoM) change

    # Drop the second level in column names (if desired) after calculation
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    return df

# Helper function to calculate rolling beta
def calculate_rolling_beta(x, y, window):
    betas = []
    for i in range(len(x) - window + 1):
        x_window = x[i:i + window].values.reshape(-1, 1)
        y_window = y[i:i + window].values
        model = LinearRegression().fit(x_window, y_window)
        betas.append(model.coef_[0])
    return pd.Series(betas, index=x.index[window - 1:])


def calculate_correlation(*args):

    pulled_data = pull_data(*args)

    #read dataset
    # Replace 'path_to_cpi_data.csv' with the actual path to your CPI CSV file
    cpi_data = pd.read_csv(dataset_path)

    #prepare dataset DATETIME and ORDER
    # Convert the 'Date' column to datetime format
    cpi_data['Date'] = pd.to_datetime(cpi_data['Date'], errors='coerce')
    cpi_data.set_index('Date', inplace=True)
    cpi_data = cpi_data.sort_index(ascending=True)
    cpi_data['Value'] = pd.to_numeric(cpi_data['Value'], errors='coerce')

    #get RATES
    cpi_data['Inflation_Rate_YoY'] = cpi_data['Value'].pct_change(12) * 100  # 12-month (YoY) change
    cpi_data['Inflation_Rate_MoM'] = cpi_data['Value'].pct_change() * 100  # 1-month (MoM) change

    df_with_rates = calculate_rates(pulled_data)

    #here we merge the dataset that same START and END point
    # - merge that same START and END point

    # Find the latest start date and earliest end date between the two datasets
    start_date = max(cpi_data.index.min(), df_with_rates.index.min())
    end_date = min(cpi_data.index.max(), df_with_rates.index.max())

    # Filter both datasets to only include this date range
    cpi_data = cpi_data[start_date:end_date]
    df_with_rates = df_with_rates[start_date:end_date]

    # Prepare to merge CPI with all assets' MoM and YoY rates
    # Step 1: Merge CPI and assets data on Date for Month-over-Month Rates
    merged_data_mom = pd.merge(
        cpi_data[['Inflation_Rate_MoM']], 
        df_with_rates.filter(like='_Return_Rate_MoM'),  # Select all MoM rate columns
        left_index=True, 
        right_index=True
    )

    # Step 2: Merge CPI and assets data on Date for Year-over-Year Rates
    merged_data_yoy = pd.merge(
        cpi_data[['Inflation_Rate_YoY']], 
        df_with_rates.filter(like='_Return_Rate_YoY'),  # Select all YoY rate columns
        left_index=True, 
        right_index=True
    )

    # Step 2: Drop any rows with NaN values, as these will interfere with correlation calculation
    merged_data_mom.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged_data_mom.dropna(inplace=True)

    merged_data_yoy.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged_data_yoy.dropna(inplace=True)

    # Define the rolling window size (e.g., 12 months for a year-long window)
    window_size = 12


        # Calculate rolling beta for MoM data
    for column in merged_data_mom.columns:
        if column != 'Inflation_Rate_MoM':  # Skip CPI column itself
            # Calculate rolling beta with CPI data
            merged_data_mom[f'Rolling_Beta_{column}'] = calculate_rolling_beta(
                merged_data_mom['Inflation_Rate_MoM'], 
                merged_data_mom[column], 
                window=window_size
            )

    # Calculate rolling beta for YoY data
    for column in merged_data_yoy.columns:
        if column != 'Inflation_Rate_YoY':  # Skip CPI column itself
            # Calculate rolling beta with CPI data
            merged_data_yoy[f'Rolling_Beta_{column}'] = calculate_rolling_beta(
                merged_data_yoy['Inflation_Rate_YoY'], 
                merged_data_yoy[column], 
                window=window_size
            )

    print(merged_data_yoy)

    # Define significant inflation periods with labels
    significant_periods = [
        ("2008-01-01", "2009-12-31", "Global Financial Crisis"),
        ("2021-01-01", "2023-12-31", "Post-COVID-19 Pandemic Recovery")
    ]

    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Rolling MoM Correlation for all assets on the first subplot
    for column in merged_data_mom.columns:
        if 'Rolling_Beta_' in column:
            # Extract the ticker name from the column name
            ticker_name = column.replace('Rolling_Beta_', '').replace('_Return_Rate_MoM', '')
            # Plot the rolling correlation for each asset with simplified label
            ax1.plot(merged_data_mom.index, merged_data_mom[column], label=f'{ticker_name} ')

    # Customize the first subplot
    ax1.set_title('Rolling Month-over-Month Beta Between Inflation and Asset Returns')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Rolling Beta (MoM)')
    ax1.axhline(0, color='red', linestyle='--', linewidth=0.5)
    ax1.grid(visible=True, linestyle='--', linewidth=0.5)
    ax1.legend()

    # Add shaded regions with labels for significant inflation periods
    for start, end, label in significant_periods:
        ax1.axvspan(pd.to_datetime(start), pd.to_datetime(end), color='lightgrey', alpha=0.3)
        ax1.text(pd.to_datetime(start) + (pd.to_datetime(end) - pd.to_datetime(start)) / 2,
                ax1.get_ylim()[1] * 0.9, label, ha='center', va='top', fontsize=10, color='black')

    # Plot Rolling YoY Correlation for all assets on the second subplot
    for column in merged_data_yoy.columns:
        if 'Rolling_Beta_' in column:
            # Extract the ticker name from the column name
            ticker_name = column.replace('Rolling_Beta_', '').replace('_Return_Rate_YoY', '')
            # Plot the rolling correlation for each asset with simplified label
            ax2.plot(merged_data_yoy.index, merged_data_yoy[column], label=f'{ticker_name} ')

    # Customize the second subplot
    ax2.set_title('Rolling Year-over-Year Beta Between Inflation and Asset Returns')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Rolling Beta (YoY)')
    ax2.axhline(0, color='red', linestyle='--', linewidth=0.5)
    ax2.grid(visible=True, linestyle='--', linewidth=0.5)
    ax2.legend()

    # Add shaded regions with labels for YoY plot
    for start, end, label in significant_periods:
        ax2.axvspan(pd.to_datetime(start), pd.to_datetime(end), color='lightgrey', alpha=0.3)
        ax2.text(pd.to_datetime(start) + (pd.to_datetime(end) - pd.to_datetime(start)) / 2,
                ax2.get_ylim()[1] * 0.9, label, ha='center', va='top', fontsize=10, color='black')

    # Display the side-by-side plots
    plt.tight_layout()
    plt.show()

calculate_correlation(smi)
