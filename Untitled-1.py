import pandas as pd
import numpy as np

def vectorized_exponential_smoothing_all_robust(base_path="/Users/dhineashkumar/Desktop/AI project/"):
    """
    Vectorized exponential smoothing for ALL 10 stores and 3049 items.
    Ensures every store-item pair has a forecast for all 365 days of 2014,
    with more robust handling of missing historical data.
    """
    print("🚀 Running Robust Vectorized Exponential Smoothing for all store–item combos")

    # --- 1. Load Data ---
    try:
        sell_prices = pd.read_csv(f"{base_path}sell_prices.csv")
        calendar = pd.read_csv(f"{base_path}calendar.csv", usecols=['wm_yr_wk', 'date'])
    except FileNotFoundError as e:
        print(f"❌ Error loading file: {e}. Please ensure 'sell_prices.csv' and 'calendar.csv' are in the specified base_path.")
        return pd.DataFrame() # Return empty DataFrame on error

    # --- 2. Prepare Data ---
    # Convert 'date' to datetime and extract year
    calendar['date'] = pd.to_datetime(calendar['date'])
    calendar['year'] = calendar['date'].dt.year

    # Merge sell_prices with calendar
    # Only merge relevant columns to save memory if original sell_prices has many columns
    merged = pd.merge(sell_prices, calendar[['wm_yr_wk', 'date', 'year']], on='wm_yr_wk', how='left')

    # Ensure correct data types and sort for efficient grouping
    merged['item_id'] = merged['item_id'].astype('category')
    merged['store_id'] = merged['store_id'].astype('category')
    merged = merged.sort_values(['item_id', 'store_id', 'date'])

    # Get all unique store-item combinations
    all_combos = merged[['item_id', 'store_id']].drop_duplicates()

    # --- 3. Generate Complete 2014 Date Range ---
    start_date_2014 = pd.to_datetime('2014-01-01')
    end_date_2014 = pd.to_datetime('2014-12-31')
    all_dates_2014 = pd.date_range(start=start_date_2014, end=end_date_2014, freq='D')

    # Create a full DataFrame with all store-item-date combinations for 2014
    # This ensures every combination exists for the final output
    full_2014_template = pd.MultiIndex.from_product(
        [all_combos['item_id'].unique(),
         all_combos['store_id'].unique(),
         all_dates_2014],
        names=['item_id', 'store_id', 'date']
    ).to_frame(index=False)

    # --- 4. Calculate Exponential Smoothing Forecasts (Pre-2014) ---
    # Filter training data up to 2013
    train_data = merged[merged['year'] <= 2013].copy()

    # Define the exponential smoothing function for apply
    def calculate_smoothed_forecast(group):
        if len(group) >= 2:
            # Apply EWM if at least two data points
            smoothed_series = group['sell_price'].ewm(alpha=0.3, adjust=False).mean()
            return smoothed_series.iloc[-1]
        elif len(group) == 1:
            # If only one data point, use that as the forecast
            return group['sell_price'].iloc[0]
        else:
            # No training data for this combo. Return NaN for now, will be imputed later.
            return np.nan

    # Apply smoothing to each item-store group to get one forecast value per combo
    # This forecast represents the smoothed value at the end of the training period (2013)
    forecast_values = (
        train_data.groupby(['item_id', 'store_id'])
        .apply(calculate_smoothed_forecast)
        .rename('forecast')
    )

    # --- 5. Impute Missing Forecasts (for combos with no or very little training data) ---
    # Get the last known selling price for each item-store combination (from all years)
    # This is a good fallback if no pre-2014 data for smoothing.
    last_known_prices = (
        merged.groupby(['item_id', 'store_id'])['sell_price']
        .last() # Get the last observed price
        .rename('last_known_price')
    )

    # Merge last known prices with forecast_values
    forecast_values = forecast_values.to_frame().merge(
        last_known_prices, on=['item_id', 'store_id'], how='left'
    )

    # Fill NaN forecasts with the last known price
    forecast_values['forecast'].fillna(forecast_values['last_known_price'], inplace=True)

    # If still NaN (meaning no historical data at all for this item-store combo),
    # fill with the overall mean sell price.
    overall_mean_price = merged['sell_price'].mean()
    forecast_values['forecast'].fillna(overall_mean_price if not merged['sell_price'].empty else 0, inplace=True)
    forecast_values.drop(columns=['last_known_price'], inplace=True)


    # --- 6. Construct Final Forecast DataFrame for 2014 ---
    # Merge the full 2014 template with the calculated forecasts
    final_df = pd.merge(full_2014_template, forecast_values, on=['item_id', 'store_id'], how='left')

    # --- 7. Add Actual Selling Prices for 2014 ---
    # Filter actual 2014 data
    actual_2014_data = merged[merged['year'] == 2014][['date', 'item_id', 'store_id', 'sell_price']]
    actual_2014_data.rename(columns={'sell_price': 'actual_selling_price'}, inplace=True)

    # Merge actuals with the final forecast DataFrame
    final_df = pd.merge(final_df, actual_2014_data, on=['date', 'item_id', 'store_id'], how='left')

    # --- 8. Save Output ---
    output_path = f"{base_path}forecast_vectorized_all_robust.csv"
    final_df.to_csv(output_path, index=False)
    print(f"✅ Forecast generated for {len(final_df)} records saved to {output_path}")
    print(f"   ({len(all_combos)} store–item combos × {len(all_dates_2014)} days)")

    return final_df

# Example run
if __name__ == "__main__":
    forecast_df = vectorized_exponential_smoothing_all_robust()
    if not forecast_df.empty:
        print("\nSample of generated forecast:")
        print(forecast_df.head())
        print(f"\nTotal unique item_id-store_id combinations with forecasts: {forecast_df[['item_id', 'store_id']].drop_duplicates().shape[0]}")
        print(f"Total rows in output: {len(forecast_df)}")
        print(f"Number of NaN forecasts (should be 0): {forecast_df['forecast'].isnull().sum()}")