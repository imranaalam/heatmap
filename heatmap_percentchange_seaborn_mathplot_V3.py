import duckdb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import os
import pytz
from datetime import datetime, timedelta, time as dt_time
from streamlit_autorefresh import st_autorefresh  # Import st_autorefresh for auto-refresh

# -----------------------------
# Streamlit App Configuration
# -----------------------------

# Set Streamlit page configuration for better layout
st.set_page_config(layout="wide", page_title="Stock Heatmap Dashboard")

# -----------------------------
# Configuration Variables
# -----------------------------

DATABASE_PATH = 'stock_data.duckdb'  # Ensure this path is correct and accessible
LOCAL_TIMEZONE = 'Asia/Karachi'

# Define the available time intervals in minutes
INTERVAL_OPTIONS = {
    "1 minute": 1,
    "2 minutes": 2,
    "3 minutes": 3,
    "5 minutes": 5,
    "10 minutes": 10,
    "15 minutes": 15,
    "30 minutes": 30,
    "45 minutes": 45,
    "60 minutes": 60,
    "90 minutes": 90,
    "120 minutes": 120,
    "180 minutes": 180,
    "240 minutes": 240,
    "300 minutes": 300,
    "360 minutes": 360
}

DEFAULT_NUM_SYMBOLS = 120  # Default set to 120

# -----------------------------
# Functions
# -----------------------------
@st.cache_data(ttl=60)
def get_data_from_db(database_path):
    """Load data from the DuckDB database."""
    if not os.path.isfile(database_path):
        st.error(f"Database file not found at: {database_path}")
        return None
    try:
        conn = duckdb.connect(database_path)
        query = """
            SELECT symbol, time, close, volume, change_percent
            FROM stock_data
            WHERE change_percent IS NOT NULL AND symbol IS NOT NULL
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error reading the database: {e}")
        return None

def convert_timestamp(df, timezone):
    """Convert UNIX timestamp to timezone-aware datetime."""
    try:
        df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df['datetime'] = df['datetime'].dt.tz_convert(timezone)
        return df
    except Exception as e:
        st.error(f"Error converting timestamps: {e}")
        return df

def calculate_speed_of_change(group, interval_minutes):
    """Calculate the Speed of Change for a given group."""
    group = group.sort_values('datetime')
    if len(group) < 2:
        return np.nan
    speed = (group['change_percent'].iloc[-1] - group['change_percent'].iloc[0]) / interval_minutes
    return speed

def calculate_volume_weighted_change(group):
    """Calculate the Volume-Weighted Percentage Change for a given group."""
    total_volume = group['volume'].sum()
    if total_volume > 0:
        weighted_change = (group['change_percent'] * group['volume']).sum() / total_volume
        return weighted_change
    else:
        return np.nan

def calculate_ema(group, span):
    """Calculate the Exponential Moving Average (EMA) for a given group."""
    group = group.sort_values('datetime')
    ema = group['change_percent'].ewm(span=span, adjust=False).mean().iloc[-1]
    return ema

# -----------------------------
# Sidebar Controls - Section 1: Analysis & Calculation Options
# -----------------------------
st.sidebar.title("Analysis & Calculation Options")

# Calculation Method Selection
calculation_method = st.sidebar.selectbox(
    "Data Analysis & Calculation Method",
    options=[
        "Last Percentage Change (Default)",
        "Speed of Change",
        "Volume-Weighted Percentage Change",
        "Exponential Moving Average (EMA)"
    ],
    index=0,
    help="Choose how to calculate the percentage change for intraday momentum analysis."
)

# Time Interval for Calculation
time_interval_label = st.sidebar.selectbox(
    "Time Interval for Calculation",
    options=list(INTERVAL_OPTIONS.keys()),
    index=0,  # Default to "1 minute"
    help="Select the time interval over which to calculate the selected metric."
)
time_interval_minutes = INTERVAL_OPTIONS[time_interval_label]

# -----------------------------
# Sidebar Controls - Section 2: Controls
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.title("Controls")

# **1. Sorting Option**
sort_option = st.sidebar.radio(
    "Sort Results By:",
    options=["Percentage Change (Default)", "Alphabetically"],
    index=0,
    help="Choose to sort symbols by their latest percentage change or alphabetically."
)

# **2. Date Selector**
# Load data first to determine min and max dates
df_initial = get_data_from_db(DATABASE_PATH)

# Check if there is any data available in the database
if df_initial is not None and not df_initial.empty:
    df_initial = convert_timestamp(df_initial, LOCAL_TIMEZONE)
    df_initial['change_percent'] = pd.to_numeric(df_initial['change_percent'], errors='coerce')
    df_initial['close'] = pd.to_numeric(df_initial['close'], errors='coerce')
    df_initial['volume'] = pd.to_numeric(df_initial['volume'], errors='coerce')
    df_initial.dropna(subset=['change_percent', 'close', 'volume'], inplace=True)

    if not df_initial.empty:
        min_date = df_initial['datetime'].dt.date.min()
        max_date = df_initial['datetime'].dt.date.max()

        # Set the default date to the latest available date in the dataset
        default_date = max_date
    else:
        min_date = datetime.now().date()
        max_date = datetime.now().date()
        default_date = max_date
else:
    min_date = datetime.now().date()
    max_date = datetime.now().date()
    default_date = max_date

# Use the latest available date as the default
selected_date = st.sidebar.date_input(
    "Select Date",
    value=default_date,
    min_value=min_date,
    max_value=max_date,
    help="Select the date for which you want to analyze the stock data."
)

# **3. Time Range Selectors**
start_time_default = dt_time(9, 0)  # Default start time at 9:00 AM
end_time_default = dt_time(16, 0)   # Default end time at 4:00 PM

selected_start_time = st.sidebar.time_input(
    "Start Time",
    value=start_time_default,
    help="Select the start time for filtering the stock data."
)

selected_end_time = st.sidebar.time_input(
    "End Time",
    value=end_time_default,
    help="Select the end time for filtering the stock data."
)

# Validate that start time is before end time
if selected_start_time >= selected_end_time:
    st.sidebar.error("Start time must be earlier than end time.")

# **4. Display Options: Number of Symbols to Display**
num_symbols = st.sidebar.slider(
    "Select number of top symbols to display",
    min_value=10,
    max_value=300,
    value=DEFAULT_NUM_SYMBOLS,
    step=10,
    help="Choose how many top symbols to display based on the selected sorting method."
)

# **5. Refresh & Sync Interval Selection**
st.sidebar.markdown("### Refresh Interval")
REFRESH_OPTIONS = {
    "10 seconds": 10000,
    "30 seconds": 30000,
    "45 seconds": 45000,
    "1 minute": 60000,
    "2 minutes": 120000,
    "3 minutes": 180000,
    "5 minutes": 300000,
    "10 minutes": 600000
}

selected_refresh_label = st.sidebar.selectbox(
    "Select Sync Interval",
    options=list(REFRESH_OPTIONS.keys()),
    index=3,  # Default to '1 minute'
    help="Choose how frequently the data and heatmap refresh."
)

refresh_interval = REFRESH_OPTIONS[selected_refresh_label]

# **6. Visualization Options: Color Theme**
st.sidebar.markdown("---")
st.sidebar.title("Visualization Options")

# Since we're using only the custom 11-color scale, no need for color theme selection
st.sidebar.info("Only the custom 11-color scale is available for the heatmap.")

# -----------------------------
# Auto Refresh Configuration
# -----------------------------

# Use st_autorefresh for automatic refresh with the user-selected interval
refresh_count = st_autorefresh(
    interval=refresh_interval,  # Use the interval from the sidebar (in milliseconds)
    limit=None,                 # Set to None for infinite refreshes
    key="data_refresh"          # Unique key for this component
)

# -----------------------------
# Load and Process Data
# -----------------------------

df = get_data_from_db(DATABASE_PATH)

if df is not None and not df.empty:
    df = convert_timestamp(df, LOCAL_TIMEZONE)
    # Ensure numeric columns are properly typed
    df['change_percent'] = pd.to_numeric(df['change_percent'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    # Drop rows with NaN in essential columns
    df.dropna(subset=['change_percent', 'close', 'volume'], inplace=True)

    # Debugging: Check if 'datetime' column exists
    if 'datetime' not in df.columns:
        st.error("The 'datetime' column is missing after processing. Please check your data.")
        st.stop()

    # -----------------------------
    # Filter Data Based on Date and Time Range
    # -----------------------------
    try:
        start_datetime = datetime.combine(selected_date, selected_start_time)
        end_datetime = datetime.combine(selected_date, selected_end_time)
        timezone = pytz.timezone(LOCAL_TIMEZONE)
        start_datetime = timezone.localize(start_datetime)
        end_datetime = timezone.localize(end_datetime)
    except Exception as e:
        st.error(f"Error with date/time selection: {e}")
        st.stop()

    # Validate that start time is before end time
    if start_datetime >= end_datetime:
        st.error("Start time must be before end time.")
        st.stop()

    selected_df = df[(df['datetime'] >= start_datetime) & (df['datetime'] <= end_datetime)]
    selected_df = selected_df.sort_values('datetime')

    if selected_df.empty:
        st.warning("No trading data available for the selected time range.")
        st.stop()

    # -----------------------------
    # Resample Data Based on Selected Interval
    # -----------------------------
    # Define the resampling rule
    resample_rule = f"{time_interval_minutes}T"  # e.g., '5T' for 5 minutes

    # Group the data by symbol and resample
    resampled_df = selected_df.groupby('symbol').apply(
        lambda x: x.set_index('datetime').resample(resample_rule).agg({
            'change_percent': 'last',
            'volume': 'sum'
        }).fillna(method='ffill')
    ).reset_index()

    # Ensure 'change_percent' is numeric
    resampled_df['change_percent'] = pd.to_numeric(resampled_df['change_percent'], errors='coerce')
    resampled_df['volume'] = pd.to_numeric(resampled_df['volume'], errors='coerce')

    # Drop rows with NaN in 'change_percent'
    resampled_df.dropna(subset=['change_percent'], inplace=True)

    # -----------------------------
    # Metric Calculations
    # -----------------------------
    metric_label = ""

    if calculation_method == "Last Percentage Change (Default)":
        # For each symbol and interval, take the last 'change_percent'
        metric_df = resampled_df.copy()
        metric_df['metric'] = metric_df['change_percent']
        metric_label = "Last Percentage Change (%)"
    elif calculation_method == "Speed of Change":
        # Calculate Speed of Change as (current_change - previous_change) / time_interval_minutes
        metric_df = resampled_df.copy()
        metric_df = metric_df.sort_values(['symbol', 'datetime'])
        metric_df['speed_of_change'] = metric_df.groupby('symbol')['change_percent'].diff() / time_interval_minutes
        metric_df = metric_df[['symbol', 'datetime', 'speed_of_change']].rename(columns={'speed_of_change': 'metric'})
        metric_label = "Speed of Change (% per minute)"
    elif calculation_method == "Volume-Weighted Percentage Change":
        # Calculate Volume-Weighted Percentage Change
        resampled_df['weighted_change'] = resampled_df['change_percent'] * resampled_df['volume']
        metric_df = resampled_df.copy()
        metric_df = metric_df.sort_values(['symbol', 'datetime'])
        # For each symbol and interval, calculate the Volume-Weighted Change
        metric_df['volume_weighted_change'] = metric_df.groupby('symbol')['weighted_change'].transform('sum') / metric_df.groupby('symbol')['volume'].transform('sum')
        metric_df = metric_df[['symbol', 'datetime', 'volume_weighted_change']].drop_duplicates()
        metric_df = metric_df.rename(columns={'volume_weighted_change': 'metric'})
        metric_label = "Volume-Weighted Change (%)"
    elif calculation_method == "Exponential Moving Average (EMA)":
        # Calculate EMA with span equal to a specified value (e.g., 10)
        span = 10  # You can adjust the span as needed
        metric_df = resampled_df.copy()
        metric_df = metric_df.sort_values(['symbol', 'datetime'])
        metric_df['EMA'] = metric_df.groupby('symbol')['change_percent'].transform(
            lambda x: x.ewm(span=span, adjust=False).mean()
        )
        metric_df = metric_df[['symbol', 'datetime', 'EMA']].rename(columns={'EMA': 'metric'})
        metric_label = f"Exponential Moving Average (EMA) - Span {span}"
    else:
        st.error("Invalid Calculation Method Selected.")
        st.stop()

    # -----------------------------
    # Pivot Data for Heatmap
    # -----------------------------
    # Ensure that 'metric' is numeric
    metric_df['metric'] = pd.to_numeric(metric_df['metric'], errors='coerce')

    # Pivot the DataFrame to have symbols as rows and time intervals as columns
    # 'datetime' should be sorted
    pivot_df = metric_df.pivot(index='symbol', columns='datetime', values='metric')

    # Sort the symbols based on the sort option
    if sort_option == "Percentage Change (Default)":
        # Sort by the latest metric value
        latest_metrics = pivot_df.iloc[:, -1]
        sorted_symbols = latest_metrics.sort_values(ascending=False).dropna().index.tolist()[:num_symbols]
    elif sort_option == "Alphabetically":
        # Sort alphabetically
        sorted_symbols = sorted(pivot_df.index.tolist())[:num_symbols]
    else:
        # Default to sorting by latest metric value
        latest_metrics = pivot_df.iloc[:, -1]
        sorted_symbols = latest_metrics.sort_values(ascending=False).dropna().index.tolist()[:num_symbols]

    # Filter the pivot DataFrame to include only the top symbols
    pivot_df = pivot_df.loc[sorted_symbols]

    # Drop any symbols with all NaN metrics
    pivot_df.dropna(how='all', inplace=True)

    if pivot_df.empty:
        st.warning("No data available for the selected options.")
        st.stop()

    # -----------------------------
    # Enhance Symbol Labels with Percentage Change
    # -----------------------------
    # Get the latest metric value for each symbol
    latest_metrics = pivot_df.iloc[:, -1]

    # Handle symbols that might not have a latest metric (dropna)
    latest_metrics = latest_metrics.dropna()

    # For all symbols in pivot_df.index, get the latest metric or set to N/A if missing
    symbols_with_change = []
    for symbol in pivot_df.index:
        if symbol in latest_metrics:
            change = latest_metrics[symbol]
            symbols_with_change.append(f"{symbol} ({change:+.2f}%)")
        else:
            symbols_with_change.append(f"{symbol} (N/A)")

    # -----------------------------
    # Prepare Data for Plotly Heatmap
    # -----------------------------
    heatmap_data = pivot_df.values
    time_labels = [dt.strftime("%H:%M") for dt in pivot_df.columns]

    # Create hover text
    hover_text = []
    for i, symbol in enumerate(pivot_df.index):
        row = []
        for j, time_label in enumerate(time_labels):
            value = heatmap_data[i][j]
            if not np.isnan(value):
                row.append(f"Symbol: {symbol}<br>Time: {time_label}<br>Change: {value:.2f}%")
            else:
                row.append(f"Symbol: {symbol}<br>Time: {time_label}<br>Change: N/A")
        hover_text.append(row)

    # -----------------------------
    # Plot Heatmap using Plotly
    # -----------------------------
    # Construct additional details for the title
    additional_details = f"Calculation Method: {calculation_method} | Time Interval: {time_interval_label} | From: {selected_start_time.strftime('%H:%M')} To: {selected_end_time.strftime('%H:%M')}"

    # Display the main title
    st.markdown(f"## {metric_label} Heatmap for {selected_date} - Top {num_symbols} Symbols")
    # Display the additional details below the main title
    st.markdown(f"<div style='text-align: center;'><sup>{additional_details}</sup></div>", unsafe_allow_html=True)

    # Define the custom 10-color scale with adjusted boundaries (removed black color for 0%)
    custom_colorscale = [
        [0.0, '#8B0000'],   # Dark Red for -10% to -8%
        [0.09, '#8B0000'],  # Dark Red
        [0.1, '#FF0000'],   # Red for -8% to -6%
        [0.19, '#FF0000'],  # Red
        [0.2, '#FF4500'],   # Orange Red for -6% to -4%
        [0.29, '#FF4500'],  # Orange Red
        [0.3, '#FFA500'],   # Orange for -4% to -2%
        [0.39, '#FFA500'],  # Orange
        [0.4, '#FFD700'],   # Yellow for -2% to 0%
        [0.49, '#FFD700'],  # Yellow
        [0.5, '#ADFF2F'],   # Green Yellow for 0% to 2%
        [0.59, '#ADFF2F'],  # Green Yellow
        [0.6, '#7CFC00'],   # Lawn Green for 2% to 4%
        [0.69, '#7CFC00'],  # Lawn Green
        [0.7, '#32CD32'],   # Lime Green for 4% to 6%
        [0.79, '#32CD32'],  # Lime Green
        [0.8, '#228B22'],   # Forest Green for 6% to 8%
        [0.89, '#228B22'],  # Forest Green
        [0.9, '#006400'],   # Dark Green for 8% to 10%
        [1.0, '#006400']    # Dark Green
    ]

    # Generate the colorscale list for Plotly
    colorscale = custom_colorscale

    fig = go.Figure()

    # Add the Heatmap trace with discrete color bands
    fig.add_trace(go.Heatmap(
        z=heatmap_data,
        x=time_labels,
        y=symbols_with_change,
        colorscale=colorscale,
        zmin=-10,
        zmax=10,
        colorbar=dict(
            title=dict(
                text="Percentage Change",
                side="top"  # Position the title above the color bar
            ),
            tickmode="array",
            tickvals=[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10],
            ticktext=["-10%", "-8%", "-6%", "-4%", "-2%", "0%", "2%", "4%", "6%", "8%", "10%"],
            ticks="outside",
            orientation='h',
            x=0.5,               # Center the colorbar horizontally
            xanchor='center',
            y=-0.04,             # Slightly below the heatmap
            yanchor='top',
            lenmode='fraction',
            len=0.8,             # Adjust the width of the color bar
            xpad=0,
            ypad=0               # Remove extra padding
        ),
        hoverinfo='text',
        text=hover_text,
        showscale=True
    ))

    # Update layout to adjust the y-axis space and show solid bands
    fig.update_layout(
        autosize=True,
        height=max(600, 25 * len(symbols_with_change)),  # Dynamically adjust height based on the number of symbols
        xaxis_title='Time Interval',
        yaxis_title='Symbols',
        title=f"{metric_label} Heatmap for {selected_date}",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(symbols_with_change))),
            ticktext=symbols_with_change,
            tickfont=dict(size=12),  # Adjust font size as needed
            tickangle=0,             # Keep text horizontal
            automargin=True,         # Remove unnecessary padding
        ),
        margin=dict(l=0, r=0, t=50, b=20)  # Minimize margins for maximum heatmap area
    )

    # Display the heatmap in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Display Metric Value
    # -----------------------------
    st.markdown("### Selected Metric:")
    st.write(f"{metric_label} displayed in the heatmap.")

else:
    st.warning("No data available in the database.")
