import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import os
import pytz
from datetime import datetime, timedelta, time as dt_time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time

# Set Streamlit page configuration for better layout
st.set_page_config(layout="wide", page_title="Stock Heatmap Dashboard")

# -----------------------------
# Configuration Variables
# -----------------------------

DATABASE_PATH = 'stock_data.db'
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

# Comprehensive Colormap Options for Theme Selection
cmaps = [
    ('Perceptually Uniform Sequential', [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
    ('Sequential', [
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
    ('Sequential (2)', [
        'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
        'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
        'hot', 'afmhot', 'gist_heat', 'copper']),
    ('Diverging', [
        'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
        'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
    ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
    ('Qualitative', [
        'Pastel1', 'Pastel2', 'Paired', 'Accent',
        'Dark2', 'Set1', 'Set2', 'Set3',
        'tab10', 'tab20', 'tab20b', 'tab20c']),
    ('Miscellaneous', [
        'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
        'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
        'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
        'gist_ncar'])
]

# -----------------------------
# Functions
# -----------------------------
def get_data_from_db(database_path):
    """Load data from the SQLite database."""
    if not os.path.isfile(database_path):
        st.error(f"Database file not found at: {database_path}")
        return None
    try:
        conn = sqlite3.connect(database_path)
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

def get_custom_colorscale(cmap_name):
    """
    Converts a Matplotlib colormap to a Plotly-compatible colorscale.
    Specifically handles the custom 'blue2green20' as per user requirements.
    """
    if cmap_name == 'blue2green20':
        # Define custom 'blue2green20' colorscale: dark blue -> light blue -> light green -> dark green
        colorscale = [
            [0.0, '#00008B'],    # Dark Blue
            [0.05263, '#131895'],
            [0.10526, '#262C9F'],
            [0.15789, '#3948A9'],
            [0.21053, '#4C60B3'],
            [0.26316, '#5F78BD'],
            [0.31579, '#7290C7'],
            [0.36842, '#85A8D1'],
            [0.42105, '#98C0DB'],
            [0.47368, '#ABD8E6'],  # Light Blue
            [0.52632, '#90EE90'],  # Light Green
            [0.57895, '#80DE80'],
            [0.63158, '#70CE70'],
            [0.68421, '#60BE60'],
            [0.73684, '#50AE50'],
            [0.78947, '#409E40'],
            [0.84211, '#308E30'],
            [0.89474, '#207E20'],
            [0.94737, '#106E10'],
            [1.0, '#006400']       # Dark Green
        ]
        return colorscale
    try:
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(np.linspace(0, 1, 256))[:, :3]  # RGB values
        colors = [mcolors.to_hex(c) for c in colors]
        # Create list of [normalized value, color] pairs
        colorscale = [[i/255, color] for i, color in enumerate(colors)]
        return colorscale
    except ValueError:
        # If cmap_name is not a valid Matplotlib colormap, assume it's a Plotly built-in colorscale
        return cmap_name

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
# Load and Process Data
# -----------------------------
df = get_data_from_db(DATABASE_PATH)

if df is not None:
    df = convert_timestamp(df, LOCAL_TIMEZONE)
    # Ensure numeric columns are properly typed
    df['change_percent'] = pd.to_numeric(df['change_percent'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    # Drop rows with NaN in essential columns
    df.dropna(subset=['change_percent', 'close', 'volume'], inplace=True)

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
    selected_date = st.sidebar.date_input(
        "Select Date",
        value=datetime.now(pytz.timezone(LOCAL_TIMEZONE)).date(),
        min_value=df['datetime'].dt.date.min(),
        max_value=df['datetime'].dt.date.max(),
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
        st.error("Start time must be earlier than end time.")
        st.stop()

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
        "10 seconds": 10,
        "30 seconds": 30,
        "45 seconds": 45,
        "1 minute": 60,
        "2 minutes": 120,
        "3 minutes": 180,
        "5 minutes": 300,
        "10 minutes": 600
    }

    selected_refresh_label = st.sidebar.selectbox(
        "Select Sync Interval",
        options=list(REFRESH_OPTIONS.keys()),
        index=3,  # Default to '1 minute'
        help="Choose how frequently the data and heatmap refresh."
    )
    refresh_interval = REFRESH_OPTIONS[selected_refresh_label]

    # **6. Visualization Options: Color Theme** (Moved into Controls Section)
    st.sidebar.markdown("### Visualization Options")

    # Define available color themes for heatmap visualization
    color_map_options = []
    for category, maps in cmaps:
        for cmap in maps:
            display_name = f"{category} - {cmap}"
            color_map_options.append(display_name)

    # Add a custom color scale at the top of the list
    color_map_options.insert(0, 'Custom - blue2green20')

    selected_cmap_display = st.sidebar.selectbox(
        "Select Color Theme",
        options=color_map_options,
        index=0,  # Default to 'Custom - blue2green20'
        help="Choose a color theme for the heatmap."
    )

    # Determine the actual colormap to use
    if selected_cmap_display == 'Custom - blue2green20':
        selected_cmap = 'blue2green20'
    else:
        selected_cmap = selected_cmap_display.split(' - ')[-1]

    # -----------------------------
    # Determine Start and End Times
    # -----------------------------
    # Combine selected date with start and end times
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

    # -----------------------------
    # Filter Data Based on Date and Time Range
    # -----------------------------
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
    metric_display = ""

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

    # Determine if the selected_cmap is Plotly built-in or needs a custom colorscale
    plotly_colorscales = [
        'aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance', 'blackbody', 'bluered', 'blues', 'blugrn',
        'bluyl', 'brbg', 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl', 'darkmint',
        'deep', 'delta', 'dense', 'earth', 'edge', 'electric', 'emrld', 'fall', 'geyser', 'gnbu',
        'gray', 'greens', 'greys', 'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
        'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges', 'orrd', 'oryel', 'oxy',
        'peach', 'phase', 'picnic', 'pinkyl', 'piyg', 'plasma', 'plotly3', 'portland', 'prgn',
        'pubu', 'pubugn', 'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdgy',
        'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar', 'spectral', 'speed', 'sunset', 'sunsetdark',
        'teal', 'tealgrn', 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid', 'turbo', 'twilight',
        'twilight_shifted', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd'
    ]

    if selected_cmap in plotly_colorscales:
        # Use Plotly's built-in colorscale
        colorscale = selected_cmap
    elif selected_cmap == 'blue2green20':
        # Use the custom 'blue2green20' colorscale
        colorscale = get_custom_colorscale(selected_cmap)
    else:
        # Attempt to create a custom colorscale from Matplotlib
        colorscale = get_custom_colorscale(selected_cmap)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=time_labels,
        y=symbols_with_change,
        colorscale=colorscale,
        zmin=-10,              # Set minimum value of the color scale
        zmax=10,               # Set maximum value of the color scale
        colorbar=dict(
            title=dict(
                text="Metric Label",
                side="top"  # Position the title above the color bar
            ),
            orientation='h',
            x=0.5,              # Center the colorbar horizontally
            y=-0.04,            # Position very close to the x-axis
            yanchor='top',
            yref='paper',       # Reference the plot area for positioning
            lenmode='fraction', # Use fraction mode for dynamic sizing
            len=1.0,            # Make the color bar as wide as the heatmap
            xpad=0,
            ypad=0              # Remove any extra padding
        ),
        hoverinfo='text',
        text=hover_text,
        showscale=True
    ))

    # Update layout to tighten the y-axis space
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




    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Display Metric Value
    # -----------------------------
    st.markdown("### Selected Metric:")
    st.write(f"{metric_label} displayed in the heatmap.")

    # -----------------------------
    # Refresh the App Based on the Selected Interval
    # -----------------------------
    time.sleep(refresh_interval)
    st.rerun()

