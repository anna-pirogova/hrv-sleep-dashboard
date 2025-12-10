import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
from datetime import datetime, timedelta
import requests
segoe_ui = font_manager.FontProperties(fname="C:/Windows/Fonts/segoeui.ttf")

import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import collections
import numpy as np
api_key = st.secrets["openrouter"]["api_key"]

st.set_page_config(page_title="Recovery Dashboard", layout="wide")

DB_PATH = "mock_daily_summary.db"

@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM daily_summary", conn)
    conn.close()
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df["deep_pct"] = df["deep_minutes"] / df["time_in_bed"] * 100
    df["sleep_hours"] = df["time_in_bed"] / 60
    df["month_label"] = df["date"].dt.strftime("%b %Y")
    df["month_date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
        
    return df

df = load_data()

def add_rolling_mean_7days_features(df, cols, window=7, prefix="_rolling"):
    for col in cols:
        df[f"{col}{prefix}"] = df[col].rolling(window=window, min_periods=1).mean()
    return df

def add_rolling_mean_30days_features(df, cols, window=30, prefix="_mean_30d"):
    for col in cols:
        df[f"{col}{prefix}"] = df[col].rolling(window=window, min_periods=1).mean()
    return df

def add_rolling_std_30days_features(df, cols, window=30, prefix="_std_30d"):
    for col in cols:
        df[f"{col}{prefix}"] = df[col].rolling(window=window, min_periods=1).std()
    return df

def add_upper_band(df, cols, prefix='_upper_band'):
    for col in cols:
        df[f"{col}{prefix}"]=df[col+'_mean_30d']+df[col+'_std_30d']
    return df

def add_lower_band(df, cols, prefix='_lower_band'):
    for col in cols:
        df[f"{col}{prefix}"]=df[col+'_mean_30d']-df[col+'_std_30d']
    return df

df=add_rolling_mean_7days_features(df,['deepRmssd','dailyRmssd','deep_pct','restingHeartRate','breathing_rate','deep_minutes','rem_minutes'])
df=add_rolling_mean_30days_features(df,['deep_minutes','deepRmssd','dailyRmssd','restingHeartRate','breathing_rate','rem_minutes'])
df=add_rolling_std_30days_features(df,['deep_minutes','deepRmssd','dailyRmssd','restingHeartRate','breathing_rate','rem_minutes'])
df=add_upper_band(df,['deepRmssd','dailyRmssd','restingHeartRate','breathing_rate','deep_minutes','rem_minutes'])
df=add_lower_band(df,['deepRmssd','dailyRmssd','restingHeartRate','breathing_rate','deep_minutes','rem_minutes'])

def get_color_by_threshold(
    val, ref, std=None, 
    low_thresh=-3, high_thresh=3, 
    use_zscore=False, tight=False,
    invert=False,
    fallback_color="#b0b0b0"
):
    """
    General-purpose color function with optional logic inversion.

    Parameters:
    - val, ref, std: current value, reference value, and std (if available)
    - use_zscore: use Z-score logic (mean ± 1 std)
    - tight: use tighter thresholds (±1) *only* if default thresholds are used
    - invert: if True, swaps red/green logic
    - fallback_color: used when std is zero or NaN (in z-score mode)

    Returns:
    - '#de425b' = red (bad), '#488f31' = green (good), '#3594cc' = neutral
    """
    # Handle z-score mode
    if use_zscore:
        if std is None or std == 0 or pd.isna(std):
            return fallback_color
        z = (val - ref) / std
        if z > 1:
            return "#de425b" if invert else "#488f31"
        elif z < -1:
            return "#488f31" if invert else "#de425b"
        else:
            return "#3594cc"

    # Handle absolute delta mode
    delta = val - ref
    if tight and (low_thresh == -3 and high_thresh == 3):
        low_thresh, high_thresh = -1, 1

    if delta < low_thresh:
        return "#488f31" if invert else "#de425b"
    elif delta > high_thresh:
        return "#de425b" if invert else "#488f31"
    else:
        return "#3594cc"

def get_color_rhr_std(val, ref, std):
    delta = val - ref
    if delta > std:
        return '#de425b'
    elif delta < -std:
        return '#488f31'
    else:
        return '#3594cc'

def get_color_br_std(val, ref, std):
    """
    Color coding for Breathing Rate using deviation from rolling mean.
    Red = high BR (stress), Green = low BR (recovery), Blue = normal.
    """
    delta = val - ref

    # Breathing rate tends to vary within ±1.0, so use tighter thresholds
    if delta > std:
        return '#de425b'   # 🔴 Elevated BR → potential stress
    elif delta < -std:
        return '#488f31'   # 🟢 Lower BR → better recovery
    else:
        return '#3594cc'   # 🔵 Within normal variation

def get_color_deep_minutes_adaptive(val, ref, std):
    """
    Adaptive coloring for deep sleep minutes using rolling baseline (Z-score logic).
    - Green: significantly above baseline
    - Red: significantly below baseline
    - Blue: within normal variation
    """
    delta = val - ref
    if std == 0 or pd.isna(std):
        return "#b0b0b0"  # fallback for no variance

    z = delta / std

    if z > 1:
        return '#488f31'  # 🟢 Above normal → good
    elif z < -1:
        return '#de425b'  # 🔴 Below normal → bad
    else:
        return '#3594cc'  # 🔵 Normal

# Custom month-color mapping 
month_colors = {
    "Jan 2015": "#4f81bd",  # icy blue
    "Feb 2015": "#4aacc5",
    "Mar 2015": "#76c043",  # fresh green
    "Apr 2015": "#c6e2b3",
    "May 2015": "#ffd966",  # sunny yellow
    "Jun 2015": "#f6b26b",
    "Jul 2015": "#f79646",  # orange summer
    "Aug 2015": "#e06666",  # warm red
    "Sep 2015": "#b45f06",  # autumn brown
    "Oct 2015": "#a64d79",  # plum
    "Nov 2015": "#6fa8dc",  # cool sky
    "Dec 2015": "#3d85c6",  # winter blue
}
def convert_sleep_start_to_float(dt):
    hour = dt.hour + dt.minute / 60
    # if after midnight (e.g. 0–12 a.m.), push into next‑day range
    if hour < 12:
        hour += 24
    return hour

def plot_metric_with_band_plotly(
    df,
    value_col,
    date_col="date",
    title=None,
    ylabel=None,
    color_func=None,
    use_std=False,
    tight=False,
    use_zscore=False,
    fallback_color="#b0b0b0",
    my_colors=None,  # ✅ precomputed list like colors1
    height=500,
    color_func_kwargs=None,
):
    """
    Plot a value column with shaded ±1 std band and colored bars using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    value_col : str
        The name of the metric column to visualize (e.g., "restingHeartRate").
    date_col : str, default "date"
        Column name containing datetime values.
    title : str, optional
        Plot title (auto-generated if None).
    ylabel : str, optional
        Y-axis label (defaults to value_col).
    color_func : callable, optional
        A function for coloring bars (e.g., get_color_by_threshold).
    use_std : bool, default False
        If True, passes std to the color function (for z-score or deviation logic).
    tight : bool, default False
        If True, uses tighter thresholds (±1 instead of ±3).
    use_zscore : bool, default False
        If True, uses z-score logic for coloring (requires std).
    fallback_color : str, default "#b0b0b0"
        Color to use if std is 0 or NaN.
    my_colors : list or str, optional
        Manually specify bar colors (bypasses color_func logic).
    height : int, default 500
        Figure height.
    color_func_kwargs : dict, optional
        Extra arguments to pass to the color function (e.g., {"invert": True}).
    """

    if color_func_kwargs is None:
        color_func_kwargs = {}

    # Use default fallback color function if none is provided
    if color_func is None:
        def color_func(val, ref, std=None, **kwargs):
            delta = val - ref
            if std is None or std == 0 or pd.isna(std):
                return fallback_color
            z = delta / std
            if z > 1:
                return "#488f31"  # green
            elif z < -1:
                return "#de425b"  # red
            else:
                return "#3594cc"  # blue

    # Infer column names
    rolling_col = f"{value_col}_rolling"
    mean_col = f"{value_col}_mean_30d"
    std_col = f"{value_col}_std_30d"

    # Use precomputed color list if provided
    if my_colors is not None:
        colors = my_colors
    elif use_std:
        colors = [
            color_func(v, r, s, tight=tight, use_zscore=use_zscore,
                       fallback_color=fallback_color, **color_func_kwargs)
            for v, r, s in zip(df[value_col], df[rolling_col], df[std_col])
        ]
    else:
        colors = [
            color_func(v, r, tight=tight, use_zscore=use_zscore,
                       fallback_color=fallback_color, **color_func_kwargs)
            for v, r in zip(df[value_col], df[rolling_col])
        ]

    # Create Plotly figure
    fig = go.Figure()

    # Add shaded ±1 std band
    fig.add_trace(go.Scatter(
        x=pd.concat([df[date_col], df[date_col][::-1]]),
        y=pd.concat([df[f"{value_col}_upper_band"], df[f"{value_col}_lower_band"][::-1]]),
        fill='toself',
        fillcolor='rgba(176, 217, 242, 0.4)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=True,
        name='Normal Range (±1 SD)'
    ))

    # Add main bar values
    fig.add_trace(go.Bar(
        x=df[date_col],
        y=df[value_col],
        marker_color=colors,
        name=value_col,
        showlegend=False
    ))

    # Add rolling average line
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[rolling_col],
        mode='lines',
        line=dict(color="#FF5F1F", width=2.5),
        name="7-day Avg"
    ))

    # Layout and styling
    fig.update_layout(
        title=dict(
            text=title or f"{value_col} vs 7-day Rolling Avg",
            x=0.5, y=0.9, xanchor="center", yanchor="top"
        ),
        xaxis_title="Date",
        yaxis_title=ylabel or value_col,
        yaxis=dict(range=[df[value_col].min() * 0.9, df[value_col].max() * 1.05]),
        bargap=0.1,
        template='simple_white',
        legend=dict(x=0.01, y=0.99),
        height=height
    )

    fig.update_xaxes(
        tickformat="%b %d, %Y",
        tickangle=45,
        showgrid=True,
        gridcolor='rgba(200,200,200,0.3)'
    )

    return fig


def plot_monthly_boxplot_with_median_trend(
    df,
    value_col,
    month_label_col="month_label",
    month_date_col="month_date",
    title="Monthly Boxplot with Median Trend",
    yaxis_title=None,
    color_map=None,
    height=500,
    width=900,
    boxpoints=False,
    line_style="dot"
):
    """
    Plots a boxplot by month with a median trend line overlaid.

    Parameters:
    - df: pd.DataFrame
    - value_col: str, the column to plot (e.g., 'dailyRmssd')
    - month_label_col: str, column with month names (e.g., 'Jan 2025')
    - month_date_col: str, column used for correct sorting (e.g., datetime of month)
    - title: str, plot title
    - yaxis_title: str, optional y-axis label (defaults to value_col)
    - color_map: dict, optional mapping from month labels to fill/line color
    - height: int, figure height
    - width: int, figure width
    - boxpoints: bool or str, Plotly boxpoints parameter
    - line_style: str, e.g. "dot", "dash", "solid"
    """
    fig = go.Figure()
    medians = []

    # Ensure correct month order
    ordered_labels = (
        df[[month_label_col, month_date_col]]
        .drop_duplicates()
        .sort_values(month_date_col)[month_label_col]
        .tolist()
    )

    for label in ordered_labels:
        month_data = df[df[month_label_col] == label][value_col]
        color = color_map.get(label, "lightgray") if color_map else "lightgray"
        fig.add_trace(go.Box(
            y=month_data,
            x=[label] * len(month_data),
            name=label,
            boxpoints=boxpoints,
            fillcolor=color,
            marker_color=color,
            line=dict(width=1)
        ))
        medians.append(month_data.median())

    # Add median trend line
    fig.add_trace(go.Scatter(
        x=ordered_labels,
        y=medians,
        mode="lines+markers+text",
        name="Median Trend",
        line=dict(color="black", dash=line_style),
        marker=dict(symbol="circle", size=8, color="black"),
        text=[f"{m:.1f}" for m in medians],
        textposition="top center",
        textfont=dict(color="black", size=12)
    ))

    fig.update_layout(
        title=dict(
            text=title,
            y=0.9,
            x=0.5,
            xanchor="center",
            yanchor="top"
        ),
        yaxis_title=yaxis_title or value_col,
        xaxis_title="Month",
        template="simple_white",
        height=height,
        width=width
    )

    return fig


def plot_simple_scatter(
    df,
    x_col,
    y_col,
    title=None,
    x_label=None,
    y_label=None,
    width=600,
    height=400,
    color='indigo',
    opacity=0.7,
    marker_size=8
):
    """
    Creates a simple scatterplot without a trendline.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with x and y data.
    x_col : str
        Column name for x-axis.
    y_col : str
        Column name for y-axis.
    title : str
        Plot title.
    x_label : str
        Label for x-axis (optional).
    y_label : str
        Label for y-axis (optional).
    width : int
        Width of the plot.
    height : int
        Height of the plot.
    color : str
        Marker color (e.g., "royalblue", "seagreen").
    opacity : float
        Opacity of markers.
    marker_size : int
        Size of the dots.
    """
    # Drop missing values
    df_clean = df.dropna(subset=[x_col, y_col])

    # Create scatter plot
    fig = px.scatter(
        df_clean,
        x=x_col,
        y=y_col,
        title=title or f"{y_col} vs {x_col}",
        labels={
            x_col: x_label or x_col,
            y_col: y_label or y_col
        },
        opacity=opacity,
        color_discrete_sequence=[color]  # Pass color as a single-item list
    )

    # Customize marker size
    fig.update_traces(marker=dict(size=marker_size))

    # Layout adjustments
    fig.update_layout(
        title_font_size=20,
        width=width,
        height=height,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        margin=dict(l=10, r=10, t=40, b=10)
    )

    return fig

# AI coach
def call_openrouter_chat(prompt, model="meta-llama/llama-3-8b-instruct", api_key=None):
    if not api_key:
        raise ValueError("Please provide your OpenRouter API key")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "your-site.com",  # Optional
        "X-Title": "your-app-name"        # Optional
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)

    if response.status_code != 200:
        print(response.text)
        raise Exception(f"API Error {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"]

def build_metric_series_csv(df_filtered, metric, date_col='date'):
    """
    Builds a compressed, CSV-style AI-friendly series for a metric including:
    - Raw value
    - Comparison to 7-day and 30-day bands (below/within/above)
    - Smoothed trend direction using 3-day rolling mean
    - Turning point detection
    - Trend streak length

    Returns:
        dict: {
            "metric": metric_name,
            "header": CSV header line,
            "csv_rows": [str, str, ...]  # each row: date,value,vs_7d,vs_30d,trend,turning_point,streak
        }
    """
    df = df_filtered.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    col_low = f"{metric}_lower_band"
    col_high = f"{metric}_upper_band"

    required_cols = [metric, col_low, col_high]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.sort_values(date_col).reset_index(drop=True)

    # 3-day rolling means
    df['rolling_mean_3'] = df[metric].rolling(window=3).mean()
    df['rolling_mean_3_shifted'] = df['rolling_mean_3'].shift(3)

    csv_rows = []
    previous_trend = None
    streak = 0

    for _, row in df.iterrows():
        if pd.isna(row[metric]) or pd.isna(row[col_low]) or pd.isna(row[col_high]):
            continue

        date = row[date_col].strftime('%Y-%m-%d')
        value = round(row[metric], 2)

        # Compare to 7-day band
        if value < row[col_low]:
            status_7d = "below"
        elif value > row[col_high]:
            status_7d = "above"
        else:
            status_7d = "within"

        # Reuse 7-day status for 30-day (can change later)
        status_30d = status_7d

        current_avg = row['rolling_mean_3']
        prev_avg = row['rolling_mean_3_shifted']

        if pd.isna(current_avg) or pd.isna(prev_avg):
            trend = "stable"
            turning_point = False
            streak = 0
        else:
            if current_avg > prev_avg:
                trend = "rising"
            elif current_avg < prev_avg:
                trend = "falling"
            else:
                trend = "stable"

            turning_point = previous_trend is not None and trend != previous_trend
            streak = 1 if turning_point else streak + 1
            previous_trend = trend

        row_str = f"{date},{value},{status_7d},{status_30d},{trend},{turning_point},{streak}"
        csv_rows.append(row_str)

    return {
        "metric": metric,
        "header": "date,value,vs_7d_range,vs_30d_range,trend,turning_point,trend_streak",
        "csv_rows": csv_rows
    }

def generate_insight_prompt_from_csv_series(
    metric_csv_result,
    metric_label=None,
    n_days=None,
    custom_instruction=None
):
    """
    Builds a full prompt for LLM insight generation from compact metric CSV series.
    """
    header = metric_csv_result["header"]
    rows = metric_csv_result["csv_rows"]
    metric_name = metric_csv_result["metric"]
    metric_label = metric_label or metric_name

    if n_days is not None:
        rows = rows[-n_days:]

    default_instruction = f"""
You are a recovery insights coach like the AI behind WHOOP.

You are given {metric_label} time-series data in CSV format:
Each row = date, value, 7-day range status, 30-day range status, smoothed trend, turning point flag, trend streak length.

Your job is to write a very short recovery insight that sounds like a professional wearable app notification.
No sections. No headings. Just 4–5 clear sentences max, in natural language. Use a helpful, signal-aware tone.

Avoid vague phrases like "a few times", "some changes", or "recent". Use specific observations from the data (e.g., "3 out of last 7 days", "value increased by X", etc.) when possible.
Use comparisons like "upward trend over the past 5 days" or "2 days below the lower threshold". Mention exact direction or magnitude of change if available.

Use this format:
One-sentence summary of the overall trend.
One sentence on any above/below range signals.
One sentence on turning point patterns if they matter.
One sentence of interpretation: load vs recovery.
One short tip: what to watch or do next (no health advice).

Avoid any of the following:
- Dates or timestamps
- Explanations of the data structure
- Definitions or formulas
- Emotional speculation (e.g., "stress" or "anxiety")
- Generic wellness advice (e.g., sleep more)

Example:
"Deep Sleep HRV has shown a clear upward trend over the last 6 days. 2 of these were above your 30-day range, indicating enhanced recovery. A turning point occurred 4 days ago after a stable low period. This suggests your parasympathetic activity has rebounded. Monitor if the current rise continues or stabilizes."

Here is the data:
""".strip()

    instruction = custom_instruction or default_instruction
    csv_block = "\n".join([header] + rows)

    return f"{instruction}\n\n{csv_block}"

@st.cache_data(show_spinner="Generating insight...")
def generate_metric_insight_from_df(
    df_filtered,
    value_col,
    label=None,
    rolling_col=None,  
    std_col=None,      
    api_key=None,
    model="meta-llama/llama-3.3-70b-instruct",
    max_days=60
):
    # Build CSV-style summary
    result = build_metric_series_csv(df_filtered, metric=value_col)

    # Generate prompt
    prompt = generate_insight_prompt_from_csv_series(
        result,
        metric_label=label or value_col,
        n_days=max_days
    )

    # Call the LLM via your API
    response = call_openrouter_chat(prompt, model=model, api_key=api_key)
    return response


# ---------------------
# Dashboard Header
# ---------------------
st.title("Recovery Dashboard")
st.markdown("Analyze HRV, sleep, and recovery trends across metrics and months.")

with st.expander("Key Metrics Explained"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
**Daily HRV RMSSD**  
- **What it is**: A measure of the variation in time between heartbeats measured during the day, reflecting autonomic nervous system balance.  
- **Why it matters**: Higher HRV generally indicates better recovery, resilience, and cardiovascular health.

**Deep Sleep HRV RMSSD**  
- **What it is**: HRV calculated *only during deep sleep stages*.  
- **Why it matters**: It minimizes external factors (movement, stress), giving a cleaner signal of biological recovery.
        """)

    with col2:
        st.markdown("""
**Deep Sleep Duration**  
- **What it is**: Total minutes spent in deep sleep (slow-wave sleep) per night.  
- **Why it matters**: Essential for physical recovery, hormonal regulation, and immune function.

**Total Sleep Time**  
- **What it is**: Total time spent asleep across all stages (deep, light, REM).  
- **Why it matters**: More sleep allows better systemic recovery — especially when combined with quality deep/REM sleep.

**Tip:** Use these metrics together. For example, longer sleep + stable HRV = good recovery. But if sleep increases and HRV drops, stress or illness may be present.
        """)


# ---------------------
# Tabs for each section
# ---------------------
tab4, tab1, tab2, tab3 = st.tabs([
    "Summary", 
    "Trendlines", 
    "Monthly Distributions", 
    "Scatterplots"
])


# ===========================================================
# 📈 TAB 1 — TRENDLINES
# ===========================================================
with tab1:
    st.header("Trendlines")
    
    # Preset choices
    preset_options = [
        "All Time",
        "Last 7 days",
        "Last 30 days",
        "Last 3 months",
        "Last 6 months",
        "Custom Range"
    ]
    default_index = preset_options.index("Last 30 days")
    preset = st.selectbox("Select a date range:", preset_options,index=default_index)

    today = df["date"].max()

    if preset == "All Time":
        start_date = df["date"].min()
        end_date = today
    elif preset == "Last 7 days":
        start_date = today - timedelta(days=7)
        end_date = today
    elif preset == "Last 30 days":
        start_date = today - timedelta(days=30)
        end_date = today
    elif preset == "Last 3 months":
        start_date = today - pd.DateOffset(months=3)
        end_date = today
    elif preset == "Last 6 months":
        start_date = today - pd.DateOffset(months=6)
        end_date = today
    else:  # Custom Range
        start_date, end_date = st.date_input(
            "Pick a custom date range:",
            value=(df["date"].min(), df["date"].max()),
            min_value=df["date"].min(),
            max_value=df["date"].max()
        )

    # Filter the dataframe
    df_filtered = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]

    st.write(f"Showing data from **{start_date.strftime('%Y-%m-%d')}** to **{end_date.strftime('%Y-%m-%d')}**")


    col1, col2 = st.columns(2)
    
    with col1:
        # TODO: replace with your Plotly figure
        fig_daily_rmssd=plot_metric_with_band_plotly(
        df_filtered,
        value_col="dailyRmssd",
        title="Daily HRV vs 7-day Rolling Average",
        ylabel="HRV RMSSD (ms)",
        color_func=get_color_by_threshold,
        use_std=False)
        
        st.plotly_chart(fig_daily_rmssd, width="content")

        response = generate_metric_insight_from_df(
        df_filtered,                
        value_col='dailyRmssd',           
        label='Daily HRV',       
        api_key=api_key,    
        model='meta-llama/llama-3.3-70b-instruct',  
        max_days=90)
        
        st.subheader("AI-Generated Insight")
        st.markdown(response)
    with col2:
        fig_deep_rmssd=plot_metric_with_band_plotly(
        df_filtered,
        value_col="deepRmssd",
        title="Deep Sleep HRV vs 7-day Rolling Average",
        ylabel="HRV RMSSD (ms)",
        color_func=get_color_by_threshold,
        use_std=False)
    
        st.plotly_chart(fig_deep_rmssd, width="content")
        response = generate_metric_insight_from_df(
        df_filtered,                # your filtered DataFrame
        value_col='deepRmssd',           # the main metric column
        label='Deep Sleep HRV',       # optional human-readable name
        api_key=api_key,    
        model='meta-llama/llama-3.3-70b-instruct',  # optional, can use default
        max_days=90)
        
        st.subheader("AI-Generated Insight")
        st.markdown(response)
    with col1:
        # TODO
        colorsHR = [
        get_color_rhr_std(val, ref, std)
        for val, ref, std in zip(
            df_filtered["restingHeartRate"],
            df_filtered["restingHeartRate_rolling"],
            df_filtered["restingHeartRate_std_30d"]
        )]

        fig_resting_hr=plot_metric_with_band_plotly(
        df_filtered,
        value_col="restingHeartRate",
        my_colors=colorsHR,
        title="Resting HR vs 7-day Rolling Average",
        ylabel="Resting Heart Rate (beats per minute)")

        st.plotly_chart(fig_resting_hr, width="content")
        response = generate_metric_insight_from_df(
        df_filtered,                
        value_col='restingHeartRate',           
        label='Resting Heart Rate',       
        api_key=api_key,    
        model='meta-llama/llama-3.3-70b-instruct',  
        max_days=90)
        
        st.subheader("AI-Generated Insight")
        st.markdown(response)
    with col2:
        # TODO
        colors_br_r = [
        get_color_br_std(val, ref, std)
        for val, ref, std in zip(
            df_filtered["breathing_rate"],
            df_filtered["breathing_rate_rolling"],
            df_filtered["breathing_rate_std_30d"]
        )]

        fig_breathing_rate=plot_metric_with_band_plotly(
            df_filtered,
            value_col="breathing_rate",
            my_colors=colors_br_r,
            title="Breathing rate vs 7-day Rolling Average",
            ylabel="Breathing Rate (times per minute)"
        )
        
        st.plotly_chart(fig_breathing_rate, width="content")

    with col1:
        # TODO
        colors_ds = [
        get_color_deep_minutes_adaptive(val, ref, std)
        for val, ref, std in zip(
            df_filtered["deep_minutes"],
            df_filtered["deep_minutes_mean_30d"],
            df_filtered["deep_minutes_std_30d"]
        )]

        fig_deep_minutes=plot_metric_with_band_plotly(
            df_filtered,
            value_col="deep_minutes",
            my_colors=colors_ds,
            title="Deep sleep minutes vs 7-day Rolling Average",
            ylabel="Deep Sleep (mins)"
        )
        
        st.plotly_chart(fig_deep_minutes, width="content")

    with col2:
        colors_rem = [
        get_color_deep_minutes_adaptive(val, ref, std)
        for val, ref, std in zip(
            df_filtered["rem_minutes"],
            df_filtered["rem_minutes_mean_30d"],
            df_filtered["rem_minutes_std_30d"]
        )]
        fig_rem_minutes=plot_metric_with_band_plotly(
            df_filtered,
            value_col='rem_minutes',
            my_colors=colors_rem,
            title='REM sleep minutes vs 7-day Rolling Average',
            ylabel="REM Sleep (mins)"

        )
        st.plotly_chart(fig_rem_minutes, width='content')
# ===========================================================
# 📊 TAB 2 — MONTHLY DISTRIBUTIONS
# ===========================================================
with tab2:
    st.header("Monthly Distributions")

    col1, col2 = st.columns(2)

    
    with col1:
        # TODO
        fig_monthly_rmssd_daily=plot_monthly_boxplot_with_median_trend(
        df,
        value_col="dailyRmssd",
        title="Daily RMSSD by Month",
        yaxis_title="Daily HRV RMSSD (ms)",
        color_map=month_colors)

        st.plotly_chart(fig_monthly_rmssd_daily, width="content")

    with col2:
        # TODO
        fig_monthly_rmssd_deep=plot_monthly_boxplot_with_median_trend(
        df,
        value_col="deepRmssd",
        title="Deep RMSSD by Month",
        yaxis_title="Deep HRV RMSSD (ms)",
        color_map=month_colors)
        st.plotly_chart(fig_monthly_rmssd_deep, width="content")

    with col1:
        # TODO
        fig_monthly_resting_hr=plot_monthly_boxplot_with_median_trend(
        df,
        value_col="restingHeartRate",
        title="Resting HR by Month",
        yaxis_title="Resting HR (beats per minute)",
        color_map=month_colors)
        st.plotly_chart(fig_monthly_resting_hr, width="content")

    with col2:
        # TODO
        fig_monthly_breathing_rate=plot_monthly_boxplot_with_median_trend(
        df,
        value_col="breathing_rate",
        title="Breathing Rate by Month",
        yaxis_title="Breathing Rate (times per minute)",
        color_map=month_colors)
        st.plotly_chart(fig_monthly_breathing_rate, width="content")

    with col1:
        # TODO
        fig_monthly_temp_dev=plot_monthly_boxplot_with_median_trend(
        df,
        value_col="temperature_deviation",
        title="Temperature Deviation by Month",
        yaxis_title="Temperature Deviation (relative to baseline, degrees Celcius)",
        color_map=month_colors)
        st.plotly_chart(fig_monthly_temp_dev, width="content")

    with col2:
        # Helper: convert float hour to HH:MM string
        def float_to_time_str(val):
            hours = int(val) % 24
            minutes = int((val - int(val)) * 60)
            return f"{hours:02d}:{minutes:02d}"

        # Preprocess data
        df["sleep_start_hour"] = pd.to_datetime(df["sleep_start_time"]).apply(convert_sleep_start_to_float)
        ordered_labels = df[["month_label", "month_date"]].drop_duplicates().sort_values("month_date")["month_label"]
        x_labels = ordered_labels.tolist()

        # Tick formatting for custom Y-axis (hours 20:00–06:00 = 20–30)
        tickvals = np.arange(20, 30.5, 0.5)
        ticktext = [float_to_time_str(v) for v in tickvals]

        # Compute medians and their formatted labels
        medians = []
        median_texts = []
        y_offset = 0.25  # Raise text slightly above box
        adjusted_y = []

        for label in x_labels:
            month_data = df[df["month_label"] == label]["sleep_start_hour"]
            median_val = month_data.median()
            medians.append(median_val)
            median_texts.append(float_to_time_str(median_val))
            adjusted_y.append(median_val + y_offset)

        # Build Plotly figure
        fig = go.Figure()

        # Boxplots for each month (no hover)
        for label in x_labels:
            month_data = df[df["month_label"] == label]["sleep_start_hour"]
            fig.add_trace(go.Box(
                y=month_data,
                x=[label] * len(month_data),
                name=label,
                boxpoints=False,
                fillcolor=month_colors.get(label, "lightgray"),
                marker_color=month_colors.get(label, "gray"),
                line=dict(width=1),
                hoverinfo="skip"  # Disable hover
            ))

        # Median line
        fig.add_trace(go.Scatter(
            x=x_labels,
            y=medians,
            mode="lines+markers+text",
            name="Median Trend",
            line=dict(color="black", dash="dot"),
            marker=dict(symbol="circle", size=8, color="black"),
            textposition="top center",
            textfont=dict(color="black", size=12)
        ))

        # Add HH:MM labels above boxes
        fig.add_trace(go.Scatter(
            x=x_labels,
            y=adjusted_y,
            mode="text",
            text=median_texts,
            textposition="top center",
            showlegend=False,
            textfont=dict(size=12, color="black")
        ))

        # Layout settings
        fig.update_layout(
            title=dict(
                text="Sleep Start Time by Month",
                x=0.5, y=0.9,
                xanchor='center',
                yanchor='top'
            ),
            yaxis_title="Sleep Start Time (in Hours)",
            xaxis_title="Month",
            template="simple_white",
            width=900,
            height=500
        )

        # Y-axis with time grid
        fig.update_yaxes(
            tickvals=tickvals,
            ticktext=ticktext,
            range=[20, 30],
            title_text="Sleep Start Time",
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)',
            gridwidth=1
        )

        
        st.plotly_chart(fig, width="content")

    with col1:
        # TODO
        fig_monthly_sleep_hours=plot_monthly_boxplot_with_median_trend(
        df,
        value_col="sleep_hours",
        title="Sleep Hours by Month",
        yaxis_title="Sleep Hours",
        color_map=month_colors)
        st.plotly_chart(fig_monthly_sleep_hours, width="content")
    
    with col2:
        fig_monthly_deep_minutes=plot_monthly_boxplot_with_median_trend(
        df,
        value_col="deep_minutes",
        title="Deep Sleep Minutes by Month",
        yaxis_title="Deep Sleep (minutes)",
        color_map=month_colors)
        st.plotly_chart(fig_monthly_deep_minutes, width="content")

    with col1:
        fig_monthly_rem_minutes=plot_monthly_boxplot_with_median_trend(
        df,
        value_col="rem_minutes",
        title="REM Sleep by Month",
        yaxis_title="REM Sleep (minutes)",
        color_map=month_colors)
        st.plotly_chart(fig_monthly_rem_minutes, width="content")
    
    with col2:
        fig_monthly_light_minutes=plot_monthly_boxplot_with_median_trend(
        df,
        value_col="light_minutes",
        title="Light Sleep by Month",
        yaxis_title="Light Sleep (minutes)",
        color_map=month_colors)
        st.plotly_chart(fig_monthly_light_minutes, width="content")

    with col1:
        fig_monthly_awake_minutes=plot_monthly_boxplot_with_median_trend(
        df,
        value_col="minutes_awake",
        title="Awake Mins by Month",
        yaxis_title="Awake Time (minutes)",
        color_map=month_colors)
        st.plotly_chart(fig_monthly_awake_minutes, width="content")

# ===========================================================
# 🔬 TAB 3 — SCATTERPLOTS
# ===========================================================
with tab3:
    st.header("Scatterplots")

    scatter_pairs = [
        # Group 1: dailyRmssd
        ("dailyRmssd", "deep_minutes", "Daily RMSSD vs Deep Sleep Minutes"),
        ("dailyRmssd", "rem_minutes", "Daily RMSSD vs REM Sleep Minutes"),
        ("dailyRmssd", "sleep_hours", "Daily RMSSD vs Sleep Hours"),
        ("dailyRmssd", "sleep_start_hour", "Daily RMSSD vs Bedtime"),
        ("dailyRmssd", "restingHeartRate", "Daily RMSSD vs Resting HR"),
        ("dailyRmssd", "breathing_rate", "Daily RMSSD vs Breathing Rate"),
        ("dailyRmssd", "temperature_deviation", "Daily RMSSD vs Temperature Deviation"),

        # Group 2: deepRmssd
        ("deepRmssd", "deep_minutes", "Deep RMSSD vs Deep Sleep Minutes"),
        ("deepRmssd", "rem_minutes", "Deep RMSSD vs REM Sleep Minutes"),
        ("deepRmssd", "sleep_hours", "Deep RMSSD vs Sleep Hours"),
        ("deepRmssd", "sleep_start_hour", "Deep RMSSD vs Bedtime"),
        ("deepRmssd", "restingHeartRate", "Deep RMSSD vs Resting HR"),
        ("deepRmssd", "breathing_rate", "Deep RMSSD vs Breathing Rate"),
        ("deepRmssd", "temperature_deviation", "Deep RMSSD vs Temperature Deviation"),

        # Group 3: restingHeartRate
        ("restingHeartRate", "deep_minutes", "Resting HR vs Deep Sleep Minutes"),
        ("restingHeartRate", "rem_minutes", "Resting HR vs REM Sleep Minutes"),
        ("restingHeartRate", "sleep_hours", "Resting HR vs Sleep Hours"),
        ("restingHeartRate", "sleep_start_hour", "Resting HR vs Bedtime"),
        ("restingHeartRate", "breathing_rate", "Resting HR vs Breathing Rate"),
        ("restingHeartRate", "temperature_deviation", "Resting HR vs Temperature Deviation"),

        # Group 4: others
        ("deep_minutes", "rem_minutes", "Deep Sleep Minutes vs REM Sleep Minutes"),
        ("deep_minutes", "sleep_start_hour", "Deep Sleep Minutes vs Bedtime"),
        ("deep_minutes", "breathing_rate", "Deep Sleep Minutes vs Breathing Rate"),
        ("deep_minutes", "temperature_deviation", "Deep Sleep Minutes vs Temperature Deviation"),
        ("rem_minutes", "sleep_start_hour", "REM Sleep Minutes vs Bedtime"),
        ("rem_minutes", "breathing_rate", "REM Sleep Minutes vs Breathing Rate"),
        ("rem_minutes", "temperature_deviation", "REM Sleep Minutes vs Temperature Deviation")
    ]

    color_map = {
        "dailyRmssd": "royalblue",
        "deepRmssd": "seagreen",
        "restingHeartRate": "firebrick",
        "deep_minutes": "mediumpurple",
        "rem_minutes": "orange",
    }

    # Create groups of 3
    for i in range(0, len(scatter_pairs), 3):
        cols = st.columns(3)
        for col, (y, x, title) in zip(cols, scatter_pairs[i:i+3]):
            with col:
                color = color_map.get(y, "gray")
                fig = plot_simple_scatter(df, x_col=x, y_col=y, title=title, color=color)
                st.plotly_chart(fig, width='content')


with tab4:
    
    metric_config = {
    "dailyRmssd": {"unit": "ms", "goal": "up"},
    "deepRmssd": {"unit": "ms", "goal": "up"},
    "deep_minutes": {"unit": "min", "goal": "up"},
    "sleep_hours": {"unit": "hours", "goal": "up"}}

    metric_titles = {
        "dailyRmssd": "Daily HRV",
        "deepRmssd": "Deep Sleep HRV",
        "deep_minutes": "Deep Sleep Duration",
        "sleep_hours": "Total Sleep Time"
    }

    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period("M")

    # --- Helper Functions ---

    def calculate_percent_change(current, previous):
        if previous == 0 or pd.isna(previous) or pd.isna(current):
            return None
        return round((current - previous) / previous * 100, 1)

    def get_dynamic_axis_range(pct_changes, margin=0.2, min_range=10):
        min_val = pct_changes.min()
        max_val = pct_changes.max()
        range_span = max_val - min_val

        if range_span < min_range:
            center = (max_val + min_val) / 2
            return [center - min_range / 2, center + min_range / 2]

        pad = range_span * margin
        return [min_val - pad, max_val + pad]

    def get_zone_bounds(axis_range, n_zones=5):
        return list(np.linspace(axis_range[0], axis_range[1], n_zones + 1))

    def get_zone_label(value, zone_bounds, zone_labels):
        for i in range(len(zone_bounds) - 1):
            if zone_bounds[i] <= value <= zone_bounds[i + 1]:
                return zone_labels[i]
        return "Out of range"

    def generate_dynamic_insight(metric_name, pct_change, zone_bounds, zone_labels):
        if pct_change is None:
            return f"{metric_name}: No data available."

        label = get_zone_label(pct_change, zone_bounds, zone_labels)

        messages = {
            "Excellent": f"🟢 {metric_name} improved significantly. Excellent recovery trend.",
            "Good": f"🟢 {metric_name} improved. Keep up the healthy habits.",
            "Stable": f"🟡 {metric_name} was stable. No major change.",
            "Drop": f"🔴 {metric_name} decreased slightly. May need attention.",
            "Decline": f"🔴 {metric_name} dropped significantly. Watch for stress or poor recovery.",
        }

        return messages.get(label, f"{metric_name}: Change = {pct_change:+.1f}%")

    def plot_matplotlib_gauge(percent_change, zone_bounds, zone_colors, zone_labels=None,
                            title="Metric Gauge", needle_color="black"):
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111, polar=True)

        theta_bounds = np.interp(zone_bounds, [zone_bounds[0], zone_bounds[-1]], [np.pi, 0])

        # Draw color zones
        for i in range(len(zone_colors)):
            start = theta_bounds[i]
            end = theta_bounds[i+1]
            ax.bar(x=start, width=end - start, height=1, bottom=0,
                color=zone_colors[i], edgecolor="white", linewidth=2, align="edge")

        # Draw needle
        theta = np.interp(percent_change, [zone_bounds[0], zone_bounds[-1]], [np.pi, 0])
        ax.plot([theta, theta], [0, 0.85], color=needle_color, linewidth=4)

        # Add value bubble
        ax.annotate(f"{percent_change:+.1f}%",
                    xy=(0, 0),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha="center", va="center",
                    fontsize=20,
                    bbox=dict(boxstyle="circle", fc="black", ec="white", lw=15),
                    color="white")

        # Optional labels
        if zone_labels and len(zone_labels) == len(zone_colors):
            for i in range(len(zone_labels)):
                mid_val = (zone_bounds[i] + zone_bounds[i+1]) / 2
                mid_theta = np.interp(mid_val, [zone_bounds[0], zone_bounds[-1]], [np.pi, 0])
                ax.text(mid_theta, 1.1, zone_labels[i], ha="center", va="center",
                        fontsize=9, rotation=np.rad2deg(mid_theta - np.pi/2),
                        color="white")
        rcParams['font.family'] = segoe_ui.get_name()
        rcParams['font.sans-serif'] = ['Segoi', 'DejaVu Sans',
                               'Lucida Grande', 'Verdana']
        ax.set_axis_off()
        ax.set_ylim(0, 1.2)
        plt.title(title, fontsize=12, pad=12,fontname='Segoe UI')

        st.pyplot(fig)
        return fig

    # --- Zones Setup ---
    zone_colors = ["darkred", "orange", "gold", "lightgreen", "green"]
    zone_labels = ["Decline", "Drop", "Stable", "Good", "Excellent"]

    # --- Layout ---
    cols = st.columns(4)

    for i, (metric, info) in enumerate(metric_config.items()):
        monthly_median = df.groupby('month')[metric].median()
        pct_changes = monthly_median.pct_change().dropna() * 100

        if len(pct_changes) < 2:
            st.warning(f"Not enough data for {metric}")
            continue

        bounds = get_zone_bounds(get_dynamic_axis_range(pct_changes))
        current_month = monthly_median.index[-1]
        previous_month = monthly_median.index[-2]
        current_val = monthly_median.loc[current_month]
        previous_val = monthly_median.loc[previous_month]
        pct_change = calculate_percent_change(current_val, previous_val)
        current_month_label = current_month.strftime("%b %Y")
        previous_month_label = previous_month.strftime("%b %Y")

        with cols[i % 4]:
            plot_matplotlib_gauge(
                percent_change=pct_change,
                zone_bounds=bounds,
                zone_colors=zone_colors,
                zone_labels=zone_labels,
                title=f"{metric_titles.get(metric, metric.replace('_', ' ').title())}\n{current_month_label} vs {previous_month_label}"
            )

            # 📌 Dynamic insight based on gauge zone
            insight_text = generate_dynamic_insight(
                metric_titles.get(metric, metric),
                pct_change,
                bounds,
                zone_labels
            )

            # Display insight right below the gauge
            st.markdown(insight_text)
