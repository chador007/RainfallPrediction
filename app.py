"""
Streamlit app for Bhutan Rainfall Prediction

This app:
- Loads cleaned rainfall data for Bhutan subnational regions
- Displays comprehensive data exploration and visualizations
- Loads pre-trained models (Linear Regression, Random Forest, Gradient Boosting)
- Predicts monthly rainfall for future years (2025 onwards)
- Shows interactive forecast graphs

Author: [Chador Wangchuk]
Data Source: Bhutan Subnational Rainfall Dataset
"""

import streamlit as st                   
import pandas as pd                      
import numpy as np                        
import joblib                             
import matplotlib.pyplot as plt           
import seaborn as sns                     
from pathlib import Path                 
from datetime import datetime            

# -----------------------------
# 0. Page Configuration (MUST BE FIRST)
# -----------------------------
st.set_page_config(
    page_title="Bhutan Rainfall Prediction",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# 1. Base Paths Setup
# -----------------------------
# __file__ = path of this script (app/streamlit_app.py)
# Go up one level to reach project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Build paths relative to project root
DATA_PATH = "btn-rainfall-subnat-full.csv"
MODEL_LR_PATH = "Lr_model.pkl"
MODEL_RF_PATH = "Random_Forest_Regressor_model.pkl"
MODEL_GBR_PATH = "Gradient_boosting_model.pkl"
SCALER_X_PATH = "x_scaler.pkl"
SCALER_Y_PATH = "y_scaler.pkl"

# -----------------------------
# 2. Custom CSS Styling
# -----------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .prediction-box {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)
# -----------------------------
# 3. Data Loading Functions
# -----------------------------
@st.cache_data
def load_data():
    """
    Load the cleaned Bhutan rainfall dataset and prepare it for analysis.

    Returns:
        pd.DataFrame: Cleaned rainfall dataset with datetime index
    """
    df = pd.read_csv(DATA_PATH)

    # Convert date to datetime and set as index
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()

    # Extract time components
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day

    return df


@st.cache_resource
def load_models():
    """
    Load all trained models for rainfall prediction.

    Returns:
        dict: Dictionary containing all loaded models
    """
    models = {}
    
    try:
        if Path(MODEL_GBR_PATH).exists():
            models['Gradient Boosting'] = joblib.load(MODEL_GBR_PATH)
            print(f"✅ Loaded Gradient Boosting from {MODEL_GBR_PATH}")
        else:
            print(f"❌ File not found: {MODEL_GBR_PATH}")
    except Exception as e:
        print(f"❌ Error loading Gradient Boosting: {e}")
    
    try:
        if Path(MODEL_RF_PATH).exists():
            models['Random Forest'] = joblib.load(MODEL_RF_PATH)
            print(f"✅ Loaded Random Forest from {MODEL_RF_PATH}")
        else:
            print(f"❌ File not found: {MODEL_RF_PATH}")
    except Exception as e:
        print(f"❌ Error loading Random Forest: {e}")
    
    try:
        if Path(MODEL_LR_PATH).exists():
            models['Linear Regression'] = joblib.load(MODEL_LR_PATH)
            print(f"✅ Loaded Linear Regression from {MODEL_LR_PATH}")
        else:
            print(f"❌ File not found: {MODEL_LR_PATH}")
    except Exception as e:
        print(f"❌ Error loading Linear Regression: {e}")
    
    return models  # <-- THIS WAS MISSING!


@st.cache_resource
def load_scalers():
    """
    Load the fitted StandardScaler objects for feature and target scaling.

    Returns:
        tuple: (scaler_X, scaler_y) - Feature and target scalers
    """
    scaler_X = joblib.load(SCALER_X_PATH) 
    scaler_y = joblib.load(SCALER_Y_PATH)
    return scaler_X, scaler_y


@st.cache_data
def compute_monthly_statistics(_df):
    """
    Compute historical monthly averages for all features.
    Used to fill feature values when predicting future dates.

    Args:
        _df: DataFrame with rainfall data
        
    Returns:
        pd.DataFrame: Monthly averages for each feature
    """
    features = ['rfh_avg', 'r1h', 'r1h_avg', 'r3h', 'r3h_avg', 'n_pixels']
    monthly_avg = _df.groupby('month')[features].mean()
    return 


@st.cache_resource
def load_scalers():
    """
    Load the fitted StandardScaler objects for feature and target scaling.
    
    Returns:
        tuple: (scaler_X, scaler_y) - Feature and target scalers
    """
    scaler_X = joblib.load(SCALER_X_PATH) 
    scaler_y = joblib.load(SCALER_Y_PATH)
    return scaler_X, scaler_y

@st.cache_data
def compute_monthly_statistics(_df):
    """
    Compute historical monthly averages for all features.
    Used to fill feature values when predicting future dates.
    
    Args:
        _df: DataFrame with rainfall data
        
    Returns:
        pd.DataFrame: Monthly averages for each feature
    """
    features = ['rfh_avg', 'r1h', 'r1h_avg', 'r3h', 'r3h_avg', 'n_pixels']
    monthly_avg = _df.groupby('month')[features].mean()
    return monthly_avg

# -----------------------------
# 4. Load Resources
# -----------------------------
# Try to load data and models
try:
    df = load_data()
    models = load_models()
    scaler_X, scaler_y = load_scalers()
    monthly_feature_avg = compute_monthly_statistics(df)
    
    # Compute basic statistics
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    data_loaded = True
except Exception as e:
    data_loaded = False
    error_message = str(e)

# -----------------------------
# 5. Sidebar Navigation
# -----------------------------
st.sidebar.markdown("## 🌧️ Navigation")
page = st.sidebar.radio(
    "Select Section:",
    [
        "🏠 Home",
        "📊 Data Overview",
        "📈 Visualizations",
        "🔮 Rainfall Prediction",
        "📉 Model Performance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 About")
st.sidebar.info(
    """
    This dashboard analyzes rainfall patterns 
    in Bhutan and provides forecasts using 
    machine learning models.
    
    **Models Available:**
    - Linear Regression
    - Random Forest
    - Gradient Boosting
    """
)

# -----------------------------
# 6. Home Page
# -----------------------------
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🌧️ Bhutan Rainfall Prediction Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Analyze historical rainfall patterns and forecast future rainfall using ML models</p>',
        unsafe_allow_html=True
    )
    
    if data_loaded:
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📅 Data Range",
                value=f"{min_year} - {max_year}",
                delta=f"{max_year - min_year + 1} years"
            )
        
        with col2:
            st.metric(
                label="📝 Total Records",
                value=f"{len(df):,}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="🌧️ Avg Rainfall",
                value=f"{df['rfh'].mean():.2f} mm",
                delta=None
            )
        
        with col4:
            st.metric(
                label="🗺️ Regions",
                value=f"{df['PCODE'].nunique()}",
                delta=None
            )
        
        st.markdown("---")
        
        # Quick Overview Charts
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("📈 Monthly Rainfall Trend")
            monthly_avg = df['rfh'].resample('M').mean()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(monthly_avg.index, monthly_avg.values, color='#1E88E5', linewidth=1.5)
            ax.fill_between(monthly_avg.index, monthly_avg.values, alpha=0.3, color='#1E88E5')
            ax.set_xlabel("Date")
            ax.set_ylabel("Rainfall (mm)")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col_right:
            st.subheader("📊 Seasonal Pattern")
            monthly_pattern = df.groupby('month')['rfh'].mean()
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = plt.cm.Blues(np.linspace(0.3, 0.9, 12))
            bars = ax.bar(monthly_pattern.index, monthly_pattern.values, color=colors)
            ax.set_xlabel("Month")
            ax.set_ylabel("Avg Rainfall (mm)")
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("### 🚀 Quick Start Guide")
        st.markdown("""
        1. **📊 Data Overview** - Explore the dataset structure and statistics
        2. **📈 Visualizations** - View detailed rainfall patterns and distributions
        3. **🔮 Rainfall Prediction** - Forecast rainfall for 2025 and beyond
        4. **📉 Model Performance** - Compare different ML models
        """)
    else:
        st.error(f"⚠️ Error loading data: {error_message}")
        st.info("Please ensure the data and model files are in the correct locations.")

# -----------------------------
# 7. Data Overview Page
# -----------------------------
elif page == "📊 Data Overview":
    st.header("📊 Data Overview")
    
    if data_loaded:
        # Dataset Info
        st.subheader("Dataset Information")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
        with col2:
            st.info(f"**Date Range:** {df.index.min().date()} to {df.index.max().date()}")
        
        # Data Preview
        st.subheader("Data Preview (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Column Information
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns.tolist(),
            'Data Type': [str(df[col].dtype) for col in df.columns],
            'Non-Null Count': [df[col].notna().sum() for col in df.columns],
            'Null Count': [df[col].isna().sum() for col in df.columns],
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)
        
        # Statistical Summary
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Missing Values
        st.subheader("Missing Values Analysis")
        missing = df.isna().sum()
        if missing.sum() == 0:
            st.success("✅ No missing values in the dataset!")
        else:
            st.dataframe(missing[missing > 0], use_container_width=True)
    else:
        st.error("Data not loaded. Please check file paths.")

# -----------------------------
# 8. Visualizations Page
# -----------------------------
elif page == "📈 Visualizations":
    st.header("📈 Data Visualizations")
    
    if data_loaded:
        # Visualization selector
        viz_type = st.selectbox(
            "Select Visualization Type:",
            [

                "Monthly Rainfall Distribution (Boxplot)",
                "Rainfall Distribution Histogram",
                "Average Daily Rainfall Over Time",
                "Rainfall by Administrative Level",
                "Feature Correlation Heatmap",
            ]
        )
        
        st.markdown("---")
        
        # Generate selected visualization
        if viz_type == "Average Daily Rainfall Over Time":
            st.subheader("Average Daily Rainfall (rfh) Over Time")
            daily_rain = df.groupby(df.index.date)['rfh'].mean()
            
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(daily_rain.index, daily_rain.values, linewidth=0.7, color='#1E88E5')
            ax.set_title("Average Daily Rainfall (rfh) Over Time", fontsize=14)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Rainfall (mm)", fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info(f"📊 Average daily rainfall: {daily_rain.mean():.2f} mm | Max: {daily_rain.max():.2f} mm")
    
        
        elif viz_type == "Monthly Rainfall Distribution (Boxplot)":
            st.subheader("Monthly Rainfall Distribution")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(x='month', y='rfh', data=df, ax=ax, palette='Blues')
            ax.set_title("Monthly Rainfall Distribution", fontsize=14)
            ax.set_xlabel("Month", fontsize=12)
            ax.set_ylabel("Rainfall (mm)", fontsize=12)
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info("📊 Monsoon season (June-September) shows highest rainfall variability")
        
        elif viz_type == "Feature Correlation Heatmap":
            st.subheader("Feature Correlation Heatmap")
            
            features = ['rfh', 'rfh_avg', 'r1h', 'r1h_avg', 'r3h', 'r3h_avg', 'n_pixels']
            corr = df[features].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax,
                       square=True, linewidths=0.5)
            ax.set_title("Feature Correlation Heatmap", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info("📊 Strong correlations exist between rfh and its averaged variants")
        
        elif viz_type == "Rainfall by Administrative Level":
            st.subheader("Rainfall Distribution by Administrative Level")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='adm_level', y='rfh', data=df, ax=ax, palette='Set2')
            ax.set_title("Rainfall Distribution by Administrative Level", fontsize=14)
            ax.set_xlabel("Admin Level", fontsize=12)
            ax.set_ylabel("Rainfall (mm)", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
        
        elif viz_type == "Rainfall Distribution Histogram":
            st.subheader("Distribution of Hourly Rainfall (rfh)")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['rfh'], bins=50, kde=True, ax=ax, color='#1E88E5')
            ax.set_title("Distribution of Hourly Rainfall (rfh)", fontsize=14)
            ax.set_xlabel("Rainfall (mm)", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{df['rfh'].mean():.2f} mm")
            with col2:
                st.metric("Median", f"{df['rfh'].median():.2f} mm")
            with col3:
                st.metric("Std Dev", f"{df['rfh'].std():.2f} mm")
            with col4:
                st.metric("Max", f"{df['rfh'].max():.2f} mm")
        
    else:
        st.error("Data not loaded. Please check file paths.")

# -----------------------------
# 9. Rainfall Prediction Page
# -----------------------------
elif page == "🔮 Rainfall Prediction":
    st.header("🔮 Rainfall Prediction")

    if data_loaded and models:
        st.markdown("""
        Enter the feature values below to predict rainfall. You can use historical 
        averages as reference or input your own values based on weather forecasts 
        or climate scenarios.
        """)
        
        st.markdown("---")
        
        # Model Selection
        selected_model_name = st.selectbox(
            "🤖 Select Prediction Model:",
            list(models.keys()),
            index=0,
            key="pred_model"
        )
        selected_model = models[selected_model_name]
        
        st.markdown("---")
        
        # Input columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📅 Date Features**")
            
            input_year = st.number_input(
                "Year",
                min_value=1900,
                max_value=2100,
                value=2025,
                step=1,
                help="Year for prediction"
            )
            
            input_month = st.selectbox(
                "Month",
                options=list(range(1, 13)),
                format_func=lambda x: ['January', 'February', 'March', 'April', 
                                        'May', 'June', 'July', 'August', 
                                        'September', 'October', 'November', 'December'][x-1],
                index=0,
                help="Month of the year (1-12)"
            )
            
            input_day = st.number_input(
                "Day",
                min_value=1,
                max_value=31,
                value=15,
                step=1,
                help="Day of the month"
            )
        
        with col2:
            st.markdown("**🌧️ Rainfall Features**")
            
            # Get default values from monthly averages
            default_month_avg = monthly_feature_avg.loc[input_month] if input_month in monthly_feature_avg.index else monthly_feature_avg.mean()
            
            input_rfh_avg = st.number_input(
                "rfh_avg (Rainfall Average)",
                min_value=0.0,
                max_value=100.0,
                value=float(default_month_avg['rfh_avg']),
                step=0.1,
                format="%.3f",
                help="Average rainfall indicator"
            )
            
            input_r1h = st.number_input(
                "r1h (1-hour Rainfall)",
                min_value=0.0,
                max_value=100.0,
                value=float(default_month_avg['r1h']),
                step=0.1,
                format="%.3f",
                help="1-hour rainfall measurement"
            )
            
            input_r1h_avg = st.number_input(
                "r1h_avg (1-hour Rainfall Average)",
                min_value=0.0,
                max_value=100.0,
                value=float(default_month_avg['r1h_avg']),
                step=0.1,
                format="%.3f",
                help="Average of 1-hour rainfall"
            )
        
        with col3:
            st.markdown("**📡 Additional Features**")
            
            input_r3h = st.number_input(
                "r3h (3-hour Rainfall)",
                min_value=0.0,
                max_value=100.0,
                value=float(default_month_avg['r3h']),
                step=0.1,
                format="%.3f",
                help="3-hour rainfall measurement"
            )
            
            input_r3h_avg = st.number_input(
                "r3h_avg (3-hour Rainfall Average)",
                min_value=0.0,
                max_value=100.0,
                value=float(default_month_avg['r3h_avg']),
                step=0.1,
                format="%.3f",
                help="Average of 3-hour rainfall"
            )
        
        st.markdown("---")
        
        # Predict button
        if st.button("🔮 Predict Rainfall", type="primary", key="predict_single"):
            
            # Create feature DataFrame
            X_input = pd.DataFrame({
                'year': [float(input_year)],
                'month': [float(input_month)],
                'day': [float(input_day)],
                'rfh_avg': [float(input_rfh_avg)],
                'r1h': [float(input_r1h)],
                'r1h_avg': [float(input_r1h_avg)],
                'r3h': [float(input_r3h)],
                'r3h_avg': [float(input_r3h_avg)]
    
            })
            
            # Scale features
            if scaler_X is not None:
                X_scaled = scaler_X.transform(X_input)
            else:
                X_scaled = X_input.values
            
            # Make prediction
            pred_scaled = selected_model.predict(X_scaled)[0]
            
            # Inverse transform
            if scaler_y is not None:
                prediction = scaler_y.inverse_transform([[pred_scaled]])[0][0]
            else:
                prediction = pred_scaled
            
            # Ensure non-negative
            prediction = max(prediction, 0)
            
            # Display prediction
            st.markdown("### 🎯 Prediction Result")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown(
                    f"""
                    <div style="background-color: #e3f2fd; border-radius: 15px; 
                                padding: 30px; text-align: center; margin: 20px 0;">
                        <h2 style="color: #1565c0; margin-bottom: 10px;">
                            Predicted Rainfall
                        </h2>
                        <h1 style="color: #0d47a1; font-size: 3.5rem; margin: 0;">
                            {prediction:.2f} mm
                        </h1>
                        <p style="color: #666; margin-top: 15px;">
                            {['January', 'February', 'March', 'April', 'May', 'June',
                                'July', 'August', 'September', 'October', 'November', 
                                'December'][input_month-1]} {input_day}, {input_year}
                        </p>
                        <p style="color: #888; font-size: 0.9rem;">
                            Model: {selected_model_name}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
    else:
        if not data_loaded:
            st.error("⚠️ Data not loaded. Please check file paths.")
        if not models:
            st.error("⚠️ No models loaded. Please ensure model files exist.")
# -----------------------------
# 10. Model Performance Page
# -----------------------------
elif page == "📉 Model Performance":
    st.header("📉 Model Performance Comparison")
    
    if data_loaded and models:
        st.markdown("""
        This section compares the performance of different models trained on the 
        Bhutan rainfall dataset. The metrics shown are from the training/validation phase.
        """)
        
        # Model performance metrics (you would replace these with actual values)
        st.subheader("Model Evaluation Metrics")
        
        # Create metrics table
        metrics_data = {
            'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting (Tuned)'],
            'MAE': [0.205, 0.206, 0.203],  
            'RMSE': [0.415, 0.438, 0.411],
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        with col2:
            # Bar chart comparison
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(metrics_data['Model']))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, metrics_data['MAE'], width, label='MAE', color='#1E88E5')
            bars2 = ax.bar(x + width/2, metrics_data['RMSE'], width, label='RMSE', color='#FF7043')
            
            ax.set_ylabel('Error')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(['LR', 'RF', 'GBR'])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
        
        
    else:
        st.error("⚠️ Data or models not loaded. Please check file paths.")

# -----------------------------
# 11. Footer
# -----------------------------
st.sidebar.markdown("---")