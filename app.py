import streamlit as st          # Web app framework for creating interactive data applications
import pandas as pd             # Data manipulation and analysis library for tabular data structures
import numpy as np              # Scientific computing library for numerical operations and array handling
import matplotlib.pyplot as plt  # Plotting library for creating static, interactive, and animated visualizations
import seaborn as sns           # Statistical data visualization library based on matplotlib
import plotly.express as px     # High-level interface for interactive plotly visualizations
import plotly.graph_objects as go  # Low-level interface for creating custom interactive plots
from scipy import stats         # Statistical functions and tests from SciPy
from data_loader import load_spotify_data, get_numerical_features, get_categorical_features  # Custom data loading functions
from data_analysis import (generate_descriptive_stats, create_correlation_heatmap,
                         create_distribution_plot, build_regression_model,
                         create_scatter_plot)  # Custom data analysis functions
from sklearn.linear_model import LinearRegression  # Linear regression models from scikit-learn
from sklearn.model_selection import train_test_split  # Utilities for splitting datasets for training/testing
from sklearn.metrics import mean_squared_error, r2_score  # Evaluation metrics for regression models
from sklearn.preprocessing import StandardScaler  # Feature scaling for preprocessing data

# Page configuration
st.set_page_config(
    page_title="Spotify Track Analysis",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1DB954;  /* Spotify green */
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333333;
    }
    .card {
        border-radius: 5px;
        background-color: #F8F9FA;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the data
@st.cache_data
def get_data():
    return load_spotify_data()

# App header
st.markdown("<h1 class='main-header'>Spotify Track Analysis App</h1>", unsafe_allow_html=True)
st.markdown("This app performs statistical analysis on Spotify track data.")

# Load data
try:
    df = get_data()
    st.success(f"Successfully loaded dataset with {df.shape[0]} tracks and {df.shape[1]} features")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Home", "Data Overview", "Descriptive Statistics", "Feature Distributions", 
     "Correlation Analysis", "Regression Modeling", "About"]
)

# Get numerical and categorical features
numerical_features = get_numerical_features(df)
categorical_features = get_categorical_features(df)

# Home Page (directly in app.py instead of redirecting)
if page == "Home":
    st.markdown("<h2 class='sub-header'>Welcome to Spotify Track Analysis</h2>", unsafe_allow_html=True)
    
    # Introduction section
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: #1DB954;">Discover Insights in Music Data</h3>
        <p style=" color: #000000">
            Welcome to the Spotify Track Analysis App! This tool allows you to explore and analyze 
            audio features of Spotify tracks using statistical methods and visualizations.
        </p>
        <p style=" color: #000000">
            Use the navigation menu on the left to explore different analysis sections.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
            <h2 style="color: #1DB954;">{}</h2>
            <p style=" color: #000000">Tracks Analyzed</p>
        </div>
        """.format(df.shape[0]), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
            <h2 style="color: #1DB954;">{}</h2>
            <p style=" color: #000000">Audio Features</p>
        </div>
        """.format(len(numerical_features)), unsafe_allow_html=True)
    
    with col3:
        avg_popularity = round(df['popularity'].mean(), 1) if 'popularity' in df.columns else "N/A"
        st.markdown("""
        <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
            <h2 style="color: #1DB954;">{}</h2>
            <p style=" color: #000000">Average Popularity</p>
        </div>
        """.format(avg_popularity), unsafe_allow_html=True)
    
    # Quick visualizations section
    st.subheader("Quick Insights")
    
    # Create two columns for visualizations
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Display a feature distribution if data available
        if numerical_features and len(numerical_features) > 0:
            default_feature = 'popularity' if 'popularity' in numerical_features else numerical_features[0]
            fig = px.histogram(
                df, 
                x=default_feature,
                title=f'Distribution of {default_feature}',
                color_discrete_sequence=['#1DB954']
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_col2:
        # Display correlation between two popular features if available
        if len(numerical_features) >= 2:
            feature_x = 'danceability' if 'danceability' in numerical_features else numerical_features[0]
            feature_y = 'energy' if 'energy' in numerical_features else numerical_features[1]
            
            fig = px.scatter(
                df.sample(min(1000, len(df))), 
                x=feature_x, 
                y=feature_y,
                title=f'{feature_y} vs. {feature_x}',
                color_discrete_sequence=['#1DB954']
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Feature cards section - what you can do with the app
    st.subheader("Explore The App")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #1DB954;">
            <h4 style="color: #1DB954;">📊 Data Overview</h4>
            <p style=" color: #000000">Explore the Spotify dataset structure, feature descriptions, and basic statistics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #1DB954;">
            <h4 style="color: #1DB954;">📉 Feature Distributions</h4>
            <p style=" color: #000000">Analyze probability distributions of audio features with histograms and density plots.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #1DB954;">
            <h4 style="color: #1DB954;">📊 Regression Modeling</h4>
            <p style=" color: #000000">Build predictive models to understand what audio features influence track popularity.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #1DB954;">
            <h4 style="color: #1DB954;">📈 Descriptive Statistics</h4>
            <p style=" color: #000000">Dive into detailed statistical measures for each feature with confidence intervals.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #1DB954;">
            <h4 style="color: #1DB954;">🔄 Correlation Analysis</h4>
            <p style=" color: #000000">Discover relationships between audio features through correlation heatmaps and scatter plots.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #1DB954;">
            <h4 style="color: #1DB954;">ℹ️ About</h4>
            <p style=" color: #000000">Learn more about the project, data sources, and statistical methods used.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Get started CTA
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; margin-bottom: 30px;">
        <p style="font-size: 1.2rem; margin-bottom: 20px;">Ready to explore the data?</p>
        <p>Select a page from the navigation menu on the left to get started!</p>
    </div>
    """, unsafe_allow_html=True)

# Data Overview Page
elif page == "Data Overview":
    st.markdown("<h2 class='sub-header'>Data Overview</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Dataset Information")
        st.write(f"Number of tracks: {df.shape[0]}")
        st.write(f"Number of features: {df.shape[1]}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Data Types")
        st.write(f"Numerical features: {len(numerical_features)}")
        st.write(f"Categorical features: {len(categorical_features)}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display sample data
    st.subheader("Sample Data")
    st.dataframe(df.head(10))
    
    # Display feature descriptions
    st.subheader("Feature Descriptions")
    
    feature_descriptions = {
        'acousticness': 'A measure of whether the track is acoustic (1.0) or not (0.0).',
        'danceability': 'How suitable the track is for dancing (0.0 to 1.0).',
        'energy': 'Intensity and activity measure (0.0 to 1.0).',
        'instrumentalness': 'Predicts whether a track contains vocals (0.0) or is instrumental (1.0).',
        'key': 'The key the track is in (integers map to pitches using standard Pitch Class notation).',
        'liveness': 'Detects the presence of an audience (higher values mean higher probability of live recording).',
        'loudness': 'Overall loudness in decibels (dB), typical range between -60 and 0 db.',
        'mode': 'Modality of the track (1 = major, 0 = minor).',
        'popularity': 'Popularity of the track (0-100), calculated by Spotify algorithm.',
        'speechiness': 'Presence of spoken words (higher values indicate more speech-like recordings).',
        'tempo': 'Tempo in beats per minute (BPM).',
        'time_signature': 'Estimated time signature (3, 4, etc.).',
        'valence': 'Musical positiveness (0.0 = sad, 1.0 = happy).'
    }
    
    desc_df = pd.DataFrame({
        'Feature': feature_descriptions.keys(),
        'Description': feature_descriptions.values()
    })
    
    st.table(desc_df)
    
    # Missing value analysis
    st.subheader("Missing Value Analysis")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage (%)': missing_percent
    }).sort_values('Missing Values', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.bar(
            missing_df[missing_df['Missing Values'] > 0], 
            x=missing_df[missing_df['Missing Values'] > 0].index,
            y='Percentage (%)', 
            title='Missing Values by Feature (%)'
        )
        fig.update_layout(xaxis_title='Features', yaxis_title='Missing Values (%)')
        st.plotly_chart(fig)
    
    with col2:
        st.dataframe(missing_df[missing_df['Missing Values'] > 0])
    
    # Add pie chart for categorical data
    if categorical_features:
        st.subheader("Categorical Data Overview")
        
        # Choose a categorical feature for the pie chart
        selected_categorical = st.selectbox(
            "Select a categorical feature for visualization:",
            options=categorical_features,
            index=0 if 'mode' in categorical_features else 0
        )
        
        if selected_categorical:
            pie_col1, pie_col2 = st.columns([3, 2])
            
            with pie_col1:
                # Create a pie chart for the selected categorical feature
                cat_counts = df[selected_categorical].value_counts().reset_index()
                cat_counts.columns = [selected_categorical, 'Count']
                
                # Limit to top 10 categories if there are more than 10
                if len(cat_counts) > 10:
                    cat_counts = cat_counts.iloc[:10].copy()
                    cat_counts.loc[len(cat_counts)] = [f'Other ({len(cat_counts)-10} categories)', 
                                                      df[selected_categorical].value_counts().iloc[10:].sum()]
                
                fig = px.pie(
                    cat_counts, 
                    values='Count', 
                    names=selected_categorical,
                    title=f'Distribution of {selected_categorical}',
                    hole=0.4  # Creates a donut chart effect
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with pie_col2:
                st.dataframe(cat_counts)
    
    # Add a component bar chart showing the distribution of numerical features
    st.subheader("Numerical Features Overview")
    
    # Select top numerical features to compare 
    top_numerical_features = st.multiselect(
        "Select numerical features to compare:",
        options=numerical_features,
        default=numerical_features[:5] if len(numerical_features) >= 5 else numerical_features
    )
    
    if top_numerical_features:
        # Calculate statistics for each feature
        feature_stats = pd.DataFrame({
            'Mean': df[top_numerical_features].mean(),
            'Median': df[top_numerical_features].median(),
            'StdDev': df[top_numerical_features].std()
        }).reset_index()
        feature_stats.columns = ['Feature', 'Mean', 'Median', 'StdDev']
        
        # Create normalized version for side-by-side comparison
        feature_stats_normalized = feature_stats.copy()
        for col in ['Mean', 'Median', 'StdDev']:
            max_val = feature_stats[col].max()
            if max_val > 0:
                feature_stats_normalized[col] = feature_stats[col] / max_val
        
        # Bar chart tabs
        bar_type = st.radio(
            "Select Bar Chart Type:",
            options=["Grouped", "Stacked", "Normalized"],
            horizontal=True
        )
        
        if bar_type == "Grouped":
            fig = px.bar(
                feature_stats, 
                x='Feature', 
                y=['Mean', 'Median', 'StdDev'],
                title='Comparison of Numerical Features',
                barmode='group',
                color_discrete_sequence=['#1DB954', '#4169E1', '#FF6B6B']
            )
        elif bar_type == "Stacked":
            fig = px.bar(
                feature_stats, 
                x='Feature', 
                y=['Mean', 'Median', 'StdDev'],
                title='Comparison of Numerical Features',
                barmode='stack',
                color_discrete_sequence=['#1DB954', '#4169E1', '#FF6B6B']
            )
        else:  # Normalized
            fig = px.bar(
                feature_stats_normalized, 
                x='Feature', 
                y=['Mean', 'Median', 'StdDev'],
                title='Normalized Comparison of Numerical Features',
                barmode='group',
                color_discrete_sequence=['#1DB954', '#4169E1', '#FF6B6B']
            )
            fig.update_layout(yaxis_title='Normalized Value')
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# Descriptive Statistics Page
elif page == "Descriptive Statistics":
    st.markdown("<h2 class='sub-header'>Descriptive Statistics</h2>", unsafe_allow_html=True)
    
    # Feature selection
    selected_features = st.multiselect(
        "Select features to analyze:",
        options=numerical_features,
        default=numerical_features[:5] if len(numerical_features) >= 5 else numerical_features
    )
    
    if selected_features:
        # Generate descriptive statistics
        stats_df = generate_descriptive_stats(df, selected_features)
        
        # Display statistics table
        st.subheader("Statistical Measures")
        st.dataframe(stats_df.style.format("{:.3f}"))
        
        # Visualize means with confidence intervals
        st.subheader("Means with 95% Confidence Intervals")
        
        fig = go.Figure()
        
        for feature in selected_features:
            mean = stats_df.loc[feature, 'mean']
            ci_low = stats_df.loc[feature, '95%_CI_low']
            ci_high = stats_df.loc[feature, '95%_CI_high']
            
            fig.add_trace(go.Scatter(
                x=[feature, feature],
                y=[ci_low, ci_high],
                mode='lines',
                line=dict(width=2, color='rgba(70, 130, 180, 0.6)'),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=[feature],
                y=[mean],
                mode='markers',
                marker=dict(size=10, color='rgba(70, 130, 180, 1)'),
                showlegend=False
            ))
        
        fig.update_layout(
            title='Means with 95% Confidence Intervals',
            xaxis_title='Features',
            yaxis_title='Value',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Box plots for selected features
        st.subheader("Box Plots")
        
        fig = go.Figure()
        
        for feature in selected_features:
            fig.add_trace(go.Box(
                y=df[feature].dropna(),
                name=feature,
                boxpoints='outliers',
                jitter=0.3,
                pointpos=-1.5,
                marker_color='rgb(107,174,214)'
            ))
        
        fig.update_layout(
            title='Distribution of Features',
            yaxis_title='Value',
            boxmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least one feature to analyze.")

# Feature Distributions Page
elif page == "Feature Distributions":
    st.markdown("<h2 class='sub-header'>Feature Distributions</h2>", unsafe_allow_html=True)
    
    # Feature selection
    selected_feature = st.selectbox(
        "Select a feature to analyze its distribution:",
        options=numerical_features,
        index=0 if numerical_features else None
    )
    
    if selected_feature:
        # Create tabs for different visualizations
        dist_tab, prob_tab, qq_tab, hist_tab, bar_tab = st.tabs([
            "Distribution", "Probability Density", "Q-Q Plot", "Histograms", "Bar Charts"
        ])
        
        with dist_tab:
            st.subheader(f"Distribution of {selected_feature}")
            
            # Create distribution plot
            fig = create_distribution_plot(df, selected_feature)
            st.pyplot(fig)
            
            # Kolmogorov-Smirnov test for normality
            ks_stat, p_value = stats.kstest(
                df[selected_feature].dropna(), 
                'norm', 
                args=(df[selected_feature].mean(), df[selected_feature].std())
            )
            
            st.write("### Normality Test")
            st.write(f"Kolmogorov-Smirnov test p-value: {p_value:.6f}")
            
            if p_value < 0.05:
                st.write("The distribution is significantly different from normal (p < 0.05)")
            else:
                st.write("The distribution is not significantly different from normal (p >= 0.05)")
        
        with prob_tab:
            st.subheader(f"Probability Density Function of {selected_feature}")
            
            # Create a PDF plot using KDE
            fig, ax = plt.subplots(figsize=(10, 6))
            data = df[selected_feature].dropna()
            
            # KDE plot
            sns.kdeplot(data, ax=ax, fill=True, color='skyblue', alpha=0.7, label='Empirical PDF')
            
            # Add a normal distribution line for comparison
            x = np.linspace(data.min(), data.max(), 100)
            mean = data.mean()
            std = data.std()
            y = stats.norm.pdf(x, mean, std)
            ax.plot(x, y, 'r-', label='Normal PDF')
            
            # Add labels and title
            ax.set_xlabel(selected_feature)
            ax.set_ylabel('Density')
            ax.set_title(f'Probability Density Function of {selected_feature}')
            ax.legend()
            
            st.pyplot(fig)
            
            # Add CDF plot
            st.subheader(f"Cumulative Distribution Function of {selected_feature}")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Empirical CDF
            sorted_data = np.sort(data)
            y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.step(sorted_data, y, where='post', label='Empirical CDF')
            
            # Normal CDF
            x = np.linspace(data.min(), data.max(), 100)
            y = stats.norm.cdf(x, mean, std)
            ax.plot(x, y, 'r-', label='Normal CDF')
            
            # Add labels and title
            ax.set_xlabel(selected_feature)
            ax.set_ylabel('Cumulative Probability')
            ax.set_title(f'Cumulative Distribution Function of {selected_feature}')
            ax.legend()
            
            st.pyplot(fig)
        
        with qq_tab:
            st.subheader(f"Q-Q Plot for {selected_feature}")
            
            # Create Q-Q plot
            fig, ax = plt.subplots(figsize=(10, 6))
            data = df[selected_feature].dropna()
            
            # Q-Q plot
            stats.probplot(data, dist="norm", plot=ax)
            ax.set_title(f'Q-Q Plot for {selected_feature}')
            
            st.pyplot(fig)
        
        with hist_tab:
            st.subheader(f"Histogram Analysis for {selected_feature}")
            
            # Allow user to configure histogram parameters
            hist_col1, hist_col2 = st.columns(2)
            
            with hist_col1:
                bin_method = st.radio(
                    "Bin Selection Method:",
                    options=["Automatic", "Manual"],
                    index=0
                )
                
                if bin_method == "Automatic":
                    bins = st.select_slider(
                        "Number of bins:",
                        options=[5, 10, 15, 20, 25, 30, 50, 100],
                        value=20
                    )
                else:
                    bin_min = float(df[selected_feature].min())
                    bin_max = float(df[selected_feature].max())
                    bin_step = (bin_max - bin_min) / 20
                    
                    bin_edges = st.slider(
                        "Bin range:",
                        min_value=bin_min,
                        max_value=bin_max,
                        value=(bin_min, bin_max),
                        step=bin_step
                    )
                    
                    num_bins = st.number_input("Number of bins:", min_value=5, max_value=100, value=20)
                    bins = np.linspace(bin_edges[0], bin_edges[1], num_bins + 1)
            
            with hist_col2:
                hist_color = st.color_picker("Histogram color:", "#1DB954")
                edge_color = st.color_picker("Edge color:", "#333333")
                show_kde = st.checkbox("Show KDE", value=True)
                show_rug = st.checkbox("Show rug plot", value=True)
            
            # Create the histogram
            fig, ax = plt.subplots(figsize=(12, 6))
            data = df[selected_feature].dropna()
            
            sns.histplot(
                data, 
                bins=bins, 
                kde=show_kde, 
                color=hist_color,
                edgecolor=edge_color,
                alpha=0.7,
                ax=ax
            )
            
            if show_rug:
                sns.rugplot(data, color=edge_color, alpha=0.5, ax=ax)
            
            ax.set_xlabel(selected_feature)
            ax.set_ylabel('Count')
            ax.set_title(f'Histogram of {selected_feature}')
            
            # Add mean and median lines
            mean_val = data.mean()
            median_val = data.median()
            ymin, ymax = ax.get_ylim()
            
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                      label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='blue', linestyle='-.', alpha=0.8, 
                      label=f'Median: {median_val:.2f}')
            ax.legend()
            
            st.pyplot(fig)
            
            # Add interactive plotly histogram as an alternative
            st.subheader("Interactive Histogram")
            
            fig = px.histogram(
                df, 
                x=selected_feature,
                nbins=bins if isinstance(bins, int) else len(bins) - 1,
                color_discrete_sequence=[hist_color],
                marginal="box",  # can be 'rug', 'box', 'violin'
                title=f'Interactive Histogram of {selected_feature}'
            )
            
            fig.update_layout(
                xaxis_title=selected_feature,
                yaxis_title='Count',
                bargap=0.1,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution statistics
            st.subheader("Distribution Statistics")
            
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Mode', 'Std Dev', 'Variance', 'Skewness', 'Kurtosis'],
                'Value': [
                    data.mean(),
                    data.median(),
                    data.mode().iloc[0] if not data.mode().empty else np.nan,
                    data.std(),
                    data.var(),
                    data.skew(),
                    data.kurtosis()
                ]
            })
            
            st.dataframe(stats_df)
        
        with bar_tab:
            st.subheader(f"Bar Chart Analysis for {selected_feature}")
            
            # Options for creating bar charts
            bar_type = st.radio(
                "Bar Chart Type:",
                options=["Value Ranges", "Comparison with Other Features", "Before/After Threshold"],
                index=0
            )
            
            if bar_type == "Value Ranges":
                # Create a bar chart showing counts in different value ranges
                num_ranges = st.slider("Number of ranges:", min_value=3, max_value=10, value=5)
                
                data = df[selected_feature].dropna()
                
                # Create range bins
                min_val = data.min()
                max_val = data.max()
                bin_edges = np.linspace(min_val, max_val, num_ranges + 1)
                
                # Create labels for the ranges
                range_labels = [f"{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}" 
                                for i in range(len(bin_edges) - 1)]
                
                # Count values in each range
                counts = []
                for i in range(len(bin_edges) - 1):
                    if i == len(bin_edges) - 2:  # Last bin includes the upper edge
                        count = ((data >= bin_edges[i]) & (data <= bin_edges[i+1])).sum()
                    else:
                        count = ((data >= bin_edges[i]) & (data < bin_edges[i+1])).sum()
                    counts.append(count)
                
                # Create a dataframe for the ranges and counts
                range_df = pd.DataFrame({
                    'Range': range_labels,
                    'Count': counts
                })
                
                # Create a bar chart
                fig = px.bar(
                    range_df,
                    x='Range',
                    y='Count',
                    title=f'Value Ranges in {selected_feature}',
                    color='Count',
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    xaxis_title='Value Range',
                    yaxis_title='Count',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif bar_type == "Comparison with Other Features":
                # Allow comparing statistics across multiple features
                compare_features = st.multiselect(
                    "Select features to compare:",
                    options=numerical_features,
                    default=[selected_feature] + [f for f in numerical_features[:3] if f != selected_feature]
                )
                
                if len(compare_features) > 0:
                    stat_type = st.selectbox(
                        "Select statistic to compare:",
                        options=["Mean", "Median", "Standard Deviation", "Min", "Max", "Range"],
                        index=0
                    )
                    
                    # Calculate the selected statistic for each feature
                    stat_values = []
                    
                    for feature in compare_features:
                        data = df[feature].dropna()
                        
                        if stat_type == "Mean":
                            value = data.mean()
                        elif stat_type == "Median":
                            value = data.median()
                        elif stat_type == "Standard Deviation":
                            value = data.std()
                        elif stat_type == "Min":
                            value = data.min()
                        elif stat_type == "Max":
                            value = data.max()
                        elif stat_type == "Range":
                            value = data.max() - data.min()
                        
                        stat_values.append(value)
                    
                    # Create a dataframe for the features and statistic values
                    stat_df = pd.DataFrame({
                        'Feature': compare_features,
                        'Value': stat_values
                    })
                    
                    # Create a bar chart
                    fig = px.bar(
                        stat_df,
                        x='Feature',
                        y='Value',
                        title=f'Comparison of {stat_type} Across Features',
                        color='Value',
                        color_continuous_scale='Viridis'
                    )
                    
                    fig.update_layout(
                        xaxis_title='Feature',
                        yaxis_title=stat_type,
                        height=500
                    )
                    
                    # Add a horizontal line for the original feature's value
                    if selected_feature in compare_features:
                        selected_value = stat_df.loc[stat_df['Feature'] == selected_feature, 'Value'].iloc[0]
                        fig.add_hline(
                            y=selected_value,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"{selected_feature} {stat_type}: {selected_value:.2f}",
                            annotation_position="top right"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select at least one feature for comparison.")
                
            else:  # Before/After Threshold
                # Create a bar chart showing values above and below a threshold
                threshold = st.slider(
                    "Threshold value:",
                    min_value=float(df[selected_feature].min()),
                    max_value=float(df[selected_feature].max()),
                    value=float(df[selected_feature].median())
                )
                
                data = df[selected_feature].dropna()
                
                # Count values above and below threshold
                below_count = (data < threshold).sum()
                above_count = (data >= threshold).sum()
                
                # Create a dataframe for the threshold comparison
                threshold_df = pd.DataFrame({
                    'Category': [f'Below {threshold:.2f}', f'Above {threshold:.2f}'],
                    'Count': [below_count, above_count]
                })
                
                # Create a bar chart
                fig = px.bar(
                    threshold_df,
                    x='Category',
                    y='Count',
                    title=f'Values Above and Below Threshold in {selected_feature}',
                    color='Category',
                    color_discrete_map={
                        f'Below {threshold:.2f}': '#FF6B6B',
                        f'Above {threshold:.2f}': '#1DB954'
                    }
                )
                
                fig.update_layout(
                    xaxis_title='Category',
                    yaxis_title='Count',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add a pie chart for the same data
                fig = px.pie(
                    threshold_df,
                    values='Count',
                    names='Category',
                    title=f'Proportion of Values Above and Below Threshold in {selected_feature}',
                    color='Category',
                    color_discrete_map={
                        f'Below {threshold:.2f}': '#FF6B6B',
                        f'Above {threshold:.2f}': '#1DB954'
                    }
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=500)
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select a feature to analyze its distribution.")

# Correlation Analysis Page
elif page == "Correlation Analysis":
    st.markdown("<h2 class='sub-header'>Correlation Analysis</h2>", unsafe_allow_html=True)
    
    # Feature selection for correlation
    st.subheader("Correlation Heatmap")
    
    selected_features = st.multiselect(
        "Select features for correlation analysis:",
        options=numerical_features,
        default=numerical_features[:7] if len(numerical_features) >= 7 else numerical_features
    )
    
    if selected_features and len(selected_features) > 1:
        # Create correlation heatmap
        fig = create_correlation_heatmap(df, selected_features)
        st.pyplot(fig)
        
        # Add correlation table
        st.subheader("Correlation Matrix")
        corr_matrix = df[selected_features].corr().round(3)
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))
    else:
        st.info("Please select at least two features for correlation analysis.")
    
    # Scatter plot for two features
    st.subheader("Relationship Between Two Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_feature = st.selectbox(
            "Select X-axis feature:",
            options=numerical_features,
            index=0 if numerical_features else None
        )
    
    with col2:
        if x_feature:
            remaining_features = [f for f in numerical_features if f != x_feature]
            y_feature = st.selectbox(
                "Select Y-axis feature:",
                options=remaining_features,
                index=0 if remaining_features else None
            )
        else:
            y_feature = None
    
    if x_feature and y_feature:
        # Create scatter plot
        fig = create_scatter_plot(df, x_feature, y_feature)
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate Pearson and Spearman correlations
        pearson_corr, pearson_p = stats.pearsonr(df[x_feature].dropna(), df[y_feature].dropna())
        spearman_corr, spearman_p = stats.spearmanr(df[x_feature].dropna(), df[y_feature].dropna())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("**Pearson Correlation**")
            st.write(f"Correlation coefficient: {pearson_corr:.3f}")
            st.write(f"p-value: {pearson_p:.6f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("**Spearman Correlation**")
            st.write(f"Correlation coefficient: {spearman_corr:.3f}")
            st.write(f"p-value: {spearman_p:.6f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Interpretation
        st.subheader("Interpretation")
        
        # Based on Pearson correlation
        if abs(pearson_corr) < 0.3:
            strength = "weak"
        elif abs(pearson_corr) < 0.7:
            strength = "moderate"
        else:
            strength = "strong"
        
        direction = "positive" if pearson_corr > 0 else "negative"
        
        st.write(f"There is a {strength} {direction} correlation between {x_feature} and {y_feature}.")
        
        if pearson_p < 0.05:
            st.write("This correlation is statistically significant (p < 0.05).")
        else:
            st.write("This correlation is not statistically significant (p >= 0.05).")
    else:
        st.info("Please select both X and Y features for the scatter plot.")

# Regression Modeling Page
elif page == "Regression Modeling":
    st.markdown("<h2 class='sub-header'>Regression Modeling</h2>", unsafe_allow_html=True)
    
    # Target variable selection
    target_var = st.selectbox(
        "Select target variable to predict:",
        options=numerical_features,
        index=numerical_features.index('popularity') if 'popularity' in numerical_features else 0
    )
    
    # Simplified predictor variable selection
    st.subheader("Select Variables for Prediction")
    
    # Allow multi-selection of predictors
    selected_predictors = st.multiselect(
        "Select predictor variables:",
        options=[f for f in numerical_features if f != target_var],
        default=[f for f in numerical_features[:3] if f != target_var]
    )
    
    if target_var and selected_predictors:
        # Button to trigger model building
        if st.button("Build Linear Regression Model"):
            # Get data for modeling
            model_data = df[selected_predictors + [target_var]].dropna()
            
            if len(model_data) < 10:
                st.error("Not enough data points after removing missing values. Please select different features.")
            else:
                # Split features and target
                X = model_data[selected_predictors]
                y = model_data[target_var]
                
                # Create and fit linear regression model
                model = LinearRegression()
                model.fit(X, y)
                
                # Make predictions
                y_pred = model.predict(X)
                
                # Calculate metrics
                r2 = r2_score(y, y_pred)
                
                # Display results in a simple card with changed background color
                st.markdown(
                    f"""
                    <div style="background-color: #e8f4f8; padding: 15px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #4682B4;">
                        <h3 style="margin-top: 0; color: #2C3E50;">Model Performance</h3>
                        <p style="color: #000000"><strong>R² Score:</strong> {r2:.4f}</p>
                        <p style="color: #000000"><strong>Coefficients:</strong> {', '.join([f"{pred}: {coef:.4f}" for pred, coef in zip(selected_predictors, model.coef_)])}</p>
                        <p style="color: #000000"><strong>Intercept:</strong> {model.intercept_:.4f}</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Create scatter plot for regression visualization
                st.subheader("Regression Model: Actual vs Predicted Values")
                pred_df = pd.DataFrame({
                    'Actual': y,
                    'Predicted': y_pred
                })
                
                # If dataset is large, sample for better visualization
                if len(pred_df) > 500:
                    pred_df = pred_df.sample(500, random_state=42)
                
                # Create actual vs predicted plot
                fig = px.scatter(
                    pred_df, 
                    x='Actual', 
                    y='Predicted',
                    opacity=0.7,
                    title=f'Linear Regression: Actual vs Predicted {target_var}'
                )
                
                # Add diagonal reference line (perfect predictions)
                min_val = min(pred_df['Actual'].min(), pred_df['Predicted'].min())
                max_val = max(pred_df['Actual'].max(), pred_df['Predicted'].max())
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        line=dict(dash='dash', color='red', width=2),
                        name='Perfect Prediction'
                    )
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title=f'Actual {target_var}',
                    yaxis_title=f'Predicted {target_var}',
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # If we have exactly one predictor, show a simple regression line plot
                if len(selected_predictors) == 1:
                    predictor = selected_predictors[0]
                    st.subheader(f"Simple Linear Regression: {target_var} = {model.coef_[0]:.4f} × {predictor} + {model.intercept_:.4f}")
                    
                    # Create regression line plot
                    fig = px.scatter(
                        model_data,
                        x=predictor,
                        y=target_var,
                        opacity=0.7,
                        trendline='ols',
                        title=f'Linear Regression Line: {target_var} vs {predictor}'
                    )
                    
                    fig.update_layout(
                        xaxis_title=predictor,
                        yaxis_title=target_var,
                        template='plotly_white',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select target and predictor variables for regression modeling.")

# About Page
elif page == "About":
    st.markdown("<h2 class='sub-header'>About</h2>", unsafe_allow_html=True)
    
    st.write("""
    ### Project Information
    
    This Spotify Track Analysis App was developed as part of a data analysis project. 
    It provides statistical analysis of Spotify track data, including descriptive statistics, 
    probability distributions, correlation analysis, and regression modeling.
    
    ### Data Source
    
    The app uses the `spotify_tracks.csv` dataset, which contains various audio features 
    and metadata for Spotify tracks. The dataset includes features such as:
    
    - Acousticness, danceability, energy, and other audio features
    - Track popularity
    - Release information
    - And more...
    
    ### Statistical Methods Used
    
    - Descriptive Statistics and Confidence Intervals
    - Probability Distributions
    - Correlation Analysis
    - Linear Regression Modeling
    
    ### Tools and Libraries
    
    - Streamlit for web application development
    - Pandas and NumPy for data manipulation
    - Matplotlib, Seaborn, and Plotly for data visualization
    - Scikit-learn for regression modeling
    - SciPy and StatsModels for statistical analysis
    """)

# Add a footer
st.markdown("""
---
<p style="text-align: center; color: #888888; font-size: 0.8rem;">
    Spotify Track Analysis App | Created using Streamlit and Python
</p>
""", unsafe_allow_html=True)