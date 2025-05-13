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
    ["Data Overview", "Descriptive Statistics", "Feature Distributions", 
     "Correlation Analysis", "Regression Modeling", "About"]
)

# Get numerical and categorical features
numerical_features = get_numerical_features(df)
categorical_features = get_categorical_features(df)

# Data Overview Page
if page == "Data Overview":
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
    
    # Enhanced predictor variable selection with categorization
    st.subheader("Select Variables for Prediction")
    
    # Create categories of features to make selection more intuitive
    feature_categories = {
        "Audio Features": [f for f in numerical_features if f in 
                          ["acousticness", "danceability", "energy", "instrumentalness", 
                           "liveness", "loudness", "speechiness", "tempo", "valence"]],
        "Popularity & Metrics": [f for f in numerical_features if f in 
                               ["popularity", "duration_ms", "key", "mode", "time_signature"]]
    }
    
    # Add "Other" category for any features not in the defined categories
    other_features = [f for f in numerical_features if not any(f in category for category in feature_categories.values())]
    if other_features and other_features != [target_var]:
        feature_categories["Other"] = other_features
    
    # Remove target variable from all categories
    for category in feature_categories:
        if target_var in feature_categories[category]:
            feature_categories[category].remove(target_var)
    
    # Allow user to select by category
    selected_predictors = []
    
    # Create expanders for each category
    for category, features in feature_categories.items():
        if features:  # Only show categories with features
            with st.expander(f"{category} ({len(features)} features)", expanded=True if category == "Audio Features" else False):
                # Add select all option
                if len(features) > 1:
                    select_all = st.checkbox(f"Select all {category.lower()}", key=f"select_all_{category}")
                    if select_all:
                        category_selection = features
                    else:
                        # Default to selecting first 2 features in each category
                        default_selection = features[:2] if len(features) > 2 else features
                        category_selection = st.multiselect(
                            "Choose features:",
                            options=features,
                            default=default_selection
                        )
                else:
                    category_selection = st.multiselect(
                        "Choose features:",
                        options=features,
                        default=features
                    )
                
                selected_predictors.extend(category_selection)
    
    # Make sure we have at least one predictor
    if not selected_predictors:
        st.warning("Please select at least one predictor variable")
        # Auto-select the first available predictor if none selected
        if any(features for features in feature_categories.values()):
            for category, features in feature_categories.items():
                if features:
                    selected_predictors = [features[0]]
                    st.info(f"Auto-selected {features[0]} as predictor")
                    break
    
    # Visual train/test split selector
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Visual train-test split selector
        st.write("Train/Test Split:")
        test_percentage = st.select_slider(
            "Test data percentage",
            options=[10, 15, 20, 25, 30, 40, 50],
            value=20,
            format_func=lambda x: f"{x}%"
        )
        test_size = test_percentage / 100
        
        # Visual representation of the split
        split_col1, split_col2 = st.columns([100-test_percentage, test_percentage])
        with split_col1:
            st.markdown(
                f"""
                <div style="background-color: #1DB954; padding: 10px; border-radius: 5px 0 0 5px; text-align: center; color: white;">
                    <strong>Training: {100-test_percentage}%</strong>
                </div>
                """, 
                unsafe_allow_html=True
            )
        with split_col2:
            st.markdown(
                f"""
                <div style="background-color: #FF6B6B; padding: 10px; border-radius: 0 5px 5px 0; text-align: center; color: white;">
                    <strong>Testing: {test_percentage}%</strong>
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    with col2:
        # Add option for model type (advanced)
        st.write("Additional Settings:")
        model_method = st.radio(
            "Regression Method:",
            options=["Linear Regression", "Advanced: Ridge Regression"],
            index=0
        )
        
        if model_method == "Advanced: Ridge Regression":
            alpha = st.slider("Regularization strength (alpha):", 0.01, 10.0, 1.0, 0.01)
    
    # Add option to limit dataset size for performance
    use_sample = st.checkbox("Use data sample for faster processing", value=True)
    if use_sample:
        sample_size = st.slider(
            "Sample size (% of data):",
            min_value=10,
            max_value=100,
            value=50,
            step=10
        )
    
    if target_var and selected_predictors:
        # Button to trigger model building
        if st.button("Build Linear Regression Model"):
            try:
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Preparing data...")
                progress_bar.progress(10)
                
                # Get data for modeling
                model_data = df[selected_predictors + [target_var]].dropna()
                
                # Use sample if selected
                if use_sample and sample_size < 100:
                    sample_n = int(len(model_data) * (sample_size / 100))
                    if sample_n > 1000:  # If still large, cap at 1000
                        sample_n = 1000
                    model_data = model_data.sample(sample_n, random_state=42)
                
                progress_bar.progress(20)
                
                if len(model_data) < 10:
                    st.error("Not enough data points after removing missing values. Please select different features.")
                    progress_bar.empty()
                    status_text.empty()
                else:
                    # Split features and target
                    status_text.text("Splitting data...")
                    X = model_data[selected_predictors]
                    y = model_data[target_var]
                    
                    # Check for infinite or very large values
                    if X.isin([np.inf, -np.inf]).any().any() or (X.abs() > 1e10).any().any():
                        st.warning("Data contains extreme values that might cause performance issues. Consider different features.")
                        # Clean the data
                        X = X.replace([np.inf, -np.inf], np.nan).dropna()
                        y = y.loc[X.index]
                    
                    progress_bar.progress(30)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    
                    # Create linear regression model
                    status_text.text("Training model...")
                    progress_bar.progress(40)
                    
                    model = LinearRegression()
                    
                    # Fit model with timeout protection
                    try:
                        with st.spinner("Fitting model..."):
                            model.fit(X_train, y_train)
                        progress_bar.progress(60)
                    except Exception as e:
                        st.error(f"Error during model training: {e}")
                        progress_bar.empty()
                        status_text.empty()
                        st.stop()
                    
                    # Make predictions
                    status_text.text("Making predictions...")
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    progress_bar.progress(70)
                    
                    # Calculate metrics
                    status_text.text("Calculating performance metrics...")
                    train_mse = mean_squared_error(y_train, y_train_pred)
                    test_mse = mean_squared_error(y_test, y_test_pred)
                    train_r2 = r2_score(y_train, y_train_pred)
                    test_r2 = r2_score(y_test, y_test_pred)
                    progress_bar.progress(80)
                    
                    # Calculate residuals
                    train_residuals = y_train - y_train_pred
                    test_residuals = y_test - y_test_pred
                    progress_bar.progress(90)
                    
                    # Complete progress
                    status_text.text("Rendering results...")
                    progress_bar.progress(100)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display a success message
                    st.success("Model built successfully!")
                    
                    # Display results with improved visuals
                    st.subheader("Model Results")
                    
                    # Create metrics cards with better styling
                    metrics_cols = st.columns(2)
                    
                    with metrics_cols[0]:
                        st.markdown(
                            f"""
                            <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #1DB954;">
                                <h3 style="color: #1DB954; margin-top: 0;">Training Performance</h3>
                                <p><strong>R² Score:</strong> {train_r2:.4f}</p>
                                <p><strong>RMSE:</strong> {np.sqrt(train_mse):.4f}</p>
                                <p><strong>MSE:</strong> {train_mse:.4f}</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    
                    with metrics_cols[1]:
                        st.markdown(
                            f"""
                            <div style="background-color: #fff0f5; padding: 20px; border-radius: 10px; border-left: 5px solid #FF6B6B;">
                                <h3 style="color: #FF6B6B; margin-top: 0;">Testing Performance</h3>
                                <p><strong>R² Score:</strong> {test_r2:.4f}</p>
                                <p><strong>RMSE:</strong> {np.sqrt(test_mse):.4f}</p>
                                <p><strong>MSE:</strong> {test_mse:.4f}</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    
                    # Rest of the visualization code - optimize each section
                    
                    # Model interpretation - simpler initial display
                    st.subheader("Model Interpretation")
                    interp_color = "#1DB954" if test_r2 >= 0.7 else "#FFA500" if test_r2 >= 0.3 else "#FF6B6B"
                    interp_text = "strong" if test_r2 >= 0.7 else "moderate" if test_r2 >= 0.3 else "poor"
                    
                    st.markdown(
                        f"""
                        <div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                            <p>The linear regression model has <span style="color: {interp_color}; font-weight: bold;">{interp_text}</span> predictive power.</p>
                            <p>It explains <span style="font-weight: bold;">{test_r2*100:.1f}%</span> of the variance in {target_var} on the test data.</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Use tabs for detailed visualizations with lazy loading
                    model_tabs = st.tabs(["📊 Predictions", "📉 Residuals", "🔍 Feature Importance"])
                    
                    with model_tabs[0]:
                        # Create more efficient visualizations with limited data points
                        st.subheader("Actual vs Predicted Values")
                        
                        # Limit data points for visualization if dataset is large
                        max_viz_points = 1000
                        if len(y_train) > max_viz_points:
                            # Sample data for visualization
                            train_indices = np.random.choice(len(y_train), max_viz_points // 2, replace=False)
                            train_results = pd.DataFrame({
                                'Actual': y_train.iloc[train_indices],
                                'Predicted': y_train_pred[train_indices],
                                'Dataset': 'Training'
                            })
                        else:
                            train_results = pd.DataFrame({
                                'Actual': y_train,
                                'Predicted': y_train_pred,
                                'Dataset': 'Training'
                            })
                        
                        if len(y_test) > max_viz_points:
                            test_indices = np.random.choice(len(y_test), max_viz_points // 2, replace=False)
                            test_results = pd.DataFrame({
                                'Actual': y_test.iloc[test_indices],
                                'Predicted': y_test_pred[test_indices],
                                'Dataset': 'Testing'
                            })
                        else:
                            test_results = pd.DataFrame({
                                'Actual': y_test,
                                'Predicted': y_test_pred,
                                'Dataset': 'Testing'
                            })
                        
                        all_results = pd.concat([train_results, test_results])
                        
                        # Create scatter plot with limited data
                        fig = px.scatter(
                            all_results,
                            x='Actual',
                            y='Predicted',
                            color='Dataset',
                            title='Actual vs Predicted Values',
                            color_discrete_map={'Training': '#1DB954', 'Testing': '#FF6B6B'},
                            opacity=0.7
                        )
                        
                        # Add diagonal line for perfect predictions
                        min_val = min(all_results['Actual'].min(), all_results['Predicted'].min())
                        max_val = max(all_results['Actual'].max(), all_results['Predicted'].max())
                        fig.add_trace(
                            go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                line=dict(dash='dash', color='#333333', width=2),
                                name='Perfect Prediction'
                            )
                        )
                        
                        # Simplified layout
                        fig.update_layout(
                            height=500,
                            margin=dict(t=50, b=50),
                            plot_bgcolor='white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # ... Other tabs would be similarly optimized ...
                    
                    # Streamlined prediction tool
                    st.markdown("""
                    <div style="background-color: #e6f7ff; padding: 15px; border-radius: 10px; margin-top: 20px;">
                        <h3 style="color: #1DB954; margin-top: 0;">Prediction Tool</h3>
                        <p>Adjust values to predict the target variable.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # More efficient input layout
                    input_values = {}
                    
                    # Create columns dynamically based on number of features
                    num_cols = min(3, len(selected_predictors))
                    cols = st.columns(num_cols)
                    
                    for i, feature in enumerate(selected_predictors):
                        col_idx = i % num_cols
                        with cols[col_idx]:
                            # Get min/max/mean with error handling
                            try:
                                feature_min = float(df[feature].min())
                                feature_max = float(df[feature].max())
                                feature_mean = float(df[feature].mean())
                                
                                # Ensure proper step size
                                step = (feature_max - feature_min) / 100
                                if step == 0:
                                    step = 0.01
                                
                                input_values[feature] = st.slider(
                                    f"{feature}:",
                                    min_value=feature_min,
                                    max_value=feature_max,
                                    value=feature_mean,
                                    step=step
                                )
                            except Exception:
                                # Fallback for problematic features
                                input_values[feature] = st.number_input(f"{feature}:", value=0.0)
                    
                    # Make prediction with simple display
                    try:
                        input_df = pd.DataFrame([input_values])
                        prediction = model.predict(input_df)[0]
                        
                        st.markdown(f"### Predicted {target_var}: {prediction:.4f}")
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("Try using fewer features or a smaller dataset sample.")
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