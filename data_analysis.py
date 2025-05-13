import pandas as pd             # Data manipulation and analysis library for tabular data structures
import numpy as np              # Scientific computing library for numerical operations and array handling
import matplotlib.pyplot as plt  # Plotting library for creating static, interactive, and animated visualizations
import seaborn as sns           # Statistical data visualization library based on matplotlib
import plotly.express as px     # High-level interface for interactive plotly visualizations
import plotly.graph_objects as go  # Low-level interface for creating custom interactive plots
from scipy import stats         # Statistical functions and tests from SciPy
from sklearn.linear_model import LinearRegression  # Linear regression models from scikit-learn
from sklearn.ensemble import RandomForestRegressor  # Random forest regression from scikit-learn
from sklearn.model_selection import train_test_split  # Utilities for splitting datasets for training/testing
from sklearn.metrics import mean_squared_error, r2_score  # Evaluation metrics for regression models
import statsmodels.api as sm    # Statistical models for regression, time series, and more

def generate_descriptive_stats(df, features):
    """
    Generate descriptive statistics for the given numerical features
    """
    stats_df = df[features].describe().T
    # Add additional statistical measures
    stats_df['median'] = df[features].median()
    stats_df['skewness'] = df[features].skew()
    stats_df['kurtosis'] = df[features].kurtosis()
    
    # Calculate confidence intervals (95%)
    ci_low = []
    ci_high = []
    
    for feature in features:
        mean = df[feature].mean()
        std_err = stats.sem(df[feature].dropna())
        ci = stats.t.interval(0.95, len(df[feature].dropna())-1, loc=mean, scale=std_err)
        ci_low.append(ci[0])
        ci_high.append(ci[1])
    
    stats_df['95%_CI_low'] = ci_low
    stats_df['95%_CI_high'] = ci_high
    
    return stats_df

def create_correlation_heatmap(df, features):
    """
    Create a correlation heatmap for the given features
    """
    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
    plt.title('Correlation Heatmap of Spotify Track Features')
    return fig

def create_distribution_plot(df, feature):
    """
    Create a distribution plot for a given feature
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram with KDE
    sns.histplot(df[feature].dropna(), kde=True, ax=ax)
    
    # Add a normal distribution line for comparison
    x = np.linspace(df[feature].min(), df[feature].max(), 100)
    mean = df[feature].mean()
    std = df[feature].std()
    y = stats.norm.pdf(x, mean, std)
    y = y * (len(df) * df[feature].std() / 5)  # Scale the normal distribution
    ax.plot(x, y, 'r-', label='Normal Distribution')
    
    # Add labels and title
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {feature}')
    ax.legend()
    
    # Add statistical annotations
    skew = df[feature].skew()
    kurt = df[feature].kurtosis()
    
    stats_text = f"Mean: {mean:.2f}\nStd Dev: {std:.2f}\nSkewness: {skew:.2f}\nKurtosis: {kurt:.2f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig

def build_regression_model(df, target='popularity', features=None):
    """
    Build a regression model to predict a target variable
    """
    if features is None:
        # Use default audio features if none specified
        features = ['danceability', 'energy', 'loudness', 'speechiness', 
                   'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    # Use only numeric features that are in the dataframe
    features = [f for f in features if f in df.columns and f != target and 
                pd.api.types.is_numeric_dtype(df[f])]
    
    if not features:
        return None, None, None, None, "No valid features found for regression"
    
    # Prepare data
    X = df[features].dropna()
    y = df[target].loc[X.index]
    
    # Check if we have enough data
    if len(X) < 10:
        return None, None, None, None, "Not enough data for regression"
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Get feature importance
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)
    
    # Statistical model using statsmodels for p-values
    X_with_const = sm.add_constant(X_train)
    sm_model = sm.OLS(y_train, X_with_const).fit()
    
    return model, coef_df, mse, r2, sm_model.summary()

def create_scatter_plot(df, x_feature, y_feature):
    """
    Create a scatter plot between two features
    """
    fig = px.scatter(df, x=x_feature, y=y_feature, 
                    opacity=0.6,
                    trendline="ols",
                    color_discrete_sequence=['#1DB954'])  # Spotify green
    
    # Calculate correlation coefficient
    corr = df[[x_feature, y_feature]].corr().iloc[0, 1]
    
    # Add annotation with correlation value
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        text=f"Correlation: {corr:.3f}",
        showarrow=False,
        font=dict(size=14, color="#333333"),
        bgcolor="#FFFFFF",
        bordercolor="#333333",
        borderwidth=1,
        borderpad=4
    )
    
    # Update layout
    fig.update_layout(
        title=f'Relationship between {x_feature} and {y_feature}',
        xaxis_title=x_feature,
        yaxis_title=y_feature,
        template="plotly_white"
    )
    
    return fig
