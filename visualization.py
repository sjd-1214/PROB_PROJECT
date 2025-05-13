import matplotlib.pyplot as plt  # Plotting library for creating static, interactive, and animated visualizations
import seaborn as sns           # Statistical data visualization library based on matplotlib
import plotly.express as px     # High-level interface for interactive plotly visualizations
import plotly.graph_objects as go  # Low-level interface for creating custom interactive plots
import pandas as pd             # Data manipulation and analysis library for tabular data structures
import numpy as np              # Scientific computing library for numerical operations and array handling
from scipy import stats         # Statistical functions and tests from SciPy
import streamlit as st          # Web app framework for creating interactive data applications

# Set style for matplotlib plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

def plot_feature_distribution(df, feature, bins=30, kde=True):
    """
    Create a histogram with optional KDE for a single feature.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    feature : str
        Column name of the feature to plot
    bins : int, default=30
        Number of histogram bins
    kde : bool, default=True
        Whether to overlay a KDE plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create the histogram with KDE
    sns.histplot(df[feature].dropna(), bins=bins, kde=kde, ax=ax)
    
    # Add labels and title
    ax.set_xlabel(feature.capitalize(), fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Distribution of {feature.capitalize()}', fontsize=14)
    
    # Add mean and median lines
    mean_val = df[feature].mean()
    median_val = df[feature].median()
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, 
               label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='-.', linewidth=1.5, 
               label=f'Median: {median_val:.2f}')
    
    ax.legend()
    
    return fig

def plot_correlation_matrix(df, features=None, method='pearson'):
    """
    Create a correlation matrix heatmap.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    features : list, default=None
        List of features to include in the correlation matrix
        If None, uses all numeric columns
    method : {'pearson', 'kendall', 'spearman'}, default='pearson'
        Correlation method to use
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the plot
    """
    # If no features are specified, use all numeric columns
    if features is None:
        features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[features].corr(method=method)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    heatmap = sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt='.2f', 
        cmap='coolwarm',
        linewidths=0.5,
        ax=ax
    )
    
    # Set title
    plt.title(f'{method.capitalize()} Correlation Matrix', fontsize=14)
    
    return fig

def plot_scatter_with_trend(df, x_col, y_col, color_col=None, 
                            add_regression=True, plot_title=None):
    """
    Create a scatter plot with an optional trend line.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    x_col : str
        Column name for the x-axis
    y_col : str
        Column name for the y-axis
    color_col : str, default=None
        Column name to use for point colors
    add_regression : bool, default=True
        Whether to add a regression line
    plot_title : str, default=None
        Title for the plot
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The Plotly figure object containing the plot
    """
    # Default title if none provided
    if plot_title is None:
        plot_title = f'{y_col.capitalize()} vs {x_col.capitalize()}'
    
    # Create scatter plot
    if color_col:
        fig = px.scatter(
            df, x=x_col, y=y_col, color=color_col, 
            opacity=0.7, title=plot_title,
            trendline='ols' if add_regression else None
        )
    else:
        fig = px.scatter(
            df, x=x_col, y=y_col,
            opacity=0.7, title=plot_title,
            trendline='ols' if add_regression else None
        )
    
    # Calculate correlation coefficient
    valid_data = df[[x_col, y_col]].dropna()
    corr_coef = valid_data[x_col].corr(valid_data[y_col])
    
    # Add correlation annotation
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        text=f"Correlation: {corr_coef:.3f}",
        showarrow=False,
        font=dict(size=14, color="#000000"),
        bgcolor="#ffffff",
        bordercolor="#000000",
        borderwidth=1,
        borderpad=4
    )
    
    # Update layout for better appearance
    fig.update_layout(
        template="plotly_white",
        xaxis_title=x_col.capitalize(),
        yaxis_title=y_col.capitalize(),
        legend_title_text=color_col.capitalize() if color_col else None
    )
    
    return fig

def plot_feature_comparison_by_category(df, feature, category, 
                                         kind='box', palette='viridis'):
    """
    Create a comparison plot of a feature across categories.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    feature : str
        Column name of the feature to compare
    category : str
        Column name of the categorical variable
    kind : {'box', 'violin', 'bar'}, default='box'
        Type of plot to create
    palette : str or list, default='viridis'
        Color palette to use
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if kind == 'box':
        # Create box plot
        sns.boxplot(x=category, y=feature, data=df, palette=palette, ax=ax)
        
    elif kind == 'violin':
        # Create violin plot
        sns.violinplot(x=category, y=feature, data=df, palette=palette, ax=ax)
        
    elif kind == 'bar':
        # Create bar plot of means with error bars
        aggregated = df.groupby(category)[feature].agg(['mean', 'std']).reset_index()
        sns.barplot(x=category, y='mean', data=aggregated, palette=palette, ax=ax)
        
        # Add error bars
        for i, row in aggregated.iterrows():
            ax.errorbar(i, row['mean'], yerr=row['std'], color='black', capsize=5)
    
    # Add labels and title
    ax.set_xlabel(category.capitalize(), fontsize=12)
    ax.set_ylabel(feature.capitalize(), fontsize=12)
    ax.set_title(f'{feature.capitalize()} by {category.capitalize()}', fontsize=14)
    
    # Rotate x-axis labels if there are many categories
    if df[category].nunique() > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def plot_radar_chart(df, features, group_col=None, group_val=None, 
                     title=None, fill=True):
    """
    Create a radar chart for specified features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    features : list
        List of features to include in the radar chart
    group_col : str, default=None
        Column name to filter the data by
    group_val : any, default=None
        Value to filter the group_col by
    title : str, default=None
        Title for the plot
    fill : bool, default=True
        Whether to fill the radar chart
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The Plotly figure object containing the radar chart
    """
    # Filter data if specified
    if group_col is not None and group_val is not None:
        plot_df = df[df[group_col] == group_val]
    else:
        plot_df = df
    
    # Calculate mean values for each feature
    feature_means = [plot_df[feature].mean() for feature in features]
    
    # Normalize the values to 0-1 scale for better visualization
    feature_mins = [df[feature].min() for feature in features]
    feature_maxs = [df[feature].max() for feature in features]
    
    normalized_means = [(mean - min_val) / (max_val - min_val) 
                         if max_val > min_val else 0.5
                         for mean, min_val, max_val 
                         in zip(feature_means, feature_mins, feature_maxs)]
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_means + [normalized_means[0]],  # Close the loop
        theta=features + [features[0]],  # Close the loop
        fill='toself' if fill else None,
        name='Mean Values',
        line_color='rgba(31, 119, 180, 0.8)',
        fillcolor='rgba(31, 119, 180, 0.3)' if fill else None
    ))
    
    # Set chart title
    if title is None:
        if group_col is not None and group_val is not None:
            title = f'Feature Radar Chart for {group_col}={group_val}'
        else:
            title = 'Feature Radar Chart'
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title=title
    )
    
    return fig

def plot_time_series(df, time_col, value_col, group_col=None, kind='line',
                     title=None, rolling_window=None):
    """
    Create a time series plot.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    time_col : str
        Column name for the time axis
    value_col : str
        Column name for the value to plot
    group_col : str, default=None
        Column name to group the data by
    kind : {'line', 'scatter', 'area'}, default='line'
        Type of plot to create
    title : str, default=None
        Title for the plot
    rolling_window : int, default=None
        Window size for rolling average, if desired
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The Plotly figure object containing the plot
    """
    # Set default title if none provided
    if title is None:
        title = f'{value_col.capitalize()} over {time_col.capitalize()}'
    
    # Create time series plot
    if group_col:
        if kind == 'line':
            fig = px.line(df, x=time_col, y=value_col, color=group_col,
                         title=title)
        elif kind == 'scatter':
            fig = px.scatter(df, x=time_col, y=value_col, color=group_col,
                            title=title)
        elif kind == 'area':
            fig = px.area(df, x=time_col, y=value_col, color=group_col,
                         title=title)
    else:
        if rolling_window:
            # Calculate rolling average
            rolling_data = df.sort_values(time_col)
            rolling_data[f'{value_col}_rolling'] = rolling_data[value_col].rolling(
                window=rolling_window, center=True).mean()
            
            if kind == 'line':
                fig = px.line(rolling_data, x=time_col, y=[value_col, f'{value_col}_rolling'],
                             title=title)
            elif kind == 'scatter':
                fig = px.scatter(rolling_data, x=time_col, y=value_col,
                                title=title)
                fig.add_scatter(x=rolling_data[time_col], y=rolling_data[f'{value_col}_rolling'],
                               mode='lines', name=f'{rolling_window}-point Rolling Average')
            elif kind == 'area':
                fig = px.area(rolling_data, x=time_col, y=value_col,
                             title=title)
        else:
            if kind == 'line':
                fig = px.line(df, x=time_col, y=value_col, title=title)
            elif kind == 'scatter':
                fig = px.scatter(df, x=time_col, y=value_col, title=title)
            elif kind == 'area':
                fig = px.area(df, x=time_col, y=value_col, title=title)
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        xaxis_title=time_col.capitalize(),
        yaxis_title=value_col.capitalize()
    )
    
    return fig

def plot_feature_importance(importance_df, title="Feature Importance"):
    """
    Create a bar chart of feature importance.
    
    Parameters:
    -----------
    importance_df : pandas.DataFrame
        DataFrame with 'Feature' and importance value columns
    title : str, default="Feature Importance"
        Title for the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The Plotly figure object containing the plot
    """
    # Check if the DataFrame has necessary columns
    if 'Feature' not in importance_df.columns:
        raise ValueError("DataFrame must have a 'Feature' column")
    
    # Get the importance column (assuming it's the second column if not specified)
    importance_col = [col for col in importance_df.columns if col != 'Feature'][0]
    
    # Sort by importance value
    sorted_df = importance_df.sort_values(importance_col, ascending=False)
    
    # Create bar chart
    fig = px.bar(
        sorted_df,
        x='Feature',
        y=importance_col,
        title=title,
        color=importance_col,
        color_continuous_scale='viridis'
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Features",
        yaxis_title=importance_col.capitalize()
    )
    
    return fig

def create_streamlit_visualization_grid(plots_info, cols=2):
    """
    Create a grid of visualizations in Streamlit.
    
    Parameters:
    -----------
    plots_info : list of dicts
        List of dictionaries containing:
            'title': str - Title for the plot
            'plot': matplotlib.figure.Figure or plotly.graph_objects.Figure - The plot object
            'type': str - Type of plot: 'matplotlib' or 'plotly'
    cols : int, default=2
        Number of columns in the grid
    """
    # Calculate number of rows needed
    n_plots = len(plots_info)
    n_rows = (n_plots + cols - 1) // cols
    
    # Create each row
    for i in range(n_rows):
        row_cols = st.columns(cols)
        
        # Add plots to this row
        for j in range(cols):
            idx = i * cols + j
            if idx < n_plots:
                with row_cols[j]:
                    plot_info = plots_info[idx]
                    st.subheader(plot_info['title'])
                    
                    if plot_info['type'] == 'matplotlib':
                        st.pyplot(plot_info['plot'])
                    elif plot_info['type'] == 'plotly':
                        st.plotly_chart(plot_info['plot'], use_container_width=True)
