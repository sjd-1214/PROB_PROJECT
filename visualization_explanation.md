# Visualization Module Explanation

## Overview

The `visualization.py` module provides a suite of visualization functions for exploring and presenting the Spotify tracks dataset. It offers both static (Matplotlib/Seaborn) and interactive (Plotly) visualization capabilities, including histograms, correlation matrices, scatter plots, radar charts, and time series plots.

## Imported Libraries

- **matplotlib.pyplot (plt)**: For creating static visualizations
- **seaborn (sns)**: For statistical data visualization with enhanced aesthetics
- **plotly.express (px)**: High-level interface for interactive plotly visualizations
- **plotly.graph_objects (go)**: Low-level interface for creating custom interactive plots
- **pandas (pd)**: For data manipulation and analysis
- **numpy (np)**: For numerical operations
- **scipy.stats**: For statistical functions and tests
- **streamlit (st)**: For creating interactive web applications

## Initial Setup
```python
# Set style for matplotlib plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
```
This code sets a consistent visual style for all matplotlib plots, using a white grid background from the seaborn library and the 'viridis' color palette, which is colorblind-friendly and perceptually uniform.

## Function Explanations

### `plot_feature_distribution(df, feature, bins=30, kde=True)`

**Purpose**: Create a histogram with optional Kernel Density Estimation (KDE) to visualize the distribution of a single feature.

**Logic**:
1. Creates a figure and axis for plotting
2. Plots a histogram with optional KDE curve using seaborn
3. Adds labels, title, and formatting
4. Adds vertical lines showing the mean and median values
5. Includes a legend for reference

**Parameters**:
- `df`: DataFrame containing the data
- `feature`: Column name of the feature to plot
- `bins`: Number of histogram bins (default=30)
- `kde`: Whether to overlay a KDE plot (default=True)

**Returns**:
- Matplotlib figure object

**Key Built-in Functions**:
- **plt.subplots()**: Creates a figure and a set of subplots
- **sns.histplot()**: Creates a histogram with optional KDE
- **DataFrame.mean()** and **DataFrame.median()**: Calculate central tendency measures
- **ax.axvline()**: Adds vertical lines to the plot
- **ax.legend()**: Adds a legend to the plot

### `plot_correlation_matrix(df, features=None, method='pearson')`

**Purpose**: Create a heatmap visualization of the correlation matrix between selected features.

**Logic**:
1. If no features are specified, uses all numeric columns
2. Calculates correlation matrix using the specified method
3. Creates a heatmap visualization with correlation coefficients as annotations
4. Returns the figure object

**Parameters**:
- `df`: DataFrame containing the data
- `features`: List of features to include (default=None, which uses all numeric columns)
- `method`: Correlation method to use: 'pearson', 'kendall', or 'spearman' (default='pearson')

**Returns**:
- Matplotlib figure object

**Key Built-in Functions**:
- **DataFrame.select_dtypes()**: Selects columns with specific data types
- **DataFrame.corr()**: Computes pairwise correlation between columns
- **sns.heatmap()**: Creates a heatmap visualization with annotations

### `plot_scatter_with_trend(df, x_col, y_col, color_col=None, add_regression=True, plot_title=None)`

**Purpose**: Create an interactive scatter plot with an optional trend line to examine relationships between two variables.

**Logic**:
1. Creates an interactive scatter plot using Plotly Express
2. Optionally adds a regression trend line
3. Calculates and displays the correlation coefficient
4. Customizes layout for better appearance
5. Returns an interactive Plotly figure

**Parameters**:
- `df`: DataFrame containing the data
- `x_col`: Column name for the x-axis
- `y_col`: Column name for the y-axis
- `color_col`: Column name to use for point colors (default=None)
- `add_regression`: Whether to add a regression line (default=True)
- `plot_title`: Title for the plot (default=None)

**Returns**:
- Plotly figure object

**Key Built-in Functions**:
- **px.scatter()**: Creates an interactive scatter plot
- **DataFrame.corr()**: Computes correlation coefficient
- **fig.add_annotation()**: Adds text annotation to the plot
- **fig.update_layout()**: Customizes the plot appearance

### `plot_feature_comparison_by_category(df, feature, category, kind='box', palette='viridis')`

**Purpose**: Create a comparison plot showing the distribution of a feature across different categories.

**Logic**:
1. Creates a figure and axis for plotting
2. Based on the kind parameter, creates either:
   - Box plot: Shows median, quartiles, and outliers
   - Violin plot: Shows full distribution with kernel density estimation
   - Bar plot: Shows means with error bars for standard deviation
3. Adds labels, title, and formatting
4. Returns the figure object

**Parameters**:
- `df`: DataFrame containing the data
- `feature`: Column name of the feature to compare
- `category`: Column name of the categorical variable
- `kind`: Type of plot: 'box', 'violin', or 'bar' (default='box')
- `palette`: Color palette to use (default='viridis')

**Returns**:
- Matplotlib figure object

**Key Built-in Functions**:
- **sns.boxplot()**: Creates a box plot
- **sns.violinplot()**: Creates a violin plot
- **sns.barplot()**: Creates a bar plot
- **DataFrame.groupby()**: Groups data by category for aggregation
- **ax.errorbar()**: Adds error bars to the plot

### `plot_radar_chart(df, features, group_col=None, group_val=None, title=None, fill=True)`

**Purpose**: Create a radar chart (also known as a spider or web chart) for multiple features.

**Logic**:
1. Optionally filters data based on a group column and value
2. Calculates mean values for each feature
3. Normalizes values to a 0-1 scale for better visualization
4. Creates an interactive radar chart using Plotly
5. Customizes layout and appearance
6. Returns an interactive Plotly figure

**Parameters**:
- `df`: DataFrame containing the data
- `features`: List of features to include in the radar chart
- `group_col`: Column name to filter the data by (default=None)
- `group_val`: Value to filter the group_col by (default=None)
- `title`: Title for the plot (default=None)
- `fill`: Whether to fill the radar chart (default=True)

**Returns**:
- Plotly figure object

**Key Built-in Functions**:
- **DataFrame.mean()**: Calculates average values
- **go.Figure()**: Creates a Plotly figure
- **go.Scatterpolar()**: Creates a radar chart trace
- **fig.update_layout()**: Customizes the plot appearance

### `plot_time_series(df, time_col, value_col, group_col=None, kind='line', title=None, rolling_window=None)`

**Purpose**: Create a time series plot to visualize trends over time.

**Logic**:
1. Creates a time series plot using Plotly Express
2. Supports different visualization types: line, scatter, or area
3. Optionally groups data by a category
4. Optionally adds a rolling average smoothing
5. Customizes layout and appearance
6. Returns an interactive Plotly figure

**Parameters**:
- `df`: DataFrame containing the data
- `time_col`: Column name for the time axis
- `value_col`: Column name for the value to plot
- `group_col`: Column name to group the data by (default=None)
- `kind`: Type of plot: 'line', 'scatter', or 'area' (default='line')
- `title`: Title for the plot (default=None)
- `rolling_window`: Window size for rolling average (default=None)

**Returns**:
- Plotly figure object

**Key Built-in Functions**:
- **DataFrame.sort_values()**: Sorts data by time for proper time series display
- **DataFrame.rolling()**: Creates a rolling window calculation
- **px.line()**, **px.scatter()**, **px.area()**: Create different types of time series plots
- **fig.add_scatter()**: Adds additional traces to the plot

### `plot_feature_importance(importance_df, title="Feature Importance")`

**Purpose**: Create a bar chart visualization of feature importance from a machine learning model.

**Logic**:
1. Checks that the DataFrame has the necessary columns
2. Sorts features by importance value
3. Creates a color-coded bar chart using Plotly Express
4. Customizes layout and appearance
5. Returns an interactive Plotly figure

**Parameters**:
- `importance_df`: DataFrame with 'Feature' and importance value columns
- `title`: Title for the plot (default="Feature Importance")

**Returns**:
- Plotly figure object

**Key Built-in Functions**:
- **DataFrame.sort_values()**: Sorts features by importance
- **px.bar()**: Creates a bar chart
- **fig.update_layout()**: Customizes the plot appearance

### `create_streamlit_visualization_grid(plots_info, cols=2)`

**Purpose**: Create a grid layout of visualizations in a Streamlit app.

**Logic**:
1. Calculates the number of rows needed based on the number of plots and columns
2. Creates a grid layout using Streamlit columns
3. Adds each plot to the appropriate column in the grid
4. Displays either Matplotlib or Plotly plots as specified

**Parameters**:
- `plots_info`: List of dictionaries containing:
  - 'title': Title for the plot
  - 'plot': The plot object (Matplotlib or Plotly)
  - 'type': Type of plot: 'matplotlib' or 'plotly'
- `cols`: Number of columns in the grid (default=2)

**Key Built-in Functions**:
- **st.columns()**: Creates a row of columns in Streamlit
- **st.subheader()**: Adds a subheading
- **st.pyplot()**: Displays a Matplotlib figure
- **st.plotly_chart()**: Displays a Plotly figure

## Usage Example

```python
# Import functions
from visualization import plot_feature_distribution, plot_correlation_matrix, plot_scatter_with_trend

# Load data (assuming spotify_df is loaded)

# Create a distribution plot
dist_fig = plot_feature_distribution(spotify_df, 'danceability')
dist_fig.savefig('danceability_distribution.png')

# Create a correlation matrix
correlation_fig = plot_correlation_matrix(
    spotify_df, 
    features=['danceability', 'energy', 'loudness', 'tempo']
)
correlation_fig.savefig('correlation_matrix.png')

# Create an interactive scatter plot
scatter_fig = plot_scatter_with_trend(
    spotify_df,
    x_col='energy',
    y_col='loudness',
    color_col='popularity',
    add_regression=True
)
# For interactive plots, you would typically display them in a Streamlit app
# or export to HTML with scatter_fig.write_html('scatter_plot.html')
```
