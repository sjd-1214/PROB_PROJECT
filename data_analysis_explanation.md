# Data Analysis Module Explanation

## Overview

The `data_analysis.py` module provides statistical analysis and visualization functions for the Spotify tracks dataset. It includes capabilities for generating descriptive statistics, creating correlation heatmaps, distribution plots, regression models, and scatter plots.

## Imported Libraries

- **pandas (pd)**: For data manipulation and analysis
- **numpy (np)**: For numerical operations and array handling
- **matplotlib.pyplot (plt)**: For creating static visualizations
- **seaborn (sns)**: For statistical data visualization
- **plotly.express (px)** and **plotly.graph_objects (go)**: For interactive visualizations
- **scipy.stats**: For statistical functions and tests
- **sklearn.linear_model.LinearRegression**: For linear regression modeling
- **sklearn.ensemble.RandomForestRegressor**: For random forest regression modeling
- **sklearn.model_selection.train_test_split**: For splitting data into training and testing sets
- **sklearn.metrics**: For regression evaluation metrics
- **statsmodels.api (sm)**: For statistical models with detailed statistical output

## Function Explanations

### `generate_descriptive_stats(df, features)`

**Purpose**: Generate comprehensive descriptive statistics for specified numerical features.

**Logic**:
1. Uses pandas' `describe()` method to get basic statistics (count, mean, std, min, 25%, 50%, 75%, max)
2. Adds additional statistics: median, skewness, and kurtosis
3. Calculates 95% confidence intervals for the mean of each feature using t-distribution
4. Returns a DataFrame with all statistics

**Parameters**:
- `df`: DataFrame containing the data
- `features`: List of features to analyze

**Returns**:
- DataFrame with descriptive statistics for each specified feature

**Built-in Functions Used**:
- **DataFrame.describe()**: Generates basic descriptive statistics
- **DataFrame.median()**: Calculates the median
- **DataFrame.skew()**: Measures asymmetry of the distribution
- **DataFrame.kurtosis()**: Measures the "tailedness" of the distribution
- **stats.sem()**: Calculates standard error of the mean
- **stats.t.interval()**: Computes the confidence interval using t-distribution

### `create_correlation_heatmap(df, features)`

**Purpose**: Create a heatmap visualization of the correlation matrix.

**Logic**:
1. Calculates the correlation matrix for the specified features
2. Creates a matplotlib figure and axis
3. Uses seaborn's heatmap to visualize the correlation matrix
4. Includes correlation coefficients as annotations
5. Returns the figure object

**Parameters**:
- `df`: DataFrame containing the data
- `features`: List of features to include in the correlation matrix

**Returns**:
- Matplotlib figure object containing the heatmap

**Built-in Functions Used**:
- **DataFrame.corr()**: Computes pairwise correlation between columns
- **plt.subplots()**: Creates a figure and a set of subplots
- **sns.heatmap()**: Creates a heatmap visualization
- **plt.title()**: Sets the title of the plot

### `create_distribution_plot(df, feature)`

**Purpose**: Create a histogram with KDE (Kernel Density Estimation) to visualize the distribution of a feature.

**Logic**:
1. Creates a matplotlib figure and axis
2. Plots a histogram with KDE for the specified feature
3. Adds a normal distribution curve for comparison
4. Annotates the plot with statistical information (mean, std dev, skewness, kurtosis)
5. Returns the figure object

**Parameters**:
- `df`: DataFrame containing the data
- `feature`: Column name of the feature to plot

**Returns**:
- Matplotlib figure object containing the distribution plot

**Built-in Functions Used**:
- **plt.subplots()**: Creates a figure and a set of subplots
- **sns.histplot()**: Creates a histogram with optional KDE
- **np.linspace()**: Returns evenly spaced numbers over an interval
- **stats.norm.pdf()**: Computes the probability density function of the normal distribution
- **ax.text()**: Adds text to the plot at specified position
- **DataFrame.skew()**: Measures asymmetry of the distribution
- **DataFrame.kurtosis()**: Measures the "tailedness" of the distribution

### `build_regression_model(df, target='popularity', features=None)`

**Purpose**: Build a linear regression model to predict a target variable based on selected features.

**Logic**:
1. If no features are specified, uses default audio features
2. Filters features to include only numeric columns that are present in the DataFrame
3. Prepares data by removing rows with missing values
4. Splits data into training and testing sets
5. Fits a linear regression model to the training data
6. Evaluates the model performance using Mean Squared Error (MSE) and R² score
7. Creates a DataFrame with feature coefficients for importance analysis
8. Uses statsmodels to get detailed statistical information about the model
9. Returns the model, feature importance, performance metrics, and statistical summary

**Parameters**:
- `df`: DataFrame containing the data
- `target`: Target variable to predict (default is 'popularity')
- `features`: List of features to use as predictors (default is None)

**Returns**:
- Tuple containing:
  - The trained LinearRegression model
  - DataFrame with feature coefficients
  - Mean Squared Error
  - R² score
  - Statistical summary from statsmodels

**Built-in Functions Used**:
- **pd.api.types.is_numeric_dtype()**: Checks if a pandas object has numeric data type
- **DataFrame.dropna()**: Removes rows with missing values
- **train_test_split()**: Splits arrays into random train and test subsets
- **LinearRegression()**: Creates a linear regression model
- **model.fit()**: Fits the model to training data
- **model.predict()**: Makes predictions using the model
- **mean_squared_error()**: Computes mean squared error
- **r2_score()**: Computes coefficient of determination (R²)
- **sm.add_constant()**: Adds a constant term to a predictor matrix
- **sm.OLS()**: Creates an Ordinary Least Squares regression model

### `create_scatter_plot(df, x_feature, y_feature)`

**Purpose**: Create an interactive scatter plot between two features.

**Logic**:
1. Uses plotly.express to create a scatter plot
2. Adds a trend line using Ordinary Least Squares regression
3. Calculates the correlation coefficient between the two features
4. Adds an annotation showing the correlation value
5. Returns an interactive plotly figure

**Parameters**:
- `df`: DataFrame containing the data
- `x_feature`: Column name for the x-axis
- `y_feature`: Column name for the y-axis

**Returns**:
- Plotly figure object containing the scatter plot

**Built-in Functions Used**:
- **px.scatter()**: Creates a scatter plot
- **DataFrame.corr()**: Computes correlation between columns
- **fig.add_annotation()**: Adds an annotation to the plotly figure
- **fig.update_layout()**: Updates the layout of the plotly figure

## Usage Example

```python
# Import functions
from data_analysis import generate_descriptive_stats, create_correlation_heatmap, build_regression_model

# Load data (assuming spotify_df is loaded)

# Generate descriptive statistics for specific features
features = ['danceability', 'energy', 'loudness', 'tempo']
stats = generate_descriptive_stats(spotify_df, features)
print(stats)

# Create and display a correlation heatmap
correlation_fig = create_correlation_heatmap(spotify_df, features)
correlation_fig.show()

# Build a regression model to predict popularity
model, importance, mse, r2, summary = build_regression_model(
    spotify_df, 
    target='popularity', 
    features=['danceability', 'energy', 'loudness']
)

# Print model performance
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
print("Feature importance:")
print(importance)
```
