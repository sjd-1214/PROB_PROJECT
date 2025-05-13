# Data Loader Module Explanation

## Overview

The `data_loader.py` module handles loading, cleaning, and preprocessing the Spotify track dataset for analysis. It provides functions to access various types of features from the dataset.

## Imported Libraries

- **pandas (pd)**: Used for data manipulation and analysis with DataFrame objects
- **numpy (np)**: Provides support for numerical operations and array handling
- **os**: Interfaces with the operating system for file path operations

## Function Explanations

### `load_spotify_data(filepath='spotify_tracks.csv')`

**Purpose**: Load the Spotify tracks dataset from a CSV file.

**Logic**:
1. First attempts to find the CSV file in the current working directory
2. If not found, looks in the same directory as the script
3. Calls `clean_data()` to preprocess the loaded data
4. Returns the cleaned DataFrame

**Error Handling**:
- Uses try-except to catch file loading errors
- Provides informative error messages

**Parameters**:
- `filepath`: Path to the CSV file (defaults to 'spotify_tracks.csv')

**Returns**:
- The cleaned Spotify tracks DataFrame

### `clean_data(df)`

**Purpose**: Clean and preprocess the Spotify tracks data.

**Logic**:
1. Creates a copy of the input DataFrame to avoid modifying the original
2. Removes any "Unnamed" columns that often appear when loading CSVs with index
3. Handles missing values in numerical columns by replacing with median values
4. Handles missing values in categorical columns by replacing with the most frequent value (mode)

**Parameters**:
- `df`: The raw DataFrame containing Spotify track data

**Returns**:
- A cleaned and preprocessed DataFrame

### `get_numerical_features(df)`

**Purpose**: Extract a list of numerical features from the DataFrame.

**Logic**:
- Uses pandas' `select_dtypes()` method to find columns with numeric data types ('int64', 'float64')
- Returns these column names as a list

**Parameters**:
- `df`: DataFrame containing Spotify track data

**Returns**:
- List of column names with numerical data types

### `get_categorical_features(df)`

**Purpose**: Extract a list of categorical features from the DataFrame.

**Logic**:
- Uses pandas' `select_dtypes()` method to find columns with categorical types ('object', 'bool', 'category')
- Returns these column names as a list

**Parameters**:
- `df`: DataFrame containing Spotify track data

**Returns**:
- List of column names with categorical data types

### `get_audio_features(df)`

**Purpose**: Extract a list of Spotify-specific audio features from the DataFrame.

**Logic**:
1. Defines a predefined list of standard Spotify audio features
2. Filters this list to only include features that actually exist in the input DataFrame
3. Returns the filtered list

**Parameters**:
- `df`: DataFrame containing Spotify track data

**Returns**:
- List of Spotify audio feature column names that exist in the DataFrame

### `get_track_metadata(df)`

**Purpose**: Extract a list of track metadata columns from the DataFrame.

**Logic**:
1. Defines a predefined list of possible metadata columns
2. Filters this list to only include columns that actually exist in the input DataFrame
3. Returns the filtered list

**Parameters**:
- `df`: DataFrame containing Spotify track data

**Returns**:
- List of track metadata column names that exist in the DataFrame

## Key Built-in Functions Used

- **pandas.read_csv()**: Reads a CSV file into a DataFrame
- **os.path.exists()**: Checks if a file exists
- **os.path.dirname()** and **os.path.abspath()**: Get the directory of the current file
- **os.path.join()**: Join path components
- **DataFrame.copy()**: Create a deep copy of the DataFrame
- **DataFrame.drop()**: Remove specified columns
- **DataFrame.isna().sum()**: Count missing values in each column
- **DataFrame.fillna()**: Replace missing values
- **DataFrame.median()**: Calculate median of columns
- **DataFrame.mode()**: Find the most frequent value
- **DataFrame.select_dtypes()**: Select columns with specific data types
- **list comprehension**: Used throughout for filtering lists

## Usage Example

```python
# Import the module
from data_loader import load_spotify_data, get_numerical_features

# Load the dataset
spotify_df = load_spotify_data()

# Get numerical features for analysis
numerical_cols = get_numerical_features(spotify_df)
print(f"Numerical columns: {numerical_cols}")

# Examine the first few rows
print(spotify_df.head())
```
