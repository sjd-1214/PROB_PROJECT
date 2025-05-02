import pandas as pd
import numpy as np
import os

def load_spotify_data(filepath='spotify_tracks.csv'):
    """
    Load the Spotify tracks dataset from a CSV file.
    
    Parameters:
    -----------
    filepath : str, default='spotify_tracks.csv'
        Path to the CSV file containing Spotify track data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the Spotify track data
    """
    try:
        # Try to find the file in the current directory first
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
        else:
            # If not found, try looking in the same directory as this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(script_dir, filepath)
            df = pd.read_csv(full_path)
        
        # Clean the data
        df = clean_data(df)
        return df
    
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def clean_data(df):
    """
    Clean and preprocess the Spotify tracks data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the Spotify track data
    
    Returns:
    --------
    pandas.DataFrame
        Cleaned and preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original dataframe
    df_clean = df.copy()
    
    # Remove unnamed columns if they exist
    unnamed_cols = [col for col in df_clean.columns if 'Unnamed' in col]
    if unnamed_cols:
        df_clean.drop(columns=unnamed_cols, inplace=True)
    
    # Handle missing values for numerical columns
    numerical_cols = get_numerical_features(df_clean)
    for col in numerical_cols:
        if df_clean[col].isna().sum() > 0:
            # Replace missing values with median
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Handle missing values for categorical columns
    categorical_cols = get_categorical_features(df_clean)
    for col in categorical_cols:
        if df_clean[col].isna().sum() > 0:
            # Replace missing values with mode (most frequent value)
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    return df_clean

def get_numerical_features(df):
    """
    Get the list of numerical features from the DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the Spotify track data
    
    Returns:
    --------
    list
        List of column names with numerical data types
    """
    # Get columns with numerical data types
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    return numerical_cols

def get_categorical_features(df):
    """
    Get the list of categorical features from the DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the Spotify track data
    
    Returns:
    --------
    list
        List of column names with categorical data types
    """
    # Get columns with categorical data types
    categorical_cols = df.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
    return categorical_cols

def get_audio_features(df):
    """
    Get the list of Spotify audio features from the DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the Spotify track data
    
    Returns:
    --------
    list
        List of Spotify audio feature column names
    """
    # List of standard Spotify audio features
    audio_features = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 
        'speechiness', 'acousticness', 'instrumentalness', 
        'liveness', 'valence', 'tempo', 'time_signature'
    ]
    
    # Only return features that exist in the DataFrame
    available_features = [feat for feat in audio_features if feat in df.columns]
    
    return available_features

def get_track_metadata(df):
    """
    Get the list of track metadata columns from the DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the Spotify track data
    
    Returns:
    --------
    list
        List of track metadata column names
    """
    # List of metadata columns that might be in the data
    metadata_cols = [
        'track_id', 'track_name', 'artists', 'album_name', 
        'popularity', 'explicit', 'track_genre', 'duration_ms'
    ]
    
    # Only return columns that exist in the DataFrame
    available_cols = [col for col in metadata_cols if col in df.columns]
    
    return available_cols
