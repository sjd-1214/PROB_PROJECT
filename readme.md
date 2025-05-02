# Spotify Track Analysis App

An interactive web application for analyzing Spotify track data, built with Streamlit and Python.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

This Spotify Track Analysis App performs statistical analysis on Spotify track data, including descriptive statistics, probability distributions, correlation analysis, and regression modeling. The app is built using Streamlit for the web interface, with data analysis performed using Pandas, NumPy, and other popular Python libraries.

## Features

- **Data Overview**: Explore the dataset, view feature descriptions, and analyze missing values
- **Descriptive Statistics**: Calculate and visualize statistical measures 
- **Feature Distributions**: Analyze probability distributions with histograms, QQ plots, and more
- **Correlation Analysis**: Visualize relationships between features with heatmaps and scatter plots
- **Regression Modeling**: Build and evaluate linear regression models to predict track features
- **Interactive Visualization**: Dynamically explore the data with adjustable parameters

## Requirements

- Python 3.7+ 
- pip (Python package installer)
- Git (for cloning the repository)

## Installation

Follow these steps to set up and run the Spotify Track Analysis App on your local machine:

### 1. Clone the Repository

Open a terminal or command prompt and run the following command to clone the repository:

```bash
git clone https://github.com/yourusername/spotify-track-analysis.git
cd spotify-track-analysis
```

If you've received the project as a zip file instead, extract it to your desired location and navigate to the folder:

```bash
cd path/to/extracted/spotify-track-analysis
```

### 2. Create a Virtual Environment (Recommended)

Creating a virtual environment isolates the project dependencies from your system Python:

#### For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

Your command prompt should now show `(venv)` at the beginning, indicating the virtual environment is active.

### 3. Install Dependencies

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

If the `requirements.txt` file is not available, you can install the necessary packages directly:

```bash
pip install streamlit pandas numpy matplotlib seaborn plotly scipy scikit-learn
```

### 4. Prepare the Dataset

Make sure your dataset file `spotify_tracks.csv` is in the correct location. By default, the app looks for this file in the `data` directory:

```bash
mkdir -p data
# Place your spotify_tracks.csv file in the data directory
```

If you need to download the dataset, you can use the following commands:

```bash
# If you have curl installed
curl -o data/spotify_tracks.csv https://path-to-dataset/spotify_tracks.csv

# If you have wget installed
wget -O data/spotify_tracks.csv https://path-to-dataset/spotify_tracks.csv
```

## Running the Application

Once everything is set up, you can run the application using Streamlit:

```bash
streamlit run app.py
```

This command will start the Streamlit development server and provide you with:
- A local URL (usually http://localhost:8501)
- A network URL that allows other devices on your network to access the app

The app should automatically open in your default web browser. If not, you can manually navigate to the URL provided in the terminal.

### Command Line Arguments

You can provide additional command-line arguments to Streamlit:

```bash
streamlit run app.py --server.port 8502 --server.headless true
```

This example starts the app on port 8502 and in headless mode (without automatically opening the browser).

## Project Structure

```
spotify-track-analysis/
├── app.py              # Main application file
├── data/               # Directory containing datasets
│   └── spotify_tracks.csv  # Spotify tracks dataset
├── data_loader.py      # Functions for loading and preprocessing data
├── data_analysis.py    # Functions for statistical analysis and visualization
├── requirements.txt    # List of Python dependencies
└── README.md           # Project documentation
```

## Troubleshooting

### Common Issues and Solutions

1. **Missing Module Error**:
   ```
   ModuleNotFoundError: No module named 'streamlit'
   ```
   **Solution**: Ensure you've activated the virtual environment and installed dependencies:
   ```bash
   pip install streamlit
   ```

2. **File Not Found Error**:
   ```
   FileNotFoundError: [Errno 2] No such file or directory: 'data/spotify_tracks.csv'
   ```
   **Solution**: Make sure the dataset file is in the right location. Create a `data` directory and place your `spotify_tracks.csv` file there.

3. **Memory Error**:
   ```
   MemoryError: ...
   ```
   **Solution**: The dataset might be too large for your system's RAM. Try reducing the sample size in the app settings.

4. **Port Already in Use**:
   ```
   Address already in use: [::]:8501
   ```
   **Solution**: Change the port number when running Streamlit:
   ```bash
   streamlit run app.py --server.port 8502
   ```

5. **Visualization Not Showing**:
   **Solution**: Try refreshing the page or clearing your browser cache.

### Getting Help

If you encounter issues not listed here:
1. Check Streamlit's [documentation](https://docs.streamlit.io) and [community forum](https://discuss.streamlit.io)
2. Look for error messages in the terminal where Streamlit is running
3. Open an issue on the project's repository

## Contributing

Contributions to improve the Spotify Track Analysis App are welcome:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature-name`)
6. Open a Pull Request

---

Made with ❤️ using Streamlit and Python
