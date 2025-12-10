# streamlit-plotly-csv-dashboard
Streamlit Data Cleaning & Exploration App
1. Overview

Modern analytics teams spend 60%–80% of their time cleaning data before any analysis or modeling can begin.
Missing values, inconsistent formatting, and unfiltered datasets make it difficult for students, analysts, and organizations to work efficiently.

This project solves that problem by providing an interactive, no-code data cleaning application built with Streamlit and Python.
Users can upload a CSV file, inspect issues, clean the data, visualize patterns, and download the cleaned output instantly.
2. Problem Statement

Organizations and students often deal with messy datasets that contain:

Missing values (NaN)

Duplicates

Incorrect datatypes

Outliers

Poor visibility into the structure of the dataset

Traditional methods require:

Writing manual Python scripts

Knowledge of pandas and matplotlib

Iterative debugging
Manual export and saving

This creates a slow, repetitive workflow that blocks productivity.


3. Solution

I built a Streamlit-based automated data cleaning pipeline that solves these inefficiencies by providing:

Automatic data loading & validation

Null-value detection and filling

Duplicate detection and removal

Summary statistics generation

Interactive filtering for numerical & categorical columns

Data visualization (histograms, bar charts, line graphs)

Download buttons for cleaned dataset & summary report

This tool transforms raw messy data into clean, usable, analysis-ready datasets in seconds.
4. Key Features
✔ Upload CSV File

Upload any CSV file to begin instant processing.

✔ Data Quality Overview

View:

Shape (rows, columns)

Columns list

Missing values

Duplicate rows
✔ Data Cleaning Tools

Fill nulls (mean, median, mode)

Drop nulls

Convert datatypes

Remove duplicates

✔ Interactive Filters

Numeric range sliders

Categorical dropdown filters
✔ Visualizations

Automatically generate:

Histogram

Line plot

Bar chart
✔ Downloads

Export:

Cleaned CSV

Summary statistics (CSV)
5. Tech Stack

Python 3.x

Streamlit

Pandas

Matplotlib / Seaborn

Plotly (depending on final version)

NumPy
6. How It Works (Architecture)

User uploads a CSV file
Streamlit reads it into a pandas DataFrame.

Data profiling
System computes:

Head

Shape

Missing percentage

Duplicate count

Summary statistics
Cleaning operations
User selects:

Fill null method

Remove duplicates

Apply filters

Update visualizations

Visualization layer
Charts update based on the cleaned DataFrame.

Export
User downloads cleaned dataset and statistics.
7. Future Enhancements

Outlier detection

Data type conversion wizard

Automatic EDA report (PDF)

Correlation heatmaps

AI-assisted cleaning suggestions
