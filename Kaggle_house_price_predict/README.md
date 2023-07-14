# House Price Predict

## Result
competition from :https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
my submission result: top 3% ranking


## Project Introducion

Welcome to my data science portfolio showcasing my House Prices Kaggle competition project. In this project, I aimed to predict house prices based on various features provided in the dataset. Let's dive into the details and see how I approached this challenging problem


## Problem Statement

The task at hand was to develop a regression model that accurately predicts house prices. The ability to estimate house prices based on relevant features is crucial in the real estate industry, aiding in decision-making for buyers, sellers, and investors.

The main difficulty is :

- Feature engineering is important and difficult in the project due to the 80+ features.
- Regression method choose is essential due to the high dimension data.

## Data Understanding

The dataset provided comprehensive information about houses, including features like square footage, number of rooms, location, and more. After thoroughly examining the dataset, I identified 79 features that could potentially impact house prices. I also noticed some missing values and outliers, which required careful handling during preprocessing.

## Data Preprocessing and Feature Engineering

Here is my preprocessing procedure in order:

| Steps | Description | Example |
|-------|-------------|---------|
| Analyze each feature | Examine the field description, data type, ordinal categories, value distribution, missing value percentage, and presence of outliers for both the training and testing datasets | - Review the feature descriptions, data types, and distributions for fields in the training and testing datasets<br>- Calculate the percentage of missing values and check for outliers |
| Remove features with missing values or severe skewness | Identify features where a value of 0 represents a missing value or NA | - Remove features that have a high proportion of missing values<br>- Exclude features that exhibit severe skewness with a value of 0 representing a missing value |
| Analyze the relationship between features and the target variable | Determine the correlation between each feature and the target variable | - Calculate the correlation coefficient between each feature and the target variable<br>- Assess the strength and direction of the relationships |
| Remove irrelevant features | Eliminate features that show little or no correlation with the target variable | - Drop features with a low correlation coefficient or insignificant p-values<br>- Exclude variables that do not contribute significantly to the prediction of the target variable |
| Fill in missing values | Handle missing values based on data type for both the training and testing datasets | - For continuous numerical features, replace missing values with the mean<br>- For categorical features with skewed distributions, replace missing values with the most frequent category |
| Handle outlier data | Clean training and testing datasets by addressing obvious data errors and outliers | - Correct data errors, such as changing the year 2207 to 2007<br>- Treat area outliers greater than 40,000 as input errors and replace them with 40,000 |
| Feature combination and cleaning | Combine year-related features to create more meaningful features | - Create a new feature by subtracting the construction year from the sale year to represent the age of the house at the time of sale |
| Test dataset replacement | Replace categorical data that appears only in the test dataset | - Replace any categorical data in the test dataset that does not exist in the training dataset |
| Feature type conversion | Convert categorical features into ordinal features if there is an inherent order | - Convert categorical features with ordered categories (e.g., good, moderate, bad) to ordinal features represented by numerical values (e.g., 2, 1, 0) |
| One-hot encoding | Apply one-hot encoding to non-ordinal categorical features | - Convert non-ordinal categorical features, such as wall material (wood, brick), into binary columns representing each category |
| Check feature skewness | Apply logarithmic transformation to skewed features | - Log-transform features that exhibit significant skewness in their distributions |



