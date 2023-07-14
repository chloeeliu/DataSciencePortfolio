# House Price Predict

competition from :https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

my submission result: top 3% ranking

-- todo
贴图片，并且修改下最后train model部分


## Acknowledgements

This project is inspired by the work of [Author's Name] and their project [Project Name]. I would like to express my gratitude for their insightful contributions, which served as a valuable reference and inspiration for this project. While I have conducted additional feature engineering and outlier review on my own, their initial work provided a solid foundation for my study and analysis.

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

## Exploratory Data Analysis

Through exploratory data analysis, I gained valuable insights into the dataset. Visualizations such as scatter plots, correlation matrices, and box plots revealed interesting relationships between features and house prices. For example, I observed a strong positive correlation between the number of rooms and the sale price. These insights guided my feature selection and modeling decisions.

-- 贴个图片，一个相关矩阵，一个单独特征的。



## Modeling Approach

To tackle the regression task, I explored several algorithms, including linear regression, decision trees, and ensemble methods like random forests and gradient boosting. I selected these algorithms based on their ability to handle both numerical and categorical features. I also split the data into a training set and a separate validation set to assess model performance effectively.

I developed multiple models using different algorithms and evaluated their performance on the validation set. I implemented cross-validation techniques to fine-tune hyperparameters and mitigate overfitting. To ensure model reproducibility, I wrapped the entire modeling pipeline into modular functions and classes, allowing for easy experimentation and future improvements


| Regression Algorithm       | Algorithm Type | Algorithm Pros and Cons                                                                                                                                                                      |
|---------------------------|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ElasticNet                | Linear         | Pros: <br>- Combines L1 and L2 regularization<br>- Suitable for high-dimensional data<br>- Controls for overfitting<br><br> Cons: <br>- May struggle with highly correlated features          |
| Ridge                     | Linear         | Pros: <br>- Handles multicollinearity<br>- Stable and robust<br>- Less sensitive to outliers<br><br> Cons: <br>- Assumes linear relationship between features and target                    |
| Lasso                     | Linear         | Pros: <br>- Performs feature selection<br>- Handles multicollinearity<br><br> Cons: <br>- Selects only one feature among correlated features                                                        |
| KernelRidge               | Non-linear     | Pros: <br>- Handles non-linear relationships<br>- Incorporates kernel trick to model complex patterns<br><br> Cons: <br>- Can be computationally expensive for large datasets               |
| LGBMRegressor             | Non-linear     | Pros: <br>- High performance and efficiency<br>- Handles large datasets<br>- Supports categorical features without one-hot encoding<br><br> Cons: <br>- May overfit with small datasets        |
| XGBRegressor              | Non-linear     | Pros: <br>- Handles complex relationships<br>- Handles missing values<br>- Regularization to control overfitting<br><br> Cons: <br>- Requires careful tuning of hyperparameters                  |
| CatBoostRegressor         | Non-linear     | Pros: <br>- Handles categorical features without explicit encoding<br>- Handles missing values<br>- Robust to outliers<br><br> Cons: <br>- Can be computationally expensive for large datasets |
| GradientBoostingRegressor | Non-linear     | Pros: <br>- Handles complex relationships<br>- Robust to outliers<br>- Handles missing values<br><br> Cons: <br>- Requires careful tuning of hyperparameters                                  |


## Model Evaluation and Validation

After training and validating the models, I assessed their performance using evaluation metrics such as Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). These metrics demonstrated the model's efficacy in capturing the underlying patterns in the data.

-- 贴表格的图


## Results and Conclusion

In conclusion, the developed regression model successfully predicted house prices with a reasonable level of accuracy. The insights gained from the analysis and feature engineering techniques showcased the importance of various features in influencing house prices. However, it is essential to note that further improvements could be made, such as incorporating more advanced algorithms or exploring additional external datasets to enhance predictive performance.


