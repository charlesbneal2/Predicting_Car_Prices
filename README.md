# Car Price Predictor: Project Overview
- Created a tool that estimates market prices (RMSE ~$2600) for cars based on technical specs to help people negotiate great deals on cars
- Built the tool on 205 car listings from the 1985 Ward's Automotive Yearbook hosted on the UCI Machine Learning repository
- Implemented a kNN Regressor and tuned hyperparameters to find the optimal model

## Code and Resources Used
**Python Version:** 3.7

**Packages:** pandas, numpy, sklearn, matplotlib

**Dataset Documentation:** https://archive.ics.uci.edu/ml/datasets/automobile

## Data Overview
For each car listing, we get 25 columns detailing specifications of the vehicle and one column listing the price. For our regression model, I focused on continuous, numeric variables as predictors:
- normalized-losses
- wheel-base
- length
- width
- height
- curb-weight
- engine-size
- bore
- stroke
- compression-rate
- horsepower
- peak-rpm
- city-mpg
- highway-mpg

## Data Cleaning
After importing the package, I needed to clean it so that it would be suitable for our model. I made the following changes:
- Replaced "?" in the normalized-losses column with NaN
- Filled NaN values in each column with the mean value of that column
- Normalized predictors on a scale of 0 to 1 to avoid biasing the model

## Model Building
### Feature Selection
To evaluate the impact of each feature individually, I first implemented a univariate kNN Regressor using the default of 5 nearest neighbors. I used 50% holdout validation to test the model. I computed the root mean squared error for the predictions made with each feature and compared them to select the most promising features. I then varied the number of nearest neighbors of each model to examine how k impacted performance.

![image](https://user-images.githubusercontent.com/97380323/172740229-ae6f6095-652c-4c45-beb6-7962441010ac.png)

### Multivariate Model and Hyperparameter Optimization
I selected the top 5 features by lowest RMSE. To examine the impact of number of features on model performance, I incrementally added each feature to the model. To optimize our hyperparameter (k), I incremented k from 1 to 25 for each model. Comparing the RMSE value for each number of features and number of nearest neighbors allowed me to find the optimal model.

![image](https://user-images.githubusercontent.com/97380323/172740648-c234a53c-b7ac-4522-88c1-701b3779e06d.png)


The best performing model used 4 features (engine-size, horsepower, curb-weight, highway-mpg, width) and 2 nearest-neighbors, producing a RMSE of $2600.
