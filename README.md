# Predict Next-Day Rain in Australia with Python

## Motivation
Expect rain. Only two simple words, yet the stakes sometimes could be much higher than grabbing an umbrella before leaving the house tomorrow. Rain could ruin picnic plans or bring tremendous joy to farmers who are desperate to save their drought-stricken crops. 

Although rain forecast is so ubiquitous nowadays that it is easy to take it for granted, learning how to predict next-day rain is a simple and practical way to explore Machine Learning concepts with Python. That is also the motivation for this personal project: Predict next-day rain in Australia 

## Installation
The code requires Python versions of 3.* and general libraries specified in the file requirements.txt

## File Description
The following files are included in this repository. 
1. EDA_InitialModelvF: A Jupyter notebook containing the code for exploratory data analysis and initial experiments with several ML models 
2. RF_Optimise: A Jupyter notebook containing the code for finalised approach of data preprocessing and hyperparameters optimisation for Random Forest Classifier
3. randomforestmodel.sav: The best Random Forest model after fine-tuning - serialised with pickle operation to be used for making new predictions

## Results
Among all ML models experimented (e.g. SGD Classifier, Logistic Regression, Random Forest Classifier, AdaBoost Classifier and Gradient Tree Boost Classifier), Random Forest Classifier has the best performance across accuracy, precision, recall and F1 score (0.90 or above) prior to fine-tuning. Such performance is acceptable for a general rain forecast solution. 

After the first round of fine-tuning with RandomizedSearchCV and GridSearchCV, the improvement in performance is ... Although I can continue trying different combinations of hyperparameters to improve the model performance, I think I have reached the point of diminishing returns for hyperparameter tuning. In other words, performance improvement will not worth the time and effort spent on fine-tuning. As a result, I decided not to further fine-tuning the models. 

The best Random Forest model is serialised and saved as the filed randomforestmodel.sav, which could later be deserialised to make new predictions with unseen data. Instructions and code on how to do so can be found in the RF_Optimise Jupyter notebook, section 7. 

## Acknowledgements
1. The main dataset was obtained from from [Kaggle](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package), which comes from Australia's Bureau of Meteorology.
2. In the RF_Optimise notebook, I have also adapted the code and the approach from [this article](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74) written by Will Koehrsen. 
