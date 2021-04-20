# Predict Next-Day Rain in Australia with Python

## Executive Summary 
This project aims to build an ML model from scratch with Python to predict next-day rain in Australia. The accuracy and the time taken to build such a model will then be compared with that when creating a model with Google BigQuery ML. In doing so, I hope to answer one question: Is it still worth it to build an ML model froms scratch with Python still justifiable if training an ML model with Google BigQuery is so fast and easy?

The result showed that the Random Forest model with Python scikit-learn has an accuracy of 0.91, which is better than the 0.85 accuracy coming from the logistic regression model with BigQuery ML. Nevertheless, it took me about 6 days of data cleansing, seriously musing and waiting for the code to run (as well as tackling each red error I encountered along the way). In contrast, an ML model using BigQuery took me less than a day from start to finish. 

But will I opt for BigQuery ML from now onwards since it is so much faster and easier? Not really because the choice to favour speed over accuracy, or vice versa, has to be driven by the nature of the problem and the business context. Personally, I felt the longer time taken to try different ways to make things work has enabled me to understand more about the problem, the dataset and the implications of those choices or trade-offs I have made along the way. 

Is it possible to balance between speed and accuracy? Yes, here is what I would propose for real-life applications. 
1. Start with EDA: Obtain a solid background about the problem and the dataset before training any ML model. There is no point in wasting time on the wrong problem and/ or the wrong dataset. (Note: It does not matter whether I use Python, stick to BigQuery with Google Data Studio or other EDA tools.)
2. Create a baseline model with BigQuery ML: Quickly show it to everyone to gather feedback first and see whether it is good enough for production. Way better than spending a few months working on a problem while business has moved on to other things.
3. Explore further ML with Python/ Tensorflow/ Keras: Good for further data wrangling, feature engineering and hyperparameter optimisation if further accuracy is needed and time permits.

## Motivation
Here is what I initially want to achieve: Given today’s observations about Wind Direction, Rainfall, Minimum Temperature, Maximum Temperature, Cloud Cover and so on, can I predict whether it will rain tomorrow? 

Using Google BigQuery ML, I managed to create a logistic regression model to predict next-day rain with an accuracy of 0.85 within less than 30 minutes. You may find the blog post [here](http://thedigitalskye.com/2021/04/13/how-to-train-a-classification-model-to-predict-next-day-rain-with-google-bigquery-ml/) and the code in [this repo](https://github.com/Skye-Tran/BigQueryML-AUSRainPrediction).

However, such experiment triggered another question: Is it still worth it to build an ML model froms scratch with Python still justifiable if training an ML model with Google BigQuery is so fast and easy? In other words, I want to see whether I can build a more accurate ML model to predict next-day rain with Python. If yes, how much more time do I have to spend? That is also the motivation for this repository. 

## Installation
The code requires Python versions of 3.* and general libraries specified in the file requirements.txt. You will also have to:
- Download the weatherAUS.csv from [Kaggle](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)
- Update the "weatherAUS_path" with your Google Cloud Storage URI (if you choose to run everything on GCP AI Platform) or to the dataset stored on your local machine before loading the weather observation data

It is strongly recommened to run the second notebook named 2_RandomForest on GCP AI Platform to optimise for speed. To do that, you will have to: 
- Create a new [notebook instance](https://cloud.google.com/ai-platform/notebooks/docs/create-new) on GCP AI Platform 
- [Clone this repository](https://cloud.google.com/ai-platform/notebooks/docs/save-to-github) in your notebook instance
- Download the weatherAUS.csv from [Kaggle](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package) and upload it to Google Cloud Storage
- Update the "weatherAUS_path" with your Google Cloud Storage URI before loading the weather observation data

## File Description
The following files are included in this repository. 
1. EDA_InitialModelvF: A Jupyter notebook containing the code for exploratory data analysis and initial experiments with several ML models 
2. RandomForest: A Jupyter notebook containing the code for data preprocessing and hyperparameters optimisation for Random Forest Classifier

## Results
Among all ML models experimented (e.g. SGD Classifier, Logistic Regression, Random Forest Classifier, AdaBoost Classifier and Gradient Tree Boost Classifier), Random Forest Classifier has the best performance across accuracy, precision, recall and F1 score (0.90 or above) prior to fine-tuning. Such performance is acceptable for a general rain forecast solution. 

After the first round of fine-tuning with RandomizedSearchCV and GridSearchCV, the improvement in performance is <TBC> Although I can continue trying different combinations of hyperparameters to improve the model performance, I think I have reached the point of diminishing returns for hyperparameter tuning. In other words, performance improvement will not worth the time and effort spent on fine-tuning. As a result, I decided not to further fine-tuning the models. 

The best Random Forest model is serialised and saved as the filed randomforestmodel.sav, which could later be deserialised to make new predictions with unseen data. Instructions and code on how to do so can be found in [this article](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/).
    
## Lessons Learned
Looking back at this ML project, below are some crucial lessons I have picked up along the way about this binary classification problem. Do take it with a pinch of salt because the world of Machine Learning and Data Science is so diverse and nuanced. 

1. Exploratory Data Analysis (EDA)
    - Starting with EDA is a must to obtain a solid background about the problem and the dataset. There is no point in wasting time on the wrong problem and/ or the wrong dataset.  
    - EDA should be done with a clear purpose and structure: 
        - Load data and basic exploration (data types, descriptive statistics, null values): Does the dataset fit for purpose? Does it need to be preprocessed? 
        - Univariate analysis: It's all about data characteristics and data quality
            - Categorical features: missing values, invalid & inconsistent values to be fixed
            - Numerical features: skewness, outliers, missing values, invalid values (based on max and min)
        - Bivariate analysis and multi-variate analysis: It's all about the predictive power. Which features should be retained, discarded or KIV for further feature engineering? 
        - Key takeaways and next steps: Should we proceed with the given task and dataset? If yes, how do we proceed with the task? 


2. Model Selection for Binary Classification Problem
    - Remember to look beyond accuracy as the only evaluation metrics for binary classification model. Other alternatives include precision, recall, F1 score and ROC AUC. Select a metrics for evaluation model based on business context (Is it more important to minimise FP or reducing FN?)
    - Always glance through the confusion matrix 
    - Simple models are so much faster to train but might result in lower accuracy. Vice versa, sophisticated ensemble methods are slow to train but the prediction performance could be superior. Therefore, it is always a matter of trying out and deciding which factor is more important (speed vs accuracy, simplicity vs sophistication). 


3. Hyperparameter Fine-tuning
    - It is all about controlled experiment and knowing when to stop (because you can go on forever with fine-tuning and end up with an overfitting model). 
    - Define a threshold for performance improvement. If the fine-tuning stops exceeding the threshold, then halt the fine-tuning because you are probably hitting a point of diminishing return. This is very similar to the concept of early stopping in BigQuery ML. 
    - Start with the RandomizedSearchCV to narrow down a reasonable range of hyperparameter values. Then use GridSearchCV to identify the best performing model. 
    - As it simply took too long for my local machine to iterate through different combinations of hyperparameter for RandomizedSearchCV and GridSearchCV, there are 2 options I have used to rescue yourself: 
        - Migrate the workload to cloud platform (e.g. AWS, GCP, Azure). Of course it will cost some money if you need it frequently, but definitely faster and more scalable than running it on a bigger machine. For a one-off project, free trials from cloud platforms could be more than sufficient. In my case, Jupyter Notebooks on Google AI Platform absolutely rocks (fast to provision, Jupyter Lab and standard Python libraries readily installed + direct linkage to GitHub).
        - Reduce the volume of the data used for training. For example, randomly sampling 25% and use it to train the model. But again, this is a deliberate trade-off for speed. 
        

4. Evaluate the model with test data
    - After spending all the time and efforts to select and fine-tune an ML model, evaluating with test data is definitely a must to check for underfitting or overfitting. 
    - Stick to the same evaluation metrics. It does not make sense if you earlier choose an "out-of-the-box" model for the highest accuracy, then evaluating the same model with test data using F1 score.

## Recommended Future Improvements
Here are a few points that I wish I had done and would love to revisit when time permits. 
1. Experimenting with further feature engineering
2. Leverage more functions to streamline my coding and make it easier to understand
3. Experiment with fine-tuning Support Vector Machine model
4. Build an end-to-end ML pipeline with Apache Beam or Airflow to productionise the ML model and build a rain forecast app with weather observation data streamed from BOM website or other sources
    
## Acknowledgements
1. The main dataset was obtained from from [Kaggle](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package), which comes from Australia's Bureau of Meteorology.
2. In the RF_Optimise notebook, I have also adapted the code and the approach from [this article](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74) written by Will Koehrsen.
3. Rain image: Photo by <a href="https://unsplash.com/@ak1?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Anna Atkins</a> on <a href="https://unsplash.com/s/photos/rain?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
