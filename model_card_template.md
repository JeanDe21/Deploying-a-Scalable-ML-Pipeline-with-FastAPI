# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This project uses a Logistic Regression classification model to predict whether an individual's annual income is greater than $50,000 based on demographic and employment-related census information. The model uses numerical features and categorical features that are processed using one-hot encoding. The categorical target variable is converted into a binary label using a label binarizer. The trained model and encoder are saved as pickle files for later inference through the FastAPI application.

## Intended Use

The intended use of this model is to demonstrate a machine learning development and deployment pipeline using publicly available census data. The model predicts one of two income categories: `>50K` or `<=50K`. It is intended for educational and demonstration purposes, including model training, evaluation, data-slice analysis, and API-based inference. It should not be used as the sole basis for making decisions about an individual's employment, financial opportunities, credit, benefits, or other high-impact outcomes.

## Training Data

The model was trained using the Census Income dataset, also commonly referred to as the Adult dataset. The dataset contains 48,842 instances and 14 features and was derived from census information collected from the 1994 U.S. Census database. The prediction task is to determine whether an individual's annual income exceeds $50,000. The dataset contains both categorical and integer features, including age, workclass, education, marital status, occupation, relationship, race, sex, hours worked per week, and native country. The dataset also contains missing values in some features.

For this project, the data was divided into training and test datasets. The categorical variables were processed using one-hot encoding, while the target variable was converted into a binary representation before model training.

## Evaluation Data

The model was evaluated using the test portion of the Census Income dataset that was separated from the training data. The same preprocessing approach used during training was applied to the evaluation data using the trained encoder. Model performance was also evaluated across categorical slices, including different workclass and education values, to examine whether performance varied across subsets of the data.

## Metrics

The model was evaluated using precision, recall, and F1 score.

The model achieved the following results on the test dataset:

- Precision: 0.7376
- Recall: 0.6066
- F1 Score: 0.6657

Precision measures the proportion of positive predictions that were correct. Recall measures the proportion of actual positive cases that the model correctly identified. The F1 score combines precision and recall into a single metric using their harmonic mean.

Additional slice-level evaluation was performed across categorical features. For example, performance was calculated separately for each workclass and education category. These slice results are stored in `slice_output.txt` and are intended to identify differences in model performance across subsets of the evaluation data.

## Ethical Considerations

The dataset contains demographic and socioeconomic attributes, including race, sex, relationship status, education, workclass, and native country. Because these characteristics can be associated with sensitive demographic information, predictions made using this dataset may reflect patterns or biases present in the historical data.

The model should therefore not be interpreted as determining an individual's actual financial circumstances or worth. The model is intended for educational purposes and should not be used as an automated decision-making system for employment, lending, insurance, government benefits, or other high-impact decisions.

Model performance should also be reviewed across relevant data slices before considering any real-world deployment. Differences in precision, recall, or F1 score between groups may indicate that additional analysis, data preparation, or model evaluation is necessary.

## Caveats and Recommendations

The model is based on historical census data and therefore may not accurately represent current economic conditions or modern populations. The dataset was originally derived from 1994 census information, so changes in employment patterns, income distributions, education, and demographics may reduce the model's relevance to present-day predictions.

The model uses Logistic Regression and was developed primarily as part of an educational machine learning deployment pipeline. Its performance should not be interpreted as sufficient evidence for production use. Additional models, hyperparameter tuning, feature engineering, cross-validation, calibration, and fairness analysis could be considered to improve the evaluation.

The slice-level results should be reviewed before using the model in any application involving different demographic or socioeconomic groups. Predictions should be treated as model estimates rather than definitive statements about an individual's income.