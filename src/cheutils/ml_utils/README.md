# ml_utils

A set of reusable utilities for machine learning.

### Features

- model_options: methods for handling estimator supported estimators.
- model_builder: methods for handling model development steps
- pipeline_details: method for exposes details of model ML pipeline, etc.
- visualize: general utilities for visualizations
- bayesian_search: wrappers classes for Hyperopt optimization, including support for cross validation

### Usage

```
import cheutils

# Train the model
ml_utils.model_builder.fit(estimator, X_train, y_train)

```
