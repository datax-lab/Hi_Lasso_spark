# Hi-LASSO_spark
Hi-LASSO_Spark(High-Demensinal LASSO Spark) is to improve the LASSO solutions for extremely high-dimensional data using pyspark. 
PySpark is the Python API written in python to support Apache Spark. 
Apache Spark is a distributed framework that can handle Big Data analysis. 
Spark is basically a computational engine, that works with huge sets of data by processing them in parallel and batch systems.

## Installation
**Hi-LASSO_Spark** support Python 3.6+, Additionally, you will need ``numpy``, ``scipy``, and ``glmnet``.

``Hi-LASSO_spark`` is available through PyPI and can easily be installed with a
pip install::
```
pip install hi_lasso_spark
```

## Documentation
Read the documentation on [readthedocs](https://hi-lasso-spark.readthedocs.io/en/latest/index.html)

## Quick Start
```python
# Data load
import pandas as pd
X = pd.read_csv('simulation_data_x.csv')
y = pd.read_csv('simulation_data_y.csv')

# General Usage
from hi_lasso_spark.Hi_LASSO_spark import HiLASSO_Spark

# Create a HiLasso model
model = HiLASSO_Spark(X, y, alpha=0.05, q1='auto', q2='auto', L=30, cv=5, node=40, logistic=False)

# Fit the model
model.fit(significance_level = 0.05)

# Show the coefficients
model.coef_

# Show the p-values
model.p_values_

# Show the selected variable
model.selected_var_
```
