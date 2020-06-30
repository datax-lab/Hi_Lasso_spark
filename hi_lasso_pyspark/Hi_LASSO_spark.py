# Author: Seungha Jeong <jsh29368602@gmail.com>
# License: MIT
# Date: 30, Jun 2020

# Import package
import numpy as np
import math
from scipy.stats import binom, norm
import glmnet
import os
import binascii

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

"""
    • SparkConf() : Configuration for a Spark application. Used to set various Spark parameters as key-value pairs.
                    Most of the time, you would create a SparkConf object with new SparkConf(), which will load values from any spark.
                    For example, you can write new SparkConf().setMaster("local").setAppName("My app").

    • getOrCreate() : This function may be used to get or instantiate a SparkContext and register it as a singleton object.

"""
conf = SparkConf().setMaster("local[*]").setAppName("PySparkShell")
sc = SparkContext.getOrCreate(conf)
sqlContext = SQLContext(sc)


class HiLASSO_Spark:
    
    """
    Hi-LASSO_Spark(High-Demensinal LASSO Spark) is to improve the LASSO solutions for extremely high-dimensional data using pyspark. 
    
    PySpark is the Python API written in python to support Apache Spark. 
    Apache Spark is a distributed framework that can handle Big Data analysis. 
    Spark is basically a computational engine, that works with huge sets of data by processing them in parallel and batch systems.
    
    • RDD: PySpark basically helps data scientists to easily work with Resilient Distributed Datasets.
    • Speed: This framework is known for its greater speed compared with the other traditional data processing frameworks.
    • Caching and Disk persistence: This has a powerful caching and disk persistence mechanism for datasets that make it incredibly faster and better than others.
    • Resilient Distributed Datasets – these are basically datasets that are fault-tolerant and distributed in nature. 
    There are two types of data operations:  Transformations and Actions. Transformations are the operations that work on input data set and apply a set of transform method on them. 
    And Actions are applied by direction PySpark to work upon them.
    
    
    The main contributions of Hi-LASSO are as following:
    
    • Rectifying systematic bias introduced by bootstrapping.
    • Refining the computation for importance scores.
    • Providing a statistical strategy to determine the number of bootstrapping.
    • Taking advantage of global oracle property.
    • Allowing tests of significance for feature selection with appropriate distribution.
    
    
    Parameters
    ----------        
    X: array-like of shape (n_sample, n_feature)
       predictor variables    
    y: array-like of shape (n_sample,)
       response variables    
    q1: 'auto' or int, optional [default = 'auto']
        The number of predictors to randomly selecting in Procedure 1.
        When to set 'auto', use q1 as number of samples.
    q2: 'auto' or int, optional [default = 'auto']
        The number of predictors to randomly selecting in Procedure 2.
        When to set 'auto', use q2 as number of samples.        
    alpha: float [default=0.95]
        confidence level for determination of bootstrap sample size.
    d: float [default=0.05]
        sampling error for determination of bootsrap sample size.
    B: 'auto' or int, optional [default='auto']
        The number of bootstrap samples.
        When to set 'auto', B is determined by statistical strategy(using alpha and d). 
        
    
    Attributes
    ----------    
    n : int
        number of samples.
    p : int
        number of features.
        
    Examples
    --------
    >>> from Hi_LASSO_spark_fin import HiLASSO_Spark
    >>> model = HiLASSO_Spark(X, y)
    >>> model.fit()
    
    >>> model.coef_
    >>> model.p_values_
    >>> model.selected_var_ 
    """
    
    def __init__(self, X, y, alpha = 0.95, q1 = 'auto', q2 = 'auto', B = 'auto', d = 0.05, cv = 5):
        
        self.X = np.array(X)
        self.y = np.array(y).flatten()
        self.n_sample, self.n_feature = X.shape
        self.q1 = self.X.shape[0] if q1 == 'auto' else q1
        self.q2 = self.X.shape[0] if q1 == 'auto' else q2
        self.d = d
        self.alpha = alpha
        self.cv = cv
        self.B = math.floor(norm.ppf(self.alpha, loc = 0, scale = 1) ** 2 * self.q1 / self.n_feature * (1 - self.q1 / self.n_feature) / self.d ** 2) if B == 'auto' else B
             
        
   
    def standardization(self, X, y):
        
        """
        The response is mean-corrected and the predictors are standardized

        Parameters
        ---------
        X: array-like of shape (n_sample, n_feature)
           predictor              
        y: array-like of shape (n_sample,)
           response

        Returns
        -------
        np.ndarray
        scaled_X, scaled_y, std
        """
        
        mean_x = X - X.mean()
        std = np.sqrt((mean_x**2).sum(axis=0)) 
        X_sc = mean_x / std
        y_sc = y - y.mean()
        return X_sc, y_sc, std

    

    def fit(self, significance_level = 0.05):
        
        """Fit the model with Procedure 1 and Procedure 2. 
        
        Procedure 1: Compute importance scores for predictors. 
        
        Procedure 2: Compute coefficients and Select variables.
        
        Parallel processing of Spark
        One important parameter for parallel collections is the number of partitions to cut the dataset into. Spark will run one task for each partition of the cluster. 
        Typically you want 2-4 partitions for each CPU in your cluster. 
        Normally, Spark tries to set the number of partitions automatically based on your cluster. 
        However, you can also set it manually by passing it as a second parameter to parallelize (e.g. sc.parallelize(data, 10)).
        
        Parameters
        ----------      
        significance_level : float [default=0.05]
            Criteria used for selecting variables.
        

        Attributes
        ----------
        sc.parallelize(): method
            The sc.parallelize() method is the SparkContext's parallelize method to create a parallelized collection. 
            This allows Spark to distribute the data across multiple nodes, instead of depending on a single node to process the data.
        map(): method
            A map is a transformation operation in Apache Spark. It applies to each element of RDD and it returns the result as new RDD. 
            In the Map, operation developer can define his own custom business logic. The same logic will be applied to all the elements of RDD.
        Procedure_1_coef_ : array
            Estimated coefficients by Elastic_net.
        Procedure_2_coef_ : array
            Estimated coefficients by Adaptive_LASSO.
        coef_ : array
            Estimated coefficients by Hi-LASSO.
        p_values_ : array
            P-values of each coefficients.
        selected_var_: array
            Selected variables by significance test.
            
        Returns
        -------
        self : object 
        """
        
        # Procedure 1: Compute coefficients and importance scores for predictors.
        # Perform parallel processing by mapping the number of bootstraps.
        Elastic_Estimate = self.Estimate_coefficient_Elastic
        Procedure_1_coef = sc.parallelize(range(self.B), 8).map(lambda x: Elastic_Estimate(x)).collect()
        Procedure_1_coef_ = np.array(list(Procedure_1_coef)).T
        
        
        # Compute the importance scores
        Importance_score = np.nanmean(np.abs(Procedure_1_coef_), axis=1)
        Importance_score_2 = np.where(Importance_score == 0, 1e-10, Importance_score)
        self.Importance_score_fin = Importance_score_2 / Importance_score_2.sum()
    
        print('Procedure_1_fin.')
        
        
        # Procedure 2: Compute coefficients and Select variables.
        # Perform parallel processing by mapping the number of bootstraps.
        Adaptive_Estimate = self.Estimate_coefficient_Adaptive
        Procedure_2_coef = sc.parallelize(range(self.B), 8).map(lambda x: Adaptive_Estimate(x)).collect()
        Procedure_2_coef_ = np.array(list(Procedure_2_coef)).T

        # Estimate Final Coefficient and Select variables.
        coef = np.nanmean(Procedure_2_coef_, axis=1)
        self.coef_ = coef
        self.p_values = self.P_value_calculate(Procedure_2_coef_)
        self.selected_var = np.where(self.p_values < significance_level / self.n_feature, np.nanmean(Procedure_2_coef_, axis=1), 0)
        
        print('Procedure_2_fin.')
        return self
    
    
    
    def Estimate_coefficient_Elastic(self, value):
        
        """
        Estimation of coefficients for each bootstrap sample using Elastic_net
        
        Returns
        -------
        coef_result : coefficient for Elastic_Net        
        """
        
        # Set random seed as each bootstrap_number.
        seed = None
        seed = (seed if seed is not None else int(binascii.hexlify(os.urandom(4)), 16))
        rs = np.random.RandomState(seed)
        
        # Randomly select q1 on each bootstrap sample
        # Generate bootstrap index of sample and predictor
        q = self.q1
        select_prop = None
        coef_result = np.zeros(self.n_feature)
        Selected_q = rs.choice(np.arange(self.n_feature), size = self.q1, replace=False, p = select_prop)
        Bootstrap_Index = rs.choice(np.arange(self.n_sample), size = self.n_sample, replace=True, p = None)
        X_train_B = self.X[Bootstrap_Index, :][:, Selected_q]
        y_train_B = self.y[Bootstrap_Index]
        
        # Standardization
        X_train_Data, y_train_Data, std = self.standardization(X_train_B, y_train_B)
        
        # Search for otpimal alpha
        mses = np.array([])
        alphas = np.arange(0, 1.1, 0.1)
        for j in alphas:
            cv_enet = glmnet.ElasticNet(standardize=False, fit_intercept=False, n_splits=self.cv, scoring='mean_squared_error', alpha=j).fit(X_train_Data, y_train_Data)
            mses = np.append(mses, cv_enet.cv_mean_score_.max())
        opt_alpha = alphas[mses.argmax()]
         
        # Estimate coefficients
        enet_fin = glmnet.ElasticNet(standardize=False, fit_intercept=False, n_splits=self.cv, scoring='mean_squared_error', alpha=opt_alpha)
        coef_result[Selected_q] = (enet_fin.fit(X_train_Data, y_train_Data).coef_) / std
        
        return coef_result
        

        
    def Estimate_coefficient_Adaptive(self, value):
        
        """
        Estimation of coefficients for each bootstrap sample using Adaptive_LASSO
        
        Returns
        -------
        coef_result : coefficient for Adaprive_LASSO         
        """
        
        # Set random seed as each bootstrap_number.
        seed = None
        seed = (seed if seed is not None else int(binascii.hexlify(os.urandom(4)), 16))
        rs = np.random.RandomState(seed)

        # Randomly select q2 on each bootstrap sample.
        # Generate bootstrap index of sample and predictor.
        q = self.q2
        select_prop = self.Importance_score_fin
        coef_result = np.zeros(self.n_feature)
        Selected_q = rs.choice(np.arange(self.n_feature), size = self.q1, replace=False, p = select_prop)
        Bootstrap_Index = rs.choice(np.arange(self.n_sample), size = self.n_sample, replace=True, p = None)
        X_train_B = self.X[Bootstrap_Index, :][:, Selected_q]
        y_train_B = self.y[Bootstrap_Index]                             
        
        # Standardization
        X_train_Data, y_train_Data, std = self.standardization(X_train_B, y_train_B)
        
        # Estimate coefficients
        ad_fin = glmnet.ElasticNet(fit_intercept = False, standardize = False,  n_splits = self.cv, scoring='mean_squared_error', alpha = 1)
        coef_result[Selected_q] = (ad_fin.fit(X_train_Data, y_train_Data, relative_penalties = 1 / (select_prop[Selected_q] * 100)).coef_) / std
 
        return coef_result
        
    
    
    def P_value_calculate(self, coef_result):
        
        """
        Compute p-values of each predictor for Statistical Test of Variable Selection.
        """
        # Calculate_Boolean: non-zero and notnull of coef_result
        not_null_value = np.isfinite(coef_result)
        Calculate_Boolean = np.logical_and(not_null_value, coef_result != 0).sum(axis=1)
        
        # pi: the average of the selcetion ratio of all feature variables in B boostrap samples.
        pi = Calculate_Boolean.sum() / not_null_value.sum().sum()
        return binom.sf(Calculate_Boolean - 1, n = self.B, p = pi)
    
