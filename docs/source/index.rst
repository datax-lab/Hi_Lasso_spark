.. Hi_Lasso_spark documentation master file, created by
   sphinx-quickstart on Mon Jun 22 12:46:10 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Hi-LASSO in pyspark
==========================================
Hi-LASSO_Spark(High-Demensinal LASSO Spark) is to improve the LASSO solutions for extremely high-dimensional data using pyspark. 
    
PySpark is the Python API written in python to support Apache Spark. 
Apache Spark is a distributed framework that can handle Big Data analysis. 
Spark is basically a computational engine, that works with huge sets of data by processing them in parallel and batch systems.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   Installation
   api_reference
   Test_package_using_csv
   Test_package_using_url
   Test_package_using_json

What is Hi-LASSO?
==========================================
Hi-LASSO(High-Dimensional LASSO) can theoretically improves a LASSO model providing better performance of both prediction and feature selection on extremely high-dimensional data. Hi-LASSO alleviates bias introduced from bootstrapping, refines importance scores, improves the performance taking advantage of global oracle property, provides a statistical strategy to determine the number of bootstrapping, and allows tests of significance for feature selection with appropriate distribution. In Hi-LASSO will be applied to use the pool of the python library to process parallel multiprocessing to reduce the time required for the model.


Credit
==================
Hi-LASSO was primarily developed by Dr. Youngsoon Kim, with significant contributions and suggestions by Dr. Joongyang Park, Dr. Mingon Kang, and many others. The pyspark package was developed by Seungha Jeong. Initial supervision for the project was provided by Dr. Mingon Kang.

Development of Hi-LASSO is carried out in the `DataX lab <http://dataxlab.org/index.php>`_ lab at University of Nevada, Las Vegas (UNLV).

If you use Hi-LASSO in your research, generally it is appropriate to cite the following paper: Y. Kim, J. Hao, T. Mallavarapu, J. Park and M. Kang, "Hi-LASSO: High-Dimensional LASSO," in IEEE Access, vol. 7, pp. 44562-44573, 2019, doi: 10.1109/ACCESS.2019.2909071.
