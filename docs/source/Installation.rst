Installation guide
==================

Dependencies
------------

``Hi-LASSO-spark`` support Python 3.6+, Additionally, you will need ``numpy``, ``scipy``, and ``glmnet``. 


Installing pyspark
----------------------
To run Hi-LASSO-spark, you need to install a pyspark. 
In the prompt environment, run the following installation and update::

  pip install pyspark
  
After installing pyspark, it will be installed with 2.4.6 version. 
To run Hi-LASSO-spark, you must upgrade to version 3.0.0.::

  pip install pyspark --upgrade
  
After the upgrade, 3.0.0 version is executed.


Installing Hi-LASSO-spark
----------------------

``Hi-LASSO-spark`` is available through PyPI and can easily be installed with a
pip install::

    pip install hi_lasso_pyspark

The PyPI version is updated regularly, however for the latest update, you
should clone from GitHub and install it directly::

    git clone https://github.com/datax-lab/Hi_Lasso_spark.git
    cd hi_lasso_spark
    python setup.py
	
Installation error
---------------------
If the `glmnet` packages failed, you can try a follow solutions.


.. code-block:: console

	 error: extension '_glmnet' has Fortran sources but no Fortran compiler found
	 
You should install ``anaconda3`` and then install conda ``fortran-compiler``.

- `anaconda3 <https://www.anaconda.com/products/individual>`_
- `fortran compiler <https://anaconda.org/conda-forge/fortran-compiler/>`_

	
.. code-block:: console

	 error: Microsoft Visual C++ 14.0 is required. Get it with "Build Tools for Visual Studio": https://visualstudio.microsoft.com/downloads/	
	
You need to install `Microsoft Visual C++ 14.0`.

- `reference <https://stackoverflow.com/questions/44951456/pip-error-microsoft-visual-c-14-0-is-required/44953739>`_
