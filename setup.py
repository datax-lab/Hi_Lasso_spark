import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Hi_LASSO_spark",
    version="0.1.1",
    author="Seungha Jeong",
    author_email="jsh29368602@gmail.com",
    description="High-Demensional LASSO_spark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/datax-lab/Hi_Lasso_spark",
    packages=setuptools.find_packages(),
    install_requires=[
          'glmnet'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
