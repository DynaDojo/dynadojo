from setuptools import setup, find_packages


setup(
    name="dynadojo",
    version="0.1.5",
    description="An extensible platform for benchmarking sample efficiency in dynamical system identification",
    url="https://github.com/FlyingWorkshop/DynaDojo",
    author="Logan Mondal Bhamidipaty",
    author_email="loganmb@stanford.edu",
    license="MIT",
    python_requires=">=3.10",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10"
    ],
    install_requires=[
        "tqdm>=4.65.0",
        "numpy>=1.23.5",
        "torch>=2.0.1",
        "torchdiffeq>=0.2.3",
        "joblib>=1.2.0",
        "keras>=2.12.0",
        "tensorflow",
        "seaborn>=0.12.2",
        "scipy>=1.10.1",
        "scikit-learn>=1.2.2",
        "cellpylib",
        "matplotlib",
        "pandas",
        "networkx",
    ],
    packages=find_packages(),
)