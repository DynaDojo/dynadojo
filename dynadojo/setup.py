from setuptools import setup


setup(
    name="dynadojo",
    version="1.0.0",
    description="An extensible platform for benchmarking sample efficiency in dynamical system identification",
    url="https://github.com/FlyingWorkshop/DynaDojo",
    author="Logan Mondal Bhamidipaty",
    author_email="loganmb@stanford.edu",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10"
    ],
    # install_requires=[
    #
    # ],
    packages=["dynadojo"],
    include_package_data=True,
)