from setuptools import setup, find_packages

setup(
    name='transformer-fault-prediction',
    version='0.1.0',
    description='Predict the fault of transformer',
    author='Dr J K Mishra',
    license='MIT',
    package_dir={'': 'src'},  # Key addition
    packages=find_packages(where='src'),  # Look in src directory
    install_requires=[
        'pandas',
        'scikit-learn',
        'joblib',
        'jsonschema',
        'python-dotenv'
    ],
)