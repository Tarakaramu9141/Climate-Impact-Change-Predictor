from setuptools import setup, find_packages

setup(
    name="climate_change_predictor",
    version="0.1.0",
    description="Climate Change Impact Predictor using Satellite Data and ML",
    author="K TARAKA RAMU",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        'earthengine-api>=0.1.270',
        'geopandas>=0.9.0',
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.0',
        'tensorflow>=2.5.0',
        'streamlit>=1.0.0',
        'plotly>=5.0.0',
        'python-dotenv>=0.19.0'
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'download-climate-data=src.data.download_data:batch_download_all_regions',
            'process-climate-data=src.data.process_data:clean_and_process_data',
            'train-rf-model=src.models.train_rf:train_random_forest',
            'train-nn-model=src.models.train_nn:train_neural_network'
        ]
    }
)