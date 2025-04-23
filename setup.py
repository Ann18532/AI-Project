from setuptools import setup, find_packages

setup(
    name='uiki_auth',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'pytest',
    ],
    entry_points={
        'console_scripts': [
            'run-uiki=examples.run_uiki:main',
        ],
    },
)