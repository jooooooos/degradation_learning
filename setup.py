from setuptools import setup, find_packages
import os

setup(
    name='raas',
    version='0.1.0',
    description='Implementation of Profit Maximization for a Robotics-as-a-Service Model',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    author='Joo Seung Lee',
    author_email='jooseung_lee@berkeley.edu',  # Update as needed
    url='https://github.com/jooooooos/degradation_learning',  # Update with actual repo URL
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'numpy>=1.26.4',
        'scipy>=1.13.1',
        'gurobipy>=12.0.3',
        'torch>=2.6.0',  # Note: pytorch in yml, but package is 'torch'
        'pandas>=2.2.3',
        'matplotlib>=3.9.2',
        'tqdm>=4.66.5',
        'numba>=0.60.0',
    ],
    extras_require={
        'dev': [
            'jupyter>=1.0.0',
            'notebook>=7.2.2',
            'pytest',  # For testing
            'sphinx',  # For documentation
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Or your chosen license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9.20',
)