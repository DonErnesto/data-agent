from setuptools import setup, find_packages

setup(
    name='data_agent',
    version='0.1.0', # You can change the version as needed
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # List your project's dependencies here, e.g.:
        # 'pandas>=1.0',
        # 'litellm',
    ],
    python_requires='>=3.6',
)
