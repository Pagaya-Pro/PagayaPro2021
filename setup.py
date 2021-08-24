from setuptools import setup, find_packages

setup(
    name='PagayaPRO',
    version='1.0.0',
    url='https://github.com/mypackage.git',
    author='Author Name',
    author_email='author@gmail.com',
    description='Description of my package',
    packages=find_packages(),    
    install_requires=["papermill", "ipython", "jupyterlab", "pandas", "sklearn"],
)
