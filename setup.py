from setuptools import setup, Extension
from setuptools import find_packages


setup(name='pyFeatSel',
      version='0.1',
      description='algorithms for feature selection',
      url='https://github.com/FelixKleineBoesing/pyFeatSel',
      author='Felix Kleine Bösing',
      license='WDL',
      packages=find_packages(),
      install_requires=['numpy', "pandas", "xgboost"],
      include_package_data=True,
      zip_safe=False)