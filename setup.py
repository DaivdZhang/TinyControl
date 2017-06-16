from setuptools import setup, find_packages

PACKAGES = ['tcontrol'] + ["tcontrol.%s" % i for i in find_packages('tcontrol')]

setup(
    name='tinyControl',
    version='0.4.0',
    packages=PACKAGES,
    license='BSD 3-clause',
    install_requires=['sympy >= 1.0', 'numpy >= 1.12.0', 'matplotlib >= 2.0.0']
)
