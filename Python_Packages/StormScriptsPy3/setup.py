from setuptools import setup

setup(
    name='StormScriptsSerial',
    version='2.0.0',
    description=('Python3 Paralell Storm Data Scripts for retrieving data on' +
                 'storms in a set region and plotting various stats'),
    license='BSD-2',
    packages=['S_box', 'S_dataminer', 'S_calc-plotter'],
    author='University of Leeds',
    author_email='h.l.burns@leeds.ac.uk',
    keywords=['Storms'],
    url='https://github.com/cemac/StormTrackingScripts'
)
