from setuptools import setup

setup(
    name='StormScriptsPy3',
    version='2.0.0',
    description=('Python3 Paralell Storm Data Scripts for retrieving data on' +
                 'storms in a set region and plotting various stats'),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
      ],
    license='BSD-2',
    packages=find_packages(),
    data_files=['../data']
    setup_requires=['setuptools-yaml'],
    metadata_yaml='StormScripts_py2.yml',
    author='University of Leeds',
    author_email='h.l.burns@leeds.ac.uk',
    keywords=['Storms'],
    url='https://github.com/cemac/StormTrackingScripts',
    include_package_data=True
)
