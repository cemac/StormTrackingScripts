from setuptools import setup, find_packages

setup(
    name='StormScriptsPy2',
    version='2.0.0',
    description=('Python2 Paralell Storm Data Scripts for retrieving data on' +
                 'storms in a set region and plotting various stats'),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering',
      ],
    license='BSD-2',
    packages=find_packages(),
    setup_requires=['setuptools-yaml'],
    metadata_yaml='StormScripts_py2.yml',
    data_files=['data/all_vars_template.csv', 'data/stash_vars.csv'],
    author='University of Leeds',
    author_email='h.l.burns@leeds.ac.uk',
    keywords=['Storms'],
    url='https://github.com/cemac/StormTrackingScripts',
    include_package_data=True
)
