from setuptools import setup

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
    install_requires=[
          '',
      ], # Pip dependencies
    dependency_links=['http://github.com/user/repo/tarball/master#egg=package-1.0'] # Iris link
    author='University of Leeds',
    author_email='h.l.burns@leeds.ac.uk',
    keywords=['Storms'],
    url='https://github.com/cemac/StormTrackingScripts',
    include_package_data=True
)
