from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='quantile_transformer_tf',
    version='1.2',
    description='An implementation of QuantileTransformer in tensorflow',
    long_description=long_description,
    url='https://github.com/yandexdataschool/QuantileTransformerTF',
    author='Nikita Kazeev',
    author_email='nikita.kazeev@cern.ch',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
    ],
    packages=find_packages(),
    install_requires=['numpy>=1.15.0',
                      'tensorflow>=1.9']
)
