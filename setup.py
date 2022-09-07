#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Patrick John Chia",
    author_email='patrickjohncyh@icloud.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="RecList",
    install_requires=requirements,
    license="MIT license",
    long_description='',
    long_description_content_type='text/x-rst',
    include_package_data=True,
    keywords='nlp_ml_api',
    name='nlp_ml_api',
    packages=find_packages(include=['nlp_ml_api', 'nlp_ml_api.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/patrickjohncyh/nlp_ml_api',
    version='0.0.1',
    zip_safe=False,
)
