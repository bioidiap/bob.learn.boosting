#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <manuel.guenther@idiap.ch>
#Thu Feb 13 19:18:24 CET 2014
#
# This file contains the python (distutils/setuptools) instructions so your
# package can be installed on **any** host system. It defines some basic
# information like the package name for instance, or its homepage.
#
# It also defines which other packages this python package depends on and that
# are required for this package's operation. The python subsystem will make
# sure all dependent packages are installed or will install them for you upon
# the installation of this package.
#
# The 'buildout' system we use here will go further and wrap this package in
# such a way to create an isolated python working environment. Buildout will
# make sure that dependencies which are not yet installed do get installed, but
# **without** requiring adminstrative privileges on the host system. This
# allows you to test your package with new python dependencies w/o requiring
# administrative interventions.

bob_packages = ['bob.core', 'bob.io.base']

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.extension', 'bob.blitz'] + bob_packages))
from bob.blitz.extension import Extension, Library, build_ext

from bob.extension.utils import load_requirements
build_requires = load_requirements()

# Define package version
version = open("version.txt").read().rstrip()

packages = ['boost']
boost_modules = ['system']

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    # This is the basic information about your project. Modify all this
    # information before releasing code publicly.
    name='bob.learn.boosting',
    version=version,
    description='Boosting framework for Bob',

    url='https://gitlab.idiap.ch/bob/bob.learn.boosting',
    license='BSD',
    author='Manuel Guenther (with help of Rakesh Mehta)',
    author_email='manuel.guenther@idiap.ch',

    # If you have a better, long description of your package, place it on the
    # 'doc' directory and then hook it here
    long_description=open('README.rst').read(),

    # This line is required for any distutils based packaging.
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,

    # This line defines which packages should be installed when you "install"
    # this package. All packages that are mentioned here, but are not installed
    # on the current system will be installed locally and only visible to the
    # scripts of this package. Don't worry - You won't need adminstrative
    # privileges when using buildout.
    setup_requires = build_requires,
    install_requires = build_requires,

    # Declare that the package is in the namespace bob.learn


    ext_modules = [
      Extension("bob.learn.boosting.version",
        [
          "bob/learn/boosting/version.cpp",
        ],
        bob_packages = bob_packages,
        version = version,
        packages = packages,
        boost_modules = boost_modules,
      ),

      Library(
        'bob.learn.boosting.bob_learn_boosting',
        [
          "bob/learn/boosting/cpp/LossFunction.cpp",
          "bob/learn/boosting/cpp/JesorskyLoss.cpp",

          "bob/learn/boosting/cpp/StumpMachine.cpp",
          "bob/learn/boosting/cpp/LUTMachine.cpp",
          "bob/learn/boosting/cpp/BoostedMachine.cpp",

          "bob/learn/boosting/cpp/LUTTrainer.cpp",
        ],
        bob_packages = bob_packages,
        version = version,
        packages = packages,
        boost_modules = boost_modules,
      ),

      Extension(
        'bob.learn.boosting._library',
        [
          "bob/learn/boosting/main.cpp",
          "bob/learn/boosting/loss_function.cpp",
          "bob/learn/boosting/jesorsky_loss.cpp",

          "bob/learn/boosting/weak_machine.cpp",
          "bob/learn/boosting/stump_machine.cpp",
          "bob/learn/boosting/lut_machine.cpp",
          "bob/learn/boosting/boosted_machine.cpp",

          "bob/learn/boosting/lut_trainer.cpp",
        ],
        bob_packages = bob_packages,
        version = version,
        packages = packages,
        boost_modules = boost_modules,
      ),
    ],

    cmdclass = {
      'build_ext': build_ext
    },

    # Define the entry points for this package
    entry_points={

      # Console scripts, which will appear in ./bin/ after buildout
      'console_scripts': [
        'boosting_example.py = bob.learn.boosting.examples.mnist:main',
      ],

    },

    # Classifiers are important if you plan to distribute this package through
    # PyPI. You can find the complete list of classifiers that are valid and
    # useful here (http://pypi.python.org/pypi?%3Aaction=list_classifiers).
    classifiers = [
      'Framework :: Bob',
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
