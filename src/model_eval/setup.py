"""A setuptools based setup module.
	See:
	https://packaging.python.org/en/latest/distributing.html
	https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI
# Fields marked as "Optional" maybe commented out.

setup(
    # This is the name of your project. The first time you publish this
    # package, this name will be registered for you. It will determine how
    # users can install this project, e.g.:
    #
    # $ pip install find_nodules
    #
    # And where it will live on PyPI: https://repos.infervision.com/repository/pypi-local/
    #
    # There are some restrictions on what makes a valid project name
    # specification here:
    # https://packaging.python.org/specifications/core-metadata/#name
    name="model_eval", # Required

    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    #
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version="0.1.6", # Required

    # This is a one-line description or tagline of what your project does. This
    # corresponds to the "Summary" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#summary
    description="Python package for model evaluation", # Required

    # This is an optional longer description of your project that represents
    # the body of text which users will see when they visit PyPI
    #
    # Often, this is the same as your README, so you can just read it in from
    # that file directly (as we have already done above)
    #
    # This field corresponds to the "Description" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-optional
    long_description=long_description, # Optional

    # This should be a valid link to your project's main homepage.
    #
    # This field corresponds to the "Home-Page" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#home-page-optional
    url="https://github.com/LanqingLi/model_eval", # Optional

    # This should be your name or the name of the organization which owns the project.
    author="Lanqing Li, Infervision Co., Ltd.", # Optional

    # This should be a valid email address corresponding to the author listed
    # above.
    author_email="llanqing@infervision.com", # Optional

    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[ # Optional
        # How mature is this project? Common values are
        # 3 - Alpha
        # 4 - Beta
        # 5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. IN particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],

    # This field adds keywords for your project which will appear on the
    # project page. What does your project relate to?
    #
    # Note that this is a string of words separated by whitespace, not a list.
    keywords='sample setuptools development', # Optional

    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the 'py_modules' argument instead as follows, which will expect a file
    # called 'my_module.py' to exist:
    #
    # py_modules=["my_module"],
    #
    packages=['model_eval', 'model_eval.common', 'model_eval.lung', 'model_eval.tools', 'model_eval.brain',], # Required

    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['pandas>=0.22.0', 'pandas!=0.23.0','networkx<=2.0', 'opencv-python>=3.3.0.10', 'matplotlib<=2.1.1'], # Optional

    # List additional groups of dependencies here (e.g. development
    # dependencies). Users will be able to install these using the "extras"
    # syntax, for example:
    #
    # $ pip install sampleproject[dev]
    #
    # Similar to 'install_requires' above, these must be valid existing
    # projects.
    #extras_require={ # Optional
    #    'dev': ['check-manifest'],
    #    'test': ['coverage'],
    #},

    # If there are data files included in your packages that need to be
    # installed, specify them here.
    #
    # If using python 2.6 or earlier, then these have to be included in
    # MANIFEST.in as well.
    #package_data={  # Optional
    #    'sample': ['package_data.dat'],
    #},
    #package_data={'model_eval' : ['classname_labelname_mapping.xls'],},
    include_package_data=True,

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    #
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    #data_files=[('my_data', ['data/data_file'])], # Optional

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # 'pip' to create the appropriate form of executable for the target platform.
    #
    # For example, the following would provide a command called 'sample' which
    # executes the function 'main' from this package when invoked:
    entry_points={ # Optional
        'console_scripts': [
            'model_eval=model_eval:main',
        ],
    },
)