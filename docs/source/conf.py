# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../src'))

# ADAPTED FROM: https://github.com/numpy/numpy/blob/main/doc/source/conf.py


# -- Project information -----------------------------------------------------

project = 'DynaDojo'
copyright = '2024'
author = 'Logan Mondal Bhamidipaty, Tommy Bruzzese, Caryn Tran, Rami Ratl Mrad, Max Kanwal '

import dynadojo

release = dynadojo.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_copybutton',  # adds a copy button to code blocks
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',  # Create neat summary tables
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',  # For NumPy formatted docstrings
]
# Autosummary settings
autosummary_generate = True  # Turn on sphinx.ext.autosummary
autosummary_imported_members = False

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

autodoc_typehints = "description"  # Automatically extract typehints when specified and place them in descriptions of the relevant function/method.
autodoc_class_signature = "separated"  # Don't show class signature with the class' name.
autodoc_inherit_docstrings = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# These folders are copied to the documentation's HTML output
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_logo = '_static/logo.svg'
html_favicon = '_static/dino.svg'

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2c6213",
        "color-brand-content": "#2c6213",
    },
    "dark_css_variables": {
        "color-brand-primary": "#2c6213",
        "color-brand-content": "#2c6213",
    },
}