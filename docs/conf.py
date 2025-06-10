# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

# -- Project information -----------------------------------------------------
import os
import sphinx_rtd_theme
from physicsnemo.sym import __version__ as version

project = "NVIDIA PhysicsNeMo Symbolic"
copyright = "2023, NVIDIA PhysicsNeMo Team"
author = "NVIDIA PhysicsNeMo Team"
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "external",
    "README.md",
    "CONTRIBUTING.md",
    "LICENSE.txt",
    "tests",
    "**.ipynb_checkpoints",
]

# Fake imports
autodoc_mock_imports = ["quadpy", "functorch"]

extensions = [
    "recommonmark",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
]
# source_parsers = { '.md': 'recommonmark.parser.CommonMarkParser',}
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

pdf_documents = [
    ("index", "rst2pdf", "Sample rst2pdf doc", "Your Name"),
]

napoleon_custom_sections = ["Variable Shape"]

# -- Options for HTML output -------------------------------------------------

# HTML theme options
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": True,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#000000",
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": False,
    # 'navigation_depth': 10,
    "sidebarwidth": 12,
    "includehidden": True,
    "titles_only": False,
}

# Additional html options
html_static_path = ["_static"]
html_css_files = [
    "css/nvidia_styles.css",
]
html_js_files = ["js/pk_scripts.js"]
# html_last_updated_fmt = ''

# Additional sphinx switches
math_number_all = True
todo_include_todos = True
numfig = True

_PREAMBLE = r"""
\usepackage{amsmath}
\usepackage{esint}
\usepackage{mathtools}
\usepackage{stmaryrd}
"""
latex_elements = {
    "preamble": _PREAMBLE,
    # other settings go here
}

latex_preamble = [
    (
        "\\usepackage{amssymb}",
        "\\usepackage{amsmath}",
        "\\usepackage{amsxtra}",
        "\\usepackage{bm}",
        "\\usepackage{esint}",
        "\\usepackage{mathtools}",
        "\\usepackage{stmaryrd}",
    ),
]

autosectionlabel_maxdepth = 1
