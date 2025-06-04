# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "TWL Spike Yolo"
copyright = "2025, Kirill Hitushkin"
author = "Kirill Hitushkin"
release = "v0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

intersphinx_mapping = {
    "torch": ("https://pytorch.org/docs/stable/", None),
    "norse": ("https://norse.github.io/norse/", None),
    "python": ("https://docs.python.org/3.12/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
add_module_names = False  # Remove namespaces from class/method signatures

html_static_path = []
templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"

html_context = {"default_mode": "dark"}
html_theme_options = {
    "repository_url": "https://github.com/KirillHit/twl_spike_yolo",
    "use_repository_button": True,
}
