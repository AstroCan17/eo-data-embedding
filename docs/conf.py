# Copyright 2026 Can Deniz Kaya
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import os
import sys

sys.path.insert(0, os.path.abspath("../"))
package_path = os.path.abspath("..")
os.environ["PYTHONPATH"] = ":".join((package_path, os.environ.get("PYTHONPATH", "")))

# -- Project information -----------------------------------------------------

project = "eo-data-embedding"
copyright = "2023 ESA"
author = "Can Deniz Kaya"

# The full version, including alpha/beta/rc tags
# This value is set by the CI pipeline when building the documentation.
release = "REPLACE_ME"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinxcontrib.apidoc",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx_togglebutton",
    "jupyter_sphinx",
    "myst_nb",
    "numpydoc",
    "sphinxcontrib.mermaid",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/AstroCan17/eo-data-embedding",
    "repository_branch": "main",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "path_to_docs": "docs",
    "article_footer_items": ["help-feedback.html"],
    "home_page_in_toc": False,
    "logo": {
        # The logo image attributes must be set directly in the
        # _static/navbar-logo.html template
        "text": "eo-data-embedding",
    },
    "primary_sidebar_end": ["navbar-footer.html"],
    # The version switcher is disabled until a versions.json is published to
    # GitLab Pages; enabling it before that breaks the docs build.
}

# The logo image attributes must be set directly in the
# _static/navbar-logo.html template
# html_logo = "_static/esa.jpg"
html_title = "eo-data-embedding"
html_sidebars = {
    "**": [
        "navbar-logo.html",
        "search-field.html",
        "sbt-sidebar-nav.html",
    ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_last_updated_fmt = "%d/%m/%Y %H:%M"
today_fmt = "%d/%m/%Y %H:%M"

# Mermaid is loaded by sphinxcontrib.mermaid from its configured version (see
# `mermaid_version` below). We no longer force the local static .min.js: the
# 1.x extension injects the library itself, and a duplicate load conflicts.

# myst
myst_enable_extensions = ["linkify"]
myst_linkify_fuzzy_links = False
nitpicky = True

# Avoid errors during generation. See:
# - https://stackoverflow.com/q/11417221
# - https://github.com/python/cpython/issues/56184
# - https://bugs.python.org/issue11975
nitpick_ignore = [
    ("py:class", "abc.ABC"),
    ("py:class", "eopf.computing.abstract.EOProcessingUnit"),
    ("py:class", "eopf.product.eo_product.EOProduct"),
    ("py:data", "typing.Any"),
    ("py:obj", "identifier"),
]

nb_custom_formats = {
    ".md": ["jupytext.reads", {"fmt": "mystnb"}],
}

# Sphinx apidoc
autodoc_typehints = "signature"

apidoc_module_dir = "../eo_data_embedding"
apidoc_output_dir = "api"
apidoc_module_first = True
apidoc_toc_file = False
apidoc_separate_modules = False

autodoc_default_options = {
    "ignore-module-all": True,
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Mermaid sphinx integration
mermaid_verbose = True
# sphinxcontrib-mermaid 1.x parses this with packaging.Version, so it must be a
# valid version string (the template's "" disabled CDN injection but breaks 1.x).
mermaid_version = "10.9.1"
# House style applied to every {mermaid} block, so individual diagrams stay free
# of per-block %%{init}%% / front-matter and render consistently (ECSS: uniform
# notation across the document). 'neutral' reads well on the light docs theme;
# 'basis' curves and the spacing values reduce edge crossings under the dagre
# layout engine bundled with Mermaid 10.9.1 (ELK would require Mermaid 11+).
mermaid_init_js = (
    "mermaid.initialize({"
    "startOnLoad:true,"
    "theme:'neutral',"
    "securityLevel:'loose',"
    "flowchart:{curve:'basis',useMaxWidth:true,htmlLabels:true,"
    "nodeSpacing:45,rankSpacing:55}"
    "});"
)

# sphinx-test-report configuration
# See https://sphinx-test-reports.readthedocs.io/en/latest/configuration.html
tr_report_template = "_templates/test_report_template.txt"
