import os
import shutil

# Auto-copy notebooks into docs/notebooks/ so Sphinx can include them in the
# toctree (Sphinx cannot reference files outside its source directory).
# docs/notebooks/ is gitignored — the canonical copies live in notebooks/.
_here = os.path.dirname(__file__)
_nb_src = os.path.join(_here, "..", "notebooks")
_nb_dst = os.path.join(_here, "notebooks")
os.makedirs(_nb_dst, exist_ok=True)
for _nb in [
    "evaluation.ipynb",
    "gspy_classification.ipynb",
    "inject_into_detector_noise.ipynb",
]:
    _src = os.path.join(_nb_src, _nb)
    if os.path.exists(_src):
        shutil.copy(_src, os.path.join(_nb_dst, _nb))

project = "GlitchGAN"
author = "Tom Dooney"
copyright = "2026, Tom Dooney"
html_title = "GlitchGAN"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",       # Google/NumPy-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
    "myst_parser",               # Markdown support
    "nbsphinx",                  # Render Jupyter notebooks
]

autoapi_dirs = ["../src"]
autoapi_type = "python"
autoapi_options = ["members", "undoc-members", "show-inheritance"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable", None),
}

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "dark_css_variables": {
        "--color-background-primary":   "#0f1117",
        "--color-background-secondary": "#161b27",
        "--color-background-hover":     "#1c2333",
        "--color-background-border":    "#2a3147",
        "--color-brand-primary":        "#58a6ff",
        "--color-brand-content":        "#79c0ff",
        "--color-foreground-primary":   "#e6edf3",
        "--color-foreground-secondary": "#8b949e",
        "--color-foreground-muted":     "#6e7681",
        "--color-foreground-border":    "#30363d",
        "--color-sidebar-background":               "#0d1117",
        "--color-sidebar-background-border":        "#21262d",
        "--color-sidebar-brand-text":               "#e6edf3",
        "--color-sidebar-caption-text":             "#8b949e",
        "--color-sidebar-link-text":                "#c9d1d9",
        "--color-sidebar-link-text--top-level":     "#e6edf3",
        "--color-sidebar-item-background--current": "#1f2937",
        "--color-sidebar-item-background--hover":   "#1c2333",
        "--color-sidebar-item-expander-background--hover": "#1c2333",
        "--color-inline-code-background": "#1c2333",
        "--color-admonition-background":  "#161b27",
    },
    "light_css_variables": {
        "--color-brand-primary": "#0969da",
        "--color-brand-content": "#0969da",
    },
    "navigation_with_keys": True,
}

exclude_patterns = ["_build", "**/_build", "**.ipynb_checkpoints"]

# Do not re-execute notebooks during the docs build
nbsphinx_execute = "never"
