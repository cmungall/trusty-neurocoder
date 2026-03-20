# trusty-neurocoder justfile

# Install all dependencies
install:
    uv pip install -e ".[dev,notebooks,docs]"

# Run all example scripts
examples:
    uv run python examples/cajal_demo.py
    uv run python examples/exponential_decay.py
    uv run python examples/coupled_decay.py
    uv run python examples/learn_unknown_function.py
    uv run python examples/century_lite.py

# Run all pytest tests
test:
    uv run pytest tests/ -v

# Execute all notebooks in-place (output embedded in notebook files)
notebooks:
    mkdir -p notebooks
    uv run jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=600 \
        --inplace \
        notebooks/*.ipynb

# Execute a single notebook: just nb notebooks/01_cajal_intro.ipynb
nb notebook:
    uv run jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=600 \
        --inplace \
        {{notebook}}

# Convert executed notebooks to HTML for viewing
html:
    uv run jupyter nbconvert --to html notebooks/*.ipynb --output-dir notebooks/html/

# Launch Jupyter Lab
lab:
    uv run jupyter lab notebooks/

# Serve mkdocs locally
docs:
    uv run mkdocs serve

# Build mkdocs site
docs-build:
    uv run mkdocs build

# Deploy to GitHub Pages
docs-deploy:
    uv run mkdocs gh-deploy
