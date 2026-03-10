SRC = src
VENV = .venv
VENV_FILE = $(VENV)/.pyinstall.timestamp
UV_FILES = pyproject.toml uv.lock
UV_RUN = uv run python3 -m

all: install run

install: $(VENV_FILE)

$(VENV_FILE): $(UV_FILES)
	@echo "Installing..."
	uv sync
	@touch $(VENV_FILE)

run: install
	@echo "Running..."
	$(UV_RUN) $(SRC)

debug: install
	@echo "debugging..."
	$(UV_RUN) pdb -m $(SRC)

lint: install
	@echo "Linting..."
	@echo "Flake8: "
	$(UV_RUN) flake8 src
	@echo "Mypy: "
	$(UV_RUN) mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: install
	@echo "Linting strictly..."
	@echo "Flake8: "
	$(UV_RUN) flake8 src
	@echo "Mypy: "
	$(UV_RUN) mypy src --strict

clean:
	@echo "Cleaning temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf output/
	@rm -rf .mypy_cache
	@rm -rf .pytest_cache
	@rm -rf $(VENV)

.PHONY: all run clean lint lint-strict debug install