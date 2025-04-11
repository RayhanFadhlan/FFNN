PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

.PHONY: all setup install run clean cleanall

all: setup install

setup:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created at $(VENV)"

install: setup
	@echo "Installing required packages..."
	$(PIP) install --upgrade pip
	$(PIP) install numpy matplotlib scikit-learn tqdm networkx seaborn
	$(PIP) install numpy matplotlib scikit-learn tqdm
	@echo "Packages installed successfully"

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	@echo "Cleaned up successfully"

cleanall: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV)
	@echo "Virtual environment removed"
