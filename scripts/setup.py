#!/usr/bin/env python3
"""
Setup script for the RL Trading Agent project.
Creates necessary directories and initializes the project structure.
"""

import os
import sys
from pathlib import Path

def setup_project():
    """Setup the project directory structure and ensure all necessary folders exist."""
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Directories to create
    directories = [
        "models",
        "data/raw",
        "data/processed", 
        "logs",
        "results",
        "checkpoints",
        "scripts/experiments",
        "notebooks/backup"
    ]
    
    print("Setting up RL Trading Agent project structure...")
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Create gitignore if it doesn't exist
    gitignore_path = project_root / ".gitignore"
    if not gitignore_path.exists():
        gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyTorch
*.pth
*.pt

# Jupyter Notebook
.ipynb_checkpoints

# Environment
.env
.venv
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Project specific
models/*.zip
logs/*.log
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep
results/
checkpoints/

# OS
.DS_Store
Thumbs.db
"""
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        print("✓ Created .gitignore")
    
    # Create placeholder files to maintain directory structure
    placeholder_dirs = ["data/raw", "data/processed", "logs", "results", "checkpoints"]
    for directory in placeholder_dirs:
        placeholder_path = project_root / directory / ".gitkeep"
        if not placeholder_path.exists():
            placeholder_path.touch()
    
    print("\n🎉 Project setup complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Train a model: python src/main.py train --ticker AAPL")
    print("3. Start the web UI: streamlit run src/ui/app.py")
    print("4. Start the API: python src/inference/api.py")

if __name__ == "__main__":
    setup_project()
