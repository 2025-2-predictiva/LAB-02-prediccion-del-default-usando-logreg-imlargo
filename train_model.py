#!/usr/bin/env python3
"""
Full training script for the credit card default prediction model.
This script trains the model with comprehensive hyperparameter tuning.
NOTE: This may take several minutes to complete.
"""
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from homework.homework import main

if __name__ == "__main__":
    print("\n" + "!" * 80)
    print("WARNING: This training will take several minutes to complete.")
    print("The model uses 10-fold cross-validation with a comprehensive parameter grid.")
    print("!" * 80 + "\n")
    
    main()
