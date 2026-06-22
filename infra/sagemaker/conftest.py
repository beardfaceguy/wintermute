import sys
import os

# Add this directory to sys.path so pytest can import deploy_qwen3.
# deploy_qwen3 itself then removes this entry before importing sagemaker,
# preventing the infra/sagemaker/ directory from shadowing the installed package.
sys.path.insert(0, os.path.dirname(__file__))
