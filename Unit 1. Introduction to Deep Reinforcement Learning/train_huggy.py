#!/usr/bin/env python3

import os
from pickle import TRUE
import time
import subprocess
try:
  from fabulous import color as fb_color
except ImportError:
  subprocess.run('pip install fabulous'.split(' '), check=True)
  from fabulous import color as fb_color
import shutil
import sys

subprocess.run('pip install wandb'.split(' '), check=True)

try:
  from dotenv import load_dotenv
except ImportError:
  subprocess.run('pip install python-dotenv'.split(' '), check=True)
  from dotenv import load_dotenv
load_dotenv('/content/drive/MyDrive/.env')

assert os.environ.get('WANDB_API_KEY') is not None
subprocess.run(["wandb", "login", os.environ["WANDB_API_KEY"]], check=True)

assert os.environ.get('HF_TOKEN') is not None
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get('HF_TOKEN')ß
assert os.environ["HUGGING_FACE_HUB_TOKEN"] is not None
subprocess.run(["hf", "auth", "login", "--token", os.environ["HF_TOKEN"]], check=True)



start_time = time.perf_counter()
current_script_folder = os.getcwd()
print(fb_color.magenta("current_script_folder:"), current_script_folder)

def run_command(cmd, shell=False, cwd=None):
    """Run command with proper error handling"""
    try:
        print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        result = subprocess.run(cmd, check=True, shell=shell, cwd=cwd, 
                              capture_output=False, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        raise

def python_310_venv_setup():
    print("Starting ML-Agents setup...")
    
    # Clean up existing ml-agents directory
    if os.path.exists('./ml-agents'):
        print("Removing existing ml-agents directory...")
        shutil.rmtree('./ml-agents')

    # Clone ml-agents repository
    print("Cloning ml-agents repository...")
    run_command(['git', 'clone', '--depth', '1', 'https://github.com/Unity-Technologies/ml-agents'])

    # Check current Python version
    print("\n" + "="*10 + "CURRENT PYTHON VERSION:" + "="*10)
    run_command([sys.executable, '--version'])

    # Set up paths
    home_dir = os.path.expanduser('~')
    conda_path = os.path.join(home_dir, 'miniconda3')
    installer = 'Miniconda3-latest-Linux-x86_64.sh'
    
    try:
        # Download miniconda if not exists
        if not os.path.exists(installer):
            print("Downloading Miniconda...")
            run_command(['wget', 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh'])
        
        # Make installer executable
        run_command(['chmod', '+x', installer])
        
        # Install miniconda
        print(f"Installing Miniconda to {conda_path}...")
        run_command([f'./{installer}', '-b', '-f', '-p', conda_path])
        
        # Add conda to PATH for this session
        conda_bin = os.path.join(conda_path, 'bin')
        os.environ['PATH'] = f'{conda_bin}:' + os.environ.get('PATH', '')
        
        # Initialize conda
        conda_exe = os.path.join(conda_bin, 'conda')
        print("Initializing conda...")
        run_command([conda_exe, 'config', '--set', 'always_yes', 'true'])
        
        # Create environment with Python 3.10.12
        env_name = 'ml_agents_env'
        print(f"Creating conda environment '{env_name}' with Python 3.10.12...")
        run_command([conda_exe, 'create', '-n', env_name, 'python=3.10.12'])
        
        # Get paths to Python and pip in the new environment
        env_python = os.path.join(conda_path, 'envs', env_name, 'bin', 'python')
        env_pip = os.path.join(conda_path, 'envs', env_name, 'bin', 'pip')
        
        print(f"\n{'='*10}PYTHON VERSION IN NEW ENVIRONMENT:{'='*10}")
        run_command([env_python, '--version'])
        
        # Install ml-agents in the new environment
        print("Installing ml-agents in the new environment...")
        
        # Change to ml-agents directory
        ml_agents_dir = os.path.abspath('./ml-agents')
        run_command([env_pip, 'install', '-e', './ml-agents-envs'], cwd=ml_agents_dir)
        # run_command([env_pip, 'install', '-e', './ml-agents'], cwd=ml_agents_dir)
        # run_command([env_pip, 'install', '-e', './ml-agents[wandb]'], cwd=ml_agents_dir)
        run_command([env_pip, 'install', '-e', './ml-agents[wandb]', '--use-pep517'], cwd=ml_agents_dir)
        
        print(f"\n{'='*10}INSTALLATION COMPLETE{'='*10}")
        print(f"Python 3.10.12 installed at: {env_python}")
        print(f"Pip installed at: {env_pip}")
        print(f"To activate environment: conda activate {env_name}")
        print(f"Or run directly: {env_python} your_script.py")
        
        # Create activation script
        activation_script = 'activate_ml_agents.sh'
        with open(activation_script, 'w') as f:
            f.write(f"""#!/bin/bash
# Activate ML-Agents environment
export PATH="{conda_bin}:$PATH"
source {conda_bin}/activate {env_name}
echo "ML-Agents environment activated!"
echo "Python version: $(python --version)"
""")
        os.chmod(activation_script, 0o755)
        print(f"Created activation script: {activation_script}")
        
        print(f"\n{'='*10}CHECKING PYTHON VERSION{'='*10}")
        run_command([env_python, '--version'])
        
        # Return the paths to use later
        return env_python, env_pip, ml_agents_dir
        
    except subprocess.CalledProcessError as e:
        print(f"Error during setup: {e}")
        print("Falling back to system Python...")
        
        print(f"\n{'='*10}SYSTEM PYTHON VERSION:{'='*10}")
        run_command([sys.executable, '--version'])
        return None, None, None

# Set up the environment
env_python, env_pip, ml_agents_dir = python_310_venv_setup()

# Check if setup was successful
if not env_python or not env_pip:
    print("Environment setup failed, exiting...")
    sys.exit(1)

# Verify ml-agents installation
print(f"\n{'='*10}VERIFYING ML-AGENTS INSTALLATION{'='*10}")
try:
    run_command([env_python, '-c', 'import mlagents; print("✓ ML-Agents imported successfully!")'])
except subprocess.CalledProcessError:
    print("❌ ML-Agents import failed!")
    sys.exit(1)

# Clean up existing ml-agents directory
if os.path.exists('./trained-envs-executables'):
    print("Removing existing trained-envs-executables directory...")
    shutil.rmtree('./trained-envs-executables')
# Create directories for executables
print("Creating directories for executables...")
os.makedirs('./trained-envs-executables/linux', exist_ok=True)

# Download and extract Huggy environment
print("Downloading Huggy environment...")
huggy_zip_path = './trained-envs-executables/linux/Huggy.zip'
run_command(['wget', 'https://github.com/huggingface/Huggy/raw/main/Huggy.zip', '-O', huggy_zip_path])
run_command(['unzip', '-d', './trained-envs-executables/linux/', huggy_zip_path])
run_command(['chmod', '-R', '755', './trained-envs-executables/linux/Huggy'])

# Copy configuration file
huggy_yaml_source = os.path.join(current_script_folder, 'Huggy.yaml')
huggy_yaml_dest = os.path.join(ml_agents_dir, 'config', 'ppo', 'Huggy.yaml')

if os.path.exists(huggy_yaml_source):
    print("Copying Huggy.yaml configuration...")
    shutil.copy2(huggy_yaml_source, huggy_yaml_dest)
    
    print(fb_color.magenta("Huggy.yaml:"))
    run_command(['cat', huggy_yaml_dest])
else:
    print(f"❌ Huggy.yaml not found at {huggy_yaml_source}")
    print("Please ensure Huggy.yaml is in the same directory as this script")
    sys.exit(1)

# Set up environment variables for mlagents-learn
os.environ['PATH'] = f'{os.path.dirname(env_python)}:' + os.environ.get('PATH', '')


end_time = time.perf_counter()
print(fb_color.yellow(f"Total time before mlagents-learn: {round(end_time - start_time)} seconds"))
# Run ML-Agents training
print(f"\n{fb_color.magenta('Calling mlagents-learn:')}")
try:
    # Change to ml-agents directory for training
    huggy_env_path = os.path.abspath('./trained-envs-executables/linux/Huggy/Huggy')
    config_path = './config/ppo/Huggy.yaml'
    
    training_cmd = [
        env_python, '-m', 'mlagents.trainers.learn',
        config_path,
        '--env=' + huggy_env_path,
        '--run-id=Huggy2',
        '--no-graphics'
        '--wandb'
        # # --- ADDED FOR WANDB INTEGRATION ---
        # '--wandb',
        # '--project=huggy-ppo-rl', # Replace with your W&B project name
        # '--group=huggy-training-run',    # Optional: Group multiple runs together
        # '--entity=anton-andreitsev-constructor',   # Optional: Your W&B username
    ]
    
    print(f"Training command: {' '.join(training_cmd)}")
    run_command(training_cmd, cwd=ml_agents_dir)
    
except subprocess.CalledProcessError as e:
    print(f"❌ Training failed: {e}")
    # Try alternative approach using mlagents-learn directly
    print("Trying alternative approach with mlagents-learn command...")
    try:
        mlagents_learn = os.path.join(os.path.dirname(env_python), 'mlagents-learn')
        if os.path.exists(mlagents_learn):
            alt_cmd = [
                mlagents_learn,
                config_path,
                '--env=' + huggy_env_path,
                '--run-id=Huggy',
                '--no-graphics'
            ]
            run_command(alt_cmd, cwd=ml_agents_dir)
        else:
            print("❌ mlagents-learn executable not found")
            sys.exit(1)
    except subprocess.CalledProcessError as e2:
        print(f"❌ Alternative training approach also failed: {e2}")
        sys.exit(1)

print(f"\n{'='*50}")
print("🎉 TRAINING COMPLETE! 🎉")
print(f"{'='*50}")
print("Training results should be saved in the results directory.")


try:
  subprocess.run('mlagents-push-to-hf --run-id="Huggy" --local-dir="./results/Huggy" --repo-id="antoncio/ppo-Huggy" --commit-message="Huggy"'.split(' '), check=True)
except Exception as e:
  print(fb_color.red("mlagents-push-to-hf failed!"))
  print(fb_color.red(e))



