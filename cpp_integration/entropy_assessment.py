import os
import subprocess
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
CPP_BINARY_PATH = config['cpp_binary_path']


def run_entropy_assessment(input_file, output_file=None, method='most_common'):    
    """
    Run the SP800-90B EntropyAssessment C++ binary on the given input file.

    Args:
        input_file (str): Path to the input data file.
        output_file (str, optional): Path for output file. If None, output is captured.
        method (str): Which entropy assessment method to use (default: 'most_common').

    Returns:
        str: Output from the C++ binary (if output_file is None), else None.
    """
    binary = os.path.join(CPP_BINARY_PATH, method)
    cmd = [binary, input_file]
    if output_file:
        cmd += ['-o', output_file]
        subprocess.run(cmd, check=True)
        return None
    else:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout
