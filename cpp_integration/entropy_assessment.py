import os
import subprocess
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
CPP_BINARY_PATH = config['cpp_binary_path']


def run_entropy_assessment(
    input_file,
    output_file=None,
    method='most_common',
    bits_per_symbol=None,
    verbose=0,
    quiet=False,
    initial_entropy=False,
    conditioned_entropy=False,
    all_data=False,
    truncate=False,
    l_option=None
):    
    """
    Run the SP800-90B EntropyAssessment C++ binary on the given input file.

    Args:
        input_file (str): Path to the input data file.
        output_file (str, optional): Path for output file. If None, output is captured.
        method (str): Which entropy assessment method to use.
        bits_per_symbol (int, optional): Number of bits per symbol (e.g., 8 for byte-wise data).
        verbose (int, optional): Number of times to add -v for verbosity.
        quiet (bool, optional): If True, adds -q for quiet mode.
        initial_entropy (bool, optional): If True, adds -i for initial entropy estimate.
        conditioned_entropy (bool, optional): If True, adds -c for conditioned sequential dataset entropy estimate.
        all_data (bool, optional): If True, adds -a to use all data for H_bitstring.
        truncate (bool, optional): If True, adds -t to truncate data for H_bitstring.
        l_option (tuple or str, optional): If set, adds -l <index>,<samples>.

    Returns:
        str: Output from the C++ binary (if output_file is None), else None.
    """
    binary = os.path.join(CPP_BINARY_PATH, method)
    cmd = [binary]
    if initial_entropy:
        cmd.append('-i')
    if conditioned_entropy:
        cmd.append('-c')
    if all_data:
        cmd.append('-a')
    if truncate:
        cmd.append('-t')
    if verbose:
        cmd.extend(['-v'] * int(verbose))
    if quiet:
        cmd.append('-q')
    if l_option:
        if isinstance(l_option, str):
            cmd.extend(['-l', l_option])
        else:
            cmd.extend(['-l', f"{l_option[0]},{l_option[1]}"])
    cmd.append(input_file)
    if bits_per_symbol is not None:
        cmd.append(str(bits_per_symbol))
    if output_file:
        cmd += ['-o', output_file]
        subprocess.run(cmd, check=True)
        return None
    else:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout
