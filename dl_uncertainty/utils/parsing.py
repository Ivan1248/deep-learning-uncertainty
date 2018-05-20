import re
import numpy as np

def parse_log(log_lines, subset='val'):
    read = False
    curves = dict()
    for line in log_lines:
        if read:
            read = False
            evals = re.findall("(\w+[=\s]\d.\d+)", line, re.IGNORECASE)
            evals = map(lambda x: re.split('[=\s]', x), evals)
            for name, value in evals:
                curves[name] = curves.get(name, []) + [float(value)]
        else:
            read = re.search(f"Testing(?:\.\.\.| \({subset})", line,
                             re.IGNORECASE) is not None
    return {k: np.array(v) for k, v in curves.items()}
