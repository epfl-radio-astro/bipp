r"""
Selection export
"""

import numpy as np
import json

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)

def export_selection(selection, file_name):
    r"""
    Export selection dictionary to json.

    Args
        selection: dict
            Dictionary containing the scaled eigenvalues and indices for each image.
        file_name: str
            File name to write to.

    """
    with open(file_name, 'w') as f:
        json.dump(selection, f, cls=NumpyArrayEncoder)

