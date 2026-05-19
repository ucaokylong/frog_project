import numpy as np
import os

def inspect_npz(file_path):
    print(f"Loading file: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    keys = list(data.keys())
    print(f"Keys found in NPZ file: {keys}\n")

    for key in keys:
        val = data[key]
        print(f"=== Key: '{key}' ===")
        print(f"Type: {type(val)}")
        
        # If it's a numpy array, print shape and dtype
        if isinstance(val, np.ndarray):
            print(f"Shape: {val.shape}")
            print(f"Dtype: {val.dtype}")
            
            # Check if it has 0 dimensions (e.g. pickled dict/object)
            if val.ndim == 0:
                try:
                    item = val.item()
                    print(f"Contained object type: {type(item)}")
                    if isinstance(item, dict):
                        print_dict_structure(item)
                    else:
                        print(f"Value: {item}")
                except Exception as e:
                    print(f"Could not call .item() on 0-dim array: {e}")
            else:
                # Print a small slice or summary
                if val.size < 50:
                    print("Values:")
                    print(val)
                else:
                    print("Sample values (first 5):")
                    print(val.flatten()[:5])
        else:
            print(f"Value: {val}")
        print("-" * 50 + "\n")

def print_dict_structure(d, indent=0):
    spacing = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{spacing}- {k}: (dict)")
            print_dict_structure(v, indent + 1)
        elif isinstance(v, np.ndarray):
            print(f"{spacing}- {k}: np.ndarray of shape {v.shape}, dtype {v.dtype}")
            if v.size < 20:
                print(f"{spacing}  Value: {v}")
        else:
            print(f"{spacing}- {k}: {type(v).__name__} = {v}")

if __name__ == "__main__":
    # Use absolute path to ensure correctness regardless of from where the script is run
    file_path = "/home/s2410433/frog_project/temporal_model/results/drspaam_cutout_test_metrics.npz"
    inspect_npz(file_path)
