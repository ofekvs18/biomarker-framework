import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import gcsfs
import os
import pandas as pd

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(f"--- Starting Connectivity Test ---")
    
    # 1. RESOLVE KEY PATH
    # Hydra changes the working directory, so we must calculate the absolute path
    # to the key file, otherwise it won't be found.
    key_path = to_absolute_path(cfg.storage.key_path)
    print(f"Key identified at: {key_path}")
    
    if not os.path.exists(key_path):
        print(f"ERROR: Key file not found! Make sure '{cfg.storage.key_path}' is in your project folder.")
        return

    # 2. CONNECT TO GCS
    try:
        print(f"Attempting to connect to bucket: {cfg.storage.bucket}...")
        
        # We pass the key path directly to GCSFS
        fs = gcsfs.GCSFileSystem(token=key_path)
        
        # List files to prove we have access
        files = fs.ls(cfg.storage.bucket)
        print(f"SUCCESS! Connection established.")
        print(f"Found {len(files)} items in bucket.")
        
        if len(files) > 0:
            print("First 5 items:")
            for f in files[:5]:
                print(f" - {f}")
        else:
            print("Bucket is empty, but connection works!")
            
    except Exception as e:
        print(f"\nCRITICAL FAILURE: Could not connect to GCS.")
        print(f"Error details: {e}")
        return

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    main()