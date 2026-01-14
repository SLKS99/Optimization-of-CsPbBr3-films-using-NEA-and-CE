"""
Helper script to create the required folder structure
Run this once to set up your project directories
"""

import os
import config


def create_folder_structure():
    """Create the required folder structure for Dual GP analysis."""
    base_dir = config.BASE_DIRECTORY
    
    folders = [
        base_dir,
        os.path.join(base_dir, config.PEAK_DATA_DIR),
        os.path.join(base_dir, config.COMPOSITIONS_DIR),
        os.path.join(base_dir, config.INITIAL_DATASETS_DIR),
        os.path.join(base_dir, config.NEXT_ITERATIONS_DIR),
        os.path.join(base_dir, config.MULTIPLE_PEAKS_DIR),
        os.path.join(base_dir, config.COMBINED_SET_DIR),
        os.path.join(base_dir, config.UPDATED_DATASETS_DIR),
        os.path.join(base_dir, config.RESULTS_DIR),
    ]
    
    print("Creating folder structure...")
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"  ✓ {folder}")
    
    print("\nFolder structure created successfully!")
    print(f"\nNext steps:")
    print(f"\nFor Peak Data (Recommended):")
    print(f"1. Place your peak data files in: {os.path.join(base_dir, config.PEAK_DATA_DIR)}")
    print(f"   - Name files like: 'iteration 1 peak data.csv', 'iteration 2 peak data.csv', etc.")
    print(f"2. Place your composition file in: {os.path.join(base_dir, config.COMPOSITIONS_DIR)}")
    print(f"   - Name it: {config.COMPOSITION_FILE_NAME}")
    print(f"3. Run: python main.py (will auto-process peak data files)")
    print(f"\nFor Raw Luminescence Data:")
    print(f"1. Place your luminescence data file at: {os.path.join(base_dir, config.DATA_FILE_NAME)}")
    print(f"2. Place your composition file at: {os.path.join(base_dir, config.COMPOSITION_FILE_NAME)}")
    print(f"3. Run: python main.py")


if __name__ == "__main__":
    create_folder_structure()
