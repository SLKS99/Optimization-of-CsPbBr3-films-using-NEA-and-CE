"""
Standalone script to process a peak data file and create initial datasets
Run this once to convert your peak fitting CSV into the required format
"""

import os
import sys
import config
from peak_data_processor import process_peak_data_file


def main():
    """Process peak data file and create initial datasets."""
    print("=" * 60)
    print("Peak Data Processor")
    print("=" * 60)
    
    # Get file paths
    if os.path.isabs(config.BASE_DIRECTORY):
        base_path = config.BASE_DIRECTORY
    else:
        base_path = os.path.join(os.getcwd(), config.BASE_DIRECTORY)
    
    # Check if peak data file is specified
    if not config.PEAK_DATA_FILE_NAME:
        print("\nError: PEAK_DATA_FILE_NAME not specified in config.py")
        print("Please set PEAK_DATA_FILE_NAME to your peak data CSV file.")
        return
    
    peak_file_path = os.path.join(base_path, config.PEAK_DATA_FILE_NAME)
    composition_file_path = os.path.join(base_path, config.COMPOSITION_FILE_NAME)
    
    # Check if files exist
    if not os.path.exists(peak_file_path):
        print(f"\nError: Peak data file not found: {peak_file_path}")
        print("Please check the file path in config.py")
        return
    
    if not os.path.exists(composition_file_path):
        print(f"\nError: Composition file not found: {composition_file_path}")
        print("Please check the file path in config.py")
        return
    
    # Create output directory
    initial_datasets_dir = os.path.join(base_path, config.INITIAL_DATASETS_DIR)
    os.makedirs(initial_datasets_dir, exist_ok=True)
    
    # Process the file
    try:
        intensity_file, peak_file, multiple_peaks_file = process_peak_data_file(
            peak_file_path,
            composition_file_path,
            initial_datasets_dir,
            config.WELLS_TO_IGNORE,
            config.INITIAL_READ,
            config.FINAL_READ,
            config.COMPOSITION_COLUMNS
        )
        
        print("\n" + "=" * 60)
        print("Processing Complete!")
        print("=" * 60)
        print(f"\nCreated files:")
        print(f"  - {intensity_file}")
        print(f"  - {peak_file}")
        print(f"  - {multiple_peaks_file}")
        print(f"\nYou can now run: python main.py")
        
    except Exception as e:
        print(f"\nError processing file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
