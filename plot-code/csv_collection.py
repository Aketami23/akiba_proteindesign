import os
import shutil

SOURCE_DIR = 'proteinMPNN'
DEST_DIR = 'Replot/data'

def find_and_copy_with_new_name(source_dir, dest_dir):
    print("Searching for CSV files in:", source_dir)
    print("Destination directory:", dest_dir)

    copied_files_count = 1

    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            if filename.lower().endswith('.csv'):
                source_path = os.path.join(root, filename)
                print("Found CSV file:", source_path)
                dest_filename = f"seed{copied_files_count}.csv"
                dest_path = os.path.join(dest_dir, dest_filename)

                shutil.copy2(source_path, dest_path)
                print(f"Copied {source_path} -> {dest_path}")
                copied_files_count += 1


find_and_copy_with_new_name(SOURCE_DIR, DEST_DIR)