import os
import csv
from concurrent.futures import ProcessPoolExecutor

def process_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        for line in infile:
            row = line.strip().split("  ")
            writer.writerow(row)

def convert_txt_to_csv_parallel(input_root, output_root, max_workers=4):
    tasks = []
    os.makedirs(output_root, exist_ok=True)
    
    for root, _, files in os.walk(input_root):
        relative_path = os.path.relpath(root, input_root)
        output_dir = os.path.join(output_root, relative_path)
        os.makedirs(output_dir, exist_ok=True)
        
        for file in files:
            if file.endswith('.txt'):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_dir, file.replace('.txt', '.csv'))
                tasks.append((input_file_path, output_file_path))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, input_path, output_path) for input_path, output_path in tasks]
        for future in futures:
            future.result()  # Wait for all tasks to complete

    print("Parallel conversion completed successfully.")

if __name__ == "__main__":
    input_root = './data/BenchmarkDatasets'
    output_root = './data/BenchmarkDatasets_csv'
    convert_txt_to_csv_parallel(input_root, output_root, max_workers=8)
