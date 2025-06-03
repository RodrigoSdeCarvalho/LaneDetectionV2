import os
import pathlib

def concatenate_python_files():
    # Get the current working directory
    current_dir = os.getcwd()
    
    # Create output file
    with open('all.py', 'w', encoding='utf-8') as outfile:
        # Walk through all directories and files
        for root, dirs, files in os.walk(current_dir):
            # Skip __pycache__ directories
            if '__pycache__' in root:
                continue
                
            # Sort files to ensure consistent order
            for file in sorted(files):
                if file.endswith('.py') and file != 'concatenate_py_files.py':
                    file_path = os.path.join(root, file)
                    # Get relative path from current directory
                    rel_path = os.path.relpath(file_path, current_dir)
                    
                    # Write file path as comment
                    outfile.write(f'\n# Source: {rel_path}\n')
                    outfile.write('#' + '='*80 + '\n\n')
                    
                    # Read and write file contents
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                            outfile.write(content)
                            outfile.write('\n\n')
                    except Exception as e:
                        print(f"Error reading {file_path}: {str(e)}")

if __name__ == '__main__':
    concatenate_python_files()
    print("All Python files have been concatenated into all.py")
