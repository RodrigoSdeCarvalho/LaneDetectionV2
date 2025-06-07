import os
import re

def escape_latex_texttt(text):
    # Escape \, {, }, and _ for \texttt{}
    text = text.replace('\\', r'\textbackslash{}')
    text = text.replace('{', r'\{')
    text = text.replace('}', r'\}')
    text = text.replace('_', r'\_')  # Add underscore escaping
    return text

def generate_latex_sections():
    # Get the current working directory
    current_dir = os.getcwd()
    
    # Create output file
    with open('python_sections.txt', 'w', encoding='utf-8') as outfile:
        # Walk through all directories and files
        for root, dirs, files in os.walk(current_dir):
            # Skip __pycache__ directories
            if '__pycache__' in root:
                continue
                
            # Sort files to ensure consistent order
            for file in sorted(files):
                if file.endswith('.py') and file != 'generate_latex_sections.py':
                    file_path = os.path.join(root, file)
                    # Get relative path from current directory (with / as separator)
                    rel_path = os.path.relpath(file_path, current_dir).replace('\\', '/')
                    # Escape \, {, }, and _ for \texttt{}
                    section_name = escape_latex_texttt(rel_path)
                    
                    outfile.write(f'{{\\texttt{{{section_name}}}}}\n')
                    outfile.write('\\begin{minted}[\n')
                    outfile.write('    frame=lines,\n')
                    outfile.write('    framesep=2mm,\n')
                    outfile.write('    baselinestretch=1.2,\n')
                    outfile.write('    breaklines=True,\n')
                    outfile.write('    fontsize=\\footnotesize,\n')
                    outfile.write('    linenos\n')
                    outfile.write('    ]{python}\n\n')
                    
                    # Read and write file contents
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                            outfile.write(content)
                            outfile.write('\n\n')
                    except Exception as e:
                        print(f"Error reading {file_path}: {str(e)}")
                    
                    # Close minted and section environments
                    outfile.write('\\end{minted}\n\n')

if __name__ == '__main__':
    generate_latex_sections()
    print("LaTeX sections have been generated in python_sections.txt")
