# py2dataset - Python to Dataset

py2dataset analyzes source code to generate structured datasets describing code content and behavior. It extracts information from Python files and converts it into JSON-formatted datasets. These datasets can help you understand Python software or train AI systems.

![py2dataset Diagram](py2dataset.png)

## Overview

py2dataset flexibly and extensibly analyzes source code to generate descriptive datasets for understanding and learning Python code.

py2dataset performs the following functions:

- Obtain questions from the `py2dataset_questions.json` file or from the questions filename provided in the command line
- Obtain a file listing of all of Python files within the provided directory and its subdirectories
- Parse each Python file using the **Abstract Syntax Tree (AST)** and **visitor design pattern** to build a representation of the code structure, functions, classes, methods and variables
- Generate information for the output datasets including question-answer pairs and instruction-input-output triplets using code snippets as context to answer the questions about the Python file characteristics
- (Optional) Invoke a **language model** to generate responses to questions about the purpose of each file, function, class, method, and variable
- Output `<filename>.qa.json` and `<filename>.instruct.json` datasets, and a `<filename>.details.yaml` containing the AST analysis to the local `./dataset` directory or the command line specified output directory 
- (Optional) Create images of the **relationship graphs** between the functions and classes and save as image files in the same output directory
- Combine the datasets for all individual Python files together to produce a final `qa.json` and `instruct.json` in the same output directory

## Installation 

### From Source

Clone the repository and install from source:

    ```bash
    git clone https://github.com/jeffmeloy/py2dataset.git
    ```

Then install dependencies to use the command line interface:

    ```bash 
    pip install -r requirements.txt 
    ```

## Usage

### Command Line Interface

**Example usage:**
    
    ```bash
    python py2dataset.py ../my_python_code 
    ```
**Positional arguments:**
- `directory`: The directory containing the Python (*.py) files to analyze.

Without any arguments, the script will prompt for a directory and write output to `./datasets`.

**Optional arguments:**
- `--use_llm`: Use large language model for question answering.
- `--use_summary`: Use code summary (imports, function, class, method definitions) instead of code to reduce context length in instruct JSON files
- `--quiet`: Suppress all info logging messages.
- `--output_dir OUTPUT_DIR`: Output directory to store generated files, default is .\datasets in the current working directory
- `--graph`: Generate code relationship graphs.
- `--model_config` - Specify a model configuration file, default is model_config.yaml
 
## Questions for datasets

The following questions are answered by parsing the AST:
- Dependencies of file: ({filename})?,
- Structural graph of the relationships between the functions and classes defined in file: ({filename})?
- Structural graph of the relationships between the functions and classes defined and used in file: ({filename})?
- Functions in file: ({filename})?
- Classes in file: ({filename})?
- Control Flow in file: ({filename})?
- Inputs to function: ({function_name}) in file: ({filename})?
- Docstring of function: ({function_name}) in file: ({filename})?
- Calls in function: ({function_name}) in file: ({filename})?
- Variables in function: ({function_name}) in file: ({filename})?
- Returns from function: ({function_name}) in file: ({filename})?
- Methods in class: ({class_name}) in file: ({filename})?
- Docstring of class: ({class_name}) in file: ({filename})?
- Attributes of class: ({class_name}) in file: ({filename})?
- Variables in class: ({class_name}) in file: ({filename})?
- Inheritance of class: ({class_name}) in file: ({filename})?
- Inputs to method: ({method_name}) in class: ({class_name}) in file: ({filename})?
- Docstring of method: ({method_name}) in class: ({class_name}) in file: ({filename})?
- Calls in method: ({method_name}) in class: ({class_name}) in file: ({filename})?
- Returns from method: ({method_name}) in class: ({class_name}) in file: ({filename})?

The following questions are answered using a language model if --use_llm: 
- Purpose of file: ({filename})?
- Purpose of function: ({function_name}) in file: ({filename})?
- Purpose of class: ({class_name}) in file: ({filename})?
- Purpose of method: ({method_name}) in class: ({class_name}) in file: ({filename})?
- Purpose of variable: ({function_variable}) in function: ({function_name}) in file: ({filename})?
- Purpose of variable: ({class_variable}) in class: ({class_name}) in file: ({filename})?

## Code Structure

- `py2dataset.py` - Main script
- `get_python_file_details.py` - Extracts details from Python files using AST
- `get_python_datasets.py` - Generates question-answer pairs and instructions
- `py2dataset_questions.json` - Standard questions for Python files, functions, classes
- `py2dataset_model_config.yaml` - Configuration for language model
    
## Language Model 

Currently configured to use [ctransformers](https://github.com/marella/ctransformers) with the default configuration defined in py2dataset_model_config.yaml

    ```yaml
    prompt_template: "Provide a concise and comprehensive Response to the Instruction considering the given Context and include your reasoning. \n### Context:\n{context}\n### Instruction:\n{query}\n### Response:"
    inference_model:
      model_import_path: "ctransformers.AutoModelForCausalLM"
      model_params:
        model_path: "TheBloke/Starcoderplus-Guanaco-GPT4-15B-V1.0-GGML"
        model_type: "starcoder"
        local_files_only: false
        lib: "avx2"
        threads: 30
        max_new_tokens: 2048
    ```

## Output

The script generates the following output:

- `<filename>.details.yaml` - Python file details YAML file
- `<filename>.qa.json` - Question-answer pairs JSON file
- `<filename>.instruct.json` - Instructions JSON file
- `qa.json` - Combined question-answer JSON file
- `instruct.json` - Combined instructions JSON file
- `instruct_cleaned.json` - Replaces duplicated code elements by an empty string
- `<filename>.internal_code_graph.png` - Code relationship graph (optional)
- `<filename>.entire_code_graph.png` - Code relationship graph (optional)

If an output directory is not specified, the files will be saved in a ./datasets directory within the current working directory. If this directory does not exist, it will be created.

The ./example_datasets directory provided contains the py2dataset output generated on itself. 
    
    > python .\py2dataset.py ..\ --graph --use_summary --use_llm

## Requirements

- Python >= 3.8
- **networkx** (optional for graphs)
- **ctransformers** library for large language model support
- **yaml** library for configuration and output files
- **matplotlib** (optional for graphs)