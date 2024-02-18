# py2dataset - Python to Dataset

py2dataset analyzes source code to generate structured datasets describing code content and behavior. It extracts information from Python files and converts it into JSON-formatted datasets. These datasets can help you understand Python software or train AI systems.

![py2dataset Diagram](py2dataset.png)

## Overview

py2dataset flexibly and extensibly analyzes source code to generate descriptive datasets for understanding and learning Python code.

py2dataset performs the following functions:

- Obtain questions from the `py2dataset_questions.json` file or from the questions filename provided in the command line
- Obtain a file listing of all of Python files at given local directory or github repository
- Parse each Python file using the **Abstract Syntax Tree (AST)** and **visitor design pattern** to build a representation of the code structure, functions, classes, methods and variables
- Generate information for the output datasets in instruction-input-output triplets using code snippets as context to answer the questions about the Python file characteristics
- (Optional) Invoke a **language model** to generate responses to questions about the purpose of each file, function, class, method, and variable
- Output `<filename>.instruct.json` datasets, and a `<filename>.details.yaml` containing the AST analysis to the local `./dataset` directory or the command line specified output directory 
- Create image of the **code call graph** for the functions and classes within each python file and save in the same output directory
- Combine the datasets for all individual Python files together to produce a final `instruct.json` in the same output directory

## Installation 

### From Source

Clone the repository and install dependencies:

    ```bash
    git clone https://github.com/jeffmeloy/py2dataset.git
    pip install -r requirements.txt 
    ```

If using CUDA then:
    ```bash
    pip install ctransformers[cuda]
    ```

## Usage

### Command Line Interface

**Example usage:**
    
    ```bash
    python py2dataset.py <optional arguments>
    ```

**Optional arguments:**
- `--start <starting directory>`: Starting directory or GitHub repository Python files. Default: current working directory.
- `--output_dir <output directory>`: Directory to write the output files. Default: ./dataset/.
- `--questions_pathname <question file pathname>`: path & filename of questions file. Default: ./py2dataset_questions.json.
- `--model_config_pathname <model configuration file pathname>`: Path and filename of the model configuration file. Default: ./py2dataset_model_config.yaml.
- `--use_llm`: Use llm for generating JSON answers. Default: False.
- `--quiet`: Limit logging output. Default: False.
- `--single_process`: Use a single process to process Python files. Default: False.
- `--detailed`: Include detailed analysis. Default: False.
- `--html`: Generate HTML output. Default: False.
- `--I`: Interactive mode. Default: False.
- `--skip_regen`: Skip regeneration of existing instruct.json files. Default: False.
- `--help`: Display main() docstring.

## Questions for datasets

The following questions are answered by parsing the AST:
- Dependencies of Python file: '{filename}'?
- Call code graph of Python file: '{filename}'?
- Functions defined in Python file: '{filename}'?
- Classes defined in Python file: '{filename}'?
- Inputs to function: '{function_name}' in Python file: '{filename}'?
- Docstring of function: '{function_name}' in Python file: '{filename}'?
- Calls made in function: '{function_name}' in Python file: '{filename}'?
- Variables defined in function: '{function_name}' in Python file: '{filename}'?
- Returns from function: '{function_name}' in Python file: '{filename}'?
- Methods defined in class: '{class_name}' in Python file: '{filename}'?
- Docstring of class: '{class_name}' in Python file: '{filename}'?
- Attributes of class: '{class_name}' in Python file: '{filename}'?
- Variables defined in class: '{class_name}' in Python file: '{filename}'?
- Inheritance of class: '{class_name}' in Python file: '{filename}'?
- Inputs to method: '{method_name}' in Python file: '{filename}'?
- Docstring of method: '{method_name}' in Python file: '{filename}'?
- Calls made in method: '{method_name}' in Python file: '{filename}'?
- Returns from method: '{method_name}' in Python file: '{filename}'?

If --use_llm, the dataset includes the llm response to the file_purpose question in the `--questions_pathname` file

If --use_llm and --detailed, the dataset includes the purpose and signicance of each code object

## Code Structure

- `py2dataset.py` - Main script
- `get_params.py` - Validates parameter path and file name arguments, returns questions and model
- `get_python_file_details.py` - Extracts details from Python files using AST
- `get_code_graph.py` - Obtain call code graph
- `get_python_datasets.py` - Generates question-answer pairs and instructions
- `save_output.py` - Save the output datasets, optional graph images and .html files 
- `py2dataset_questions.json` - Standard questions for Python files, functions, classes
- `py2dataset_model_config.yaml` - Configuration for language model
    
## Language Model 

Currently configured to use [ctransformers](https://github.com/marella/ctransformers) with the default configuration defined in py2dataset_model_config.yaml

    ```yaml
    prompt_template: "Provide complete structured response for a formal software audit. Given this Context:\n'{context}'\n\nPlease provide a very detailed, accurate, and insightful Response to this Instruction and include your reasoning step by step. \n### Instruction:\n{query}\n### Response:"
    inference_model:
        model_import_path: "ctransformers.AutoModelForCausalLM"
        model_inference_function: "from_pretrained"
        model_params:
            model_path: "TheBloke/WizardCoder-Python-13B-V1.0-GGUF"
            model_type: "llama"
            local_files_only: false
            threads: 16
            batch_size: 256
            context_length: 8092
            max_new_tokens: 8092
            gpu_layers: 100
            reset: true
    ```

## Output

For each Python file assessed, the script saves the following to the output directory:

- `<filename>.details.yaml` - Python file details YAML file
- `<filename>.instruct.json` - Instructions JSON file
- `<filename>.entire_code_graph.png` - Code relationship graph
- `<filename>.instruct.json.html` - Instructions JSON file in .html format (optional)

The script then creates composite datasets by combining the files above and saves the following to the output directory:

- `instruct.json` - complete instruct dataset
- `instruct.html` - html formated file (optional if --html)
- `sharegpt.json` - sharegpt dataset (optional if --use_llm)
- `sharegpt.html` - html formated file (optional if --use_llm and html)

The sharegpt.json includes a list of conversations. Each turn in a conversation has two dictionaries, a "from" field, which denotes the role of that turn, and a "value" field which contains the actual text. Here is an example of a sharegpt.json entry for each python code file:

```
    {
        "conversations": [
            {
                "from": "system",
                "value": "code documentation:" + <code doumentation created by py2dataset model>
            },
            {
                "from": "human",
                "value": Output the Python code described by the code documentation.
            },
            {
                "from": "gpt",
                "value": <python code file listing>
            }
        ],
        "nbytes": <size of conversation in bytes>
        "source": <source code path and filename>
    },
```

If an output directory is not specified, the files will be saved in a ./datasets directory within the current working directory. Directory will be created if it does not exist.

The ./example_datasets directory contains the py2dataset output generated on itself. 
    
    ```bash
    > python .\py2dataset.py --start ..\ --use_llm --detailed
    ```
## Requirements

- Python >= 3.10
- **networkx** library for defining code graphs
- **ctransformers** library for large language model support
- **PyYAML** library for configuration and output files
- **matplotlib** library for saving code graphs
- **GitPython** library for getting Python modules from github