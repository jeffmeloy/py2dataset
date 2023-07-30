# py2dataset - Python to Dataset

py2dataset analyzes source code to generate structured datasets describing code content and behavior. It extracts information from Python files and converts it into JSON-formatted datasets. These datasets can help users understand Python software or train AI systems.

First, py2dataset obtains the questions from the 'questions.json' file and Python files from the provided code directory argument. 

Next, py2dataset parses the Python files using the **Abstract Syntax Tree (AST)** and **visitor design pattern** to build a representation of the code structure, functions, classes, methods and variables.

py2dataset then generates the information for the output datasets including question-answer pairs and instruction-input-output triplets using code snippets as context. The datasets answer the 'questions.json' questions about inputs, variables, returns, inheritance and other Python file characteristics. Optionally, py2dataset invokes a **language model** to generate responses to questions about the purpose of each file, function, class, method, and variable.

For each Python code file analyzed, py2dataset outputs a `<filename>.qa.json`, `<filename>.instruct.json`, and a `<filename>.details.yaml` containing the AST analysis to the local `'./dataset'` directory or a specified output directory. The software then consolidates all of the datasets together to produce a final `'qa.json'` and `'instruct.json'` that contains the entire dataset contents. Optionally, py2dataset also creates **relationship graphs** between functions and classes as image files.

With its AST parser, visitor pattern and configurable questions, py2dataset flexibly and extensibly analyzes source code to generate descriptive datasets. Its modular architecture and multiple formats support diverse use cases for understanding and learning Python code.

![py2dataset Diagram](py2dataset.png)

## Overview

The main script is `py2dataset.py`. It analyzes Python source code files in a given directory using the `get_python_file_details.py` module. The `get_python_json.py` module is used to generate the question-answer pairs and instructions, using a large language model if enabled by the `--use_llm` option. The large language model is used to answer the questions related to the purpose of the file, functions, classes and variables.

**Imported Libraries:** os, argparse, logging, pathlib, yaml

**Optional Keywords:**

- `--use_llm` - Use large language model for question answering
- `--quiet` - Suppress info logging messages
- `--output_dir` - Output directory to store generated files
- `--graph` - Generate code relationship graphs

## Installation

You can install py2dataset using pip:

    pip install py2dataset

## Usage

### Command Line Interface

The `py2dataset.py` script can be used from the command line with the following arguments:

    python -m py2dataset [-h] [--use_llm] [--quiet] [--output_dir OUTPUT_DIR] [--graph] directory

**Positional arguments:**
- `directory`: The directory containing the Python files to analyze.

**Optional arguments:**
- `-h, --help`: Show the help message and exit.
- `--use_llm`: Use large language model for question answering.
- `--quiet`: Suppress all info logging messages.
- `--output_dir OUTPUT_DIR`: Output directory to store generated files.
- `--graph`: Generate code relationship graphs.

**Example usage:**

    python -m py2dataset ../my_python_code --use_llm --output_dir ./dataset --graph

This will analyze the Python files in `../my_python_code`, using the large language model to generate answers, write the output files to `./dataset`, and generate code relationship graphs.

Without any arguments, the script will prompt for a directory and write output to `./datasets`.

### Python API

You can also use py2dataset directly in your Python code:

```python
from py2dataset import py2dataset

# Analyze Python files in the directory "../my_python_code"
py2dataset("../my_python_code", use_llm=True, graph=True, output_dir="./dataset")

This will have the same effect as the command line example above.

## Code Structure

- `py2dataset.py` - Main script
- `get_python_file_details.py` - Extracts details from Python files using AST
- `get_python_json.py` - Generates question-answer pairs and instructions
- `questions.json` - Standard questions for Python files, functions, classes
- `model_config.yaml` - Configuration for large language model

## Output

The script generates the following output:

- `<filename>.details.yaml` - Python file details YAML file
- `<filename>.qa.json` - Question-answer pairs JSON file
- `<filename>.instruct.json` - Instructions JSON file
- `qa.json` - Combined question-answer JSON file
- `instruct.json` - Combined instructions JSON file
- Code relationship graphs (optional)

## Requirements

- Python >= 3.8
- **networkx** module (optional for graphs)
- **ctransformers** library for large language model support
- **yaml** library for configuration and output files
