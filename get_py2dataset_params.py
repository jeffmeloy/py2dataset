"""
Obtain data parameter and model from the py2dataset functions.
Requirements:
[req01] The `get_default_questions` function shall:
        a. Return a list of default questions.
        b. Ensure each question in the list is a dictionary.
        c. Ensure each dictionary has the keys: id, text, and type.
[req02] The `get_default_model_config` function shall:
        a. Return a dictionary representing the default model configuration.
[req03] The `get_output_dir` function shall:
        a. Accept an optional output_dir argument.
        b. Return the absolute path of the provided output_dir if it exists or can be created.
        c. Return the default OUTPUT_DIR if the provided output_dir argument is not provided or invalid.
[req04] The `get_questions` function shall:
        a. Accept an optional questions_pathname argument.
        b. Validate the question file provided by the questions_pathname.
        c. Return the questions from the provided questions_pathname if valid.
        d. Return default questions if the questions_pathname is not provided or invalid.
[req05] The `instantiate_model` function shall:
        a. Accept a model_config dictionary as an argument.
        b. Import the specified module and class from the model_config.
        c. Instantiate and return the model using the provided configuration.
[req06] The `get_model` function shall:
        a. Accept an optional model_config_pathname argument.
        b. Validate the model config file provided by the model_config_pathname.
        c. Return an instantiated model and a prompt template based on the provided configuration.
        d. Return an instantiated model and a prompt template based on the default model configuration if the model_config_pathname is not provided or invalid.
[req07] The `write_questions_file` function shall:
        a. Accept an optional output_dir argument.
        b. Write the default questions to the QUESTIONS_FILE in the specified directory.
        c. Write the default questions to the QUESTIONS_FILE in the current working directory if the output_dir argument is not provided or invalid.
[req08] The `write_model_config_file` function shall:
        a. Accept an optional output_dir argument.
        b. Write the default model configuration to the MODEL_CONFIG_FILE in the specified directory.
        c. Write the default model configuration to the MODEL_CONFIG_FILE in the current working directory if the output_dir argument is not provided or invalid.
"""
import os
import json
import logging
import yaml
import importlib
from typing import Dict, List 
from pathlib import Path
from transformers import AutoTokenizer

# Setting up a basic logger
logging.basicConfig(level=logging.INFO)

QUESTIONS_FILE = 'py2dataset_questions.json'
MODEL_CONFIG_FILE = 'py2dataset_model_config.yaml'
OUTPUT_DIR = 'datasets'

def get_default_questions() -> List[Dict]:
    """Return default question list
    Args:
        None
    Returns:
        List[Dict]: The default question list
    """
    questions = [
        {
            "id": "file_dependencies",
            "text": "Dependencies of Python file: '{filename}'?",
            "type": "file"
        },
        {
            "id": "entire_code_graph",
            "text": "Call code graph of Python file: '{filename}'?",
            "type": "file"
        },
        {
            "id": "file_functions",
            "text": "Functions defined in Python file: '{filename}'?",
            "type": "file"
        },      
        {
            "id": "file_classes",
            "text": "Classes defined in Python file: '{filename}'?",
            "type": "file"
        },
        {
            "id": "function_inputs",
            "text": "Inputs to function: '{function_name}' in Python file: '{filename}'?",
            "type": "function"
        },
        {
            "id": "function_docstring",
            "text": "Docstring of function: '{function_name}' in Python file: '{filename}'?",
            "type": "function"
        },
        {
            "id": "function_calls",
            "text": "Calls made in function: '{function_name}' in Python file: '{filename}'?",
            "type": "function"
        },
        {
            "id": "function_variables",
            "text": "Variables defined in function: '{function_name}' in Python file: '{filename}'?",
            "type": "function"
        }, 
        {
            "id": "function_returns",
            "text": "Returned items from function: '{function_name}' in Python file: '{filename}'?",
            "type": "function"
        },
        {
            "id": "class_methods",
            "text": "Methods defined in class: '{class_name}' in Python file: '{filename}'?",
            "type": "class"
        },
        {
            "id": "class_docstring",
            "text": "Docstring of class: '{class_name}' in Python file: '{filename}'?",
            "type": "class"
        },
        {
            "id": "class_attributes",
            "text": "Attributes of class: '{class_name}' in Python file: '{filename}'?",
            "type": "class"
        },
        {
            "id": "class_variables",
            "text": "Variables defined in class: '{class_name}' in Python file: '{filename}'?",
            "type": "class"
        },
        {
            "id": "class_inheritance",
            "text": "Inheritance of class: '{class_name}' in Python file: '{filename}'?",
            "type": "class"
        },
        {
            "id": "method_inputs",
            "text": "Inputs to method: '{method_name}' in class: '{class_name}' in Python file: '{filename}'?",
            "type": "method"
        },
        {
            "id": "method_docstring",
            "text": "Docstring of method: '{method_name}' in class: '{class_name}' in Python file: '{filename}'?",
            "type": "method"
        },
        {
            "id": "method_calls",
            "text": "Calls made in method: '{method_name}' in class: '{class_name}' in Python file: '{filename}'?",
            "type": "method"
        },
        {
            "id": "method_returns",
            "text": "Returns from method: '{method_name}' in class: '{class_name}' in Python file: '{filename}'?",
            "type": "method"
        },
        {   
            "id": "file_purpose",
            "text": "1) DESCRIBE the purpose and processing summary of Python file: '{filename}'; 2) PROVIDE an itemized and detailed description of each applicable function, class, and method; 3) EXPLAIN what each input, output, and variable does in the code.",
            "type": "file"
        } 
    ]
    return questions


def get_default_model_config() -> Dict:
    """Return default model config dict
    Args:
        None
    Returns:
        Dict: The default model config dictionary
    """
    model_config = {
        "prompt_template": "\n### Instruction:\nGiven this context:\n'{context}'\nPlease analyze this code you created provide a comprehensive response without duplicating the input code, include enough detail for me to implement the same logic, and include your reasoning step by step: {query}\n### Response:",
        "inference_model": {
            "model_import_path": "ctransformers.AutoModelForCausalLM",
            "model_inference_function": "from_pretrained",
            "model_params": {
                "model_path": "TheBloke/WizardCoder-Python-13B-V1.0-GGUF",  
                "model_type": "llama",
                "local_files_only": False,
                ## MODEL CONFIGURATION PARAMETERS (GPU 4090 - 24GB VRAM, CPU 5950x - 32 threads, 64GB RAM)
                #avx2 and gpu_layers are not compatible 
                #"lib": "avx2",
                "threads": 28,
                "batch_size": 128,
                "context_length": 14000,
                "max_new_tokens": 8092,
                "gpu_layers": 100,
                "reset": True
                }
            }
        }
    return model_config


def get_output_dir(output_dir: str='') -> str:
    """Returns the appropriate output directory.
    Args:
        output_dir (str): The directory to write the output to.
    Returns:
        str: The absolute path of the provided output_dir if it exists or can be created.
    """
    output_dir = os.path.abspath(output_dir or OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f'Using output directory: {output_dir}')
    return output_dir


def get_questions(questions_pathname: str) -> List[Dict]:
    """
    Get questions from file or default
    Args:
        questions_pathname (str): The pathname of the questions file
    Returns:
        List[Dict]: The list of questions
    """
    try: # get questions from provided or default configuration file
        if not questions_pathname:
            questions_pathname = os.path.join(os.getcwd(), QUESTIONS_FILE)
        with open(questions_pathname, 'r') as f:
            questions = json.load(f)
        logging.info(f'Using questions from file: {questions_pathname}')
    except: # get default questions if can't read questions_pathname file 
        logging.info(f'Questions file not valid: {questions_pathname} Using default questions')
        questions = get_default_questions()
    return questions


def instantiate_model(model_config: Dict) -> object:
    """
    Imports and instantiates a model based on the provided configuration.
    Args:
        model_config (dict): model configuration dictionary.
    Returns:
        object: An instance of the specified model class, or None if error.
    """
    try:
        module_name, class_name = model_config['model_import_path'].rsplit('.', 1)
        ModelClass = getattr(importlib.import_module(module_name), class_name)
        model_params = model_config['model_params']
        model_path = model_params['model_path']
        inference_function_name = model_config['model_inference_function']
        if inference_function_name != "": 
            inference_function = getattr(ModelClass, inference_function_name)
            model = inference_function(model_params.pop('model_path'), **model_params)
        else: 
            model = ModelClass(model_params.pop('model_path'), **model_params)
        return model
    except ImportError or AttributeError or Exception as e:
        logging.info(f"Failed to instantiate the model. Error: {e}")    
        return None, None


def get_model(model_config_pathname: str) -> tuple[object, str]:
    """
    Returns an instantiated model and prompt template based on the model configuration.
    Agrs:
        model_config_pathname (str): The pathname of the model config file
    Returns:
        Tuple[object, str]: The instantiated model and prompt template 
    """
    try:
        if not model_config_pathname:
            model_config_pathname = os.path.join(os.getcwd(), MODEL_CONFIG_FILE)
        with open(model_config_pathname, 'r') as config_file:
            model_config = yaml.safe_load(config_file)
        logging.info(f'Using model config from file: {model_config_pathname}')
    except:
        logging.info(f'Model config file not valid: {model_config_pathname} Using default model config')
        model_config = get_default_model_config()
    model_config['model'] = instantiate_model(model_config['inference_model'])
    return model_config


def write_questions_file(output_dir: str='') -> None:
    """
    Writes the default questions to a file in JSON format.
    Args:
        output_dir (str): The directory to write the questions file to.
    Returns:
        None
    """
    questions = get_default_questions()
    output_dir = output_dir if output_dir and Path(output_dir).is_dir() else os.getcwd()
    with open(os.path.join(output_dir, QUESTIONS_FILE), 'w') as file:
        json.dump(questions, file, indent=4)


def write_model_config_file(output_dir: str='') -> None:
    """
    Writes the default model config to a file in YAML format.
    Args:
        output_dir (str): The directory to write the model config file to.
    Returns:
        None
    """
    model_config = get_default_model_config()
    output_dir = output_dir if output_dir and Path(output_dir).is_dir() else os.getcwd()
    with open(os.path.join(output_dir, MODEL_CONFIG_FILE), 'w') as file:
        yaml.dump(model_config, file)
