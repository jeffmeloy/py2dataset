"""
Obtain data parameter and model from the py2dataset functions.
Requirements:
[req01] The get_default_questions function shall return a list of default
        questions that consists of a dictionary containing keys: id, text, and
        type.
[req02] The get_default_model_config function shall return a dictionary
        representing the default model configuration.
[req03] The get_output_dir function shall validate the provided output_dir
        argument and return its absolute path.
[req04] The get_output_dir function shall return the default OUTPUT_DIR if the
        provided output_dir argument is not provided or invalid.
[req05] The get_questions function shall validate the question file provided by
        the questions_pathname argument and return those questions.
[req06] The get_questions function shall return default questions if the
        questions_pathname is not provided or invalid.
[req07] The instantiate_model function shall import the specified module and
        class, then instantiate the model using the provided configuration.
[req08] The get_model function shall validate the model config file provided by
        the model_config_pathname argument and return an instantiated model and
        a prompt template based on that configuration.
[req09] The get_model function shall return an instantiated model and a prompt
        template based on the default model configuration if the 
        model_config_pathname is not provided or invalid.
[req10] The write_questions_file function shall write the default questions to
        the QUESTIONS_FILE in the directory specified
[req11] The write_questions_file function shall write the default questions to
        the QUESTIONS_FILE in the current working directory if the output_dir
        argument is not provided or invalid.
[req12] The write_model_config_file function shall write the default model 
        configuration to the MODEL_CONFIG_FILE in the directory specified.
[req13] The write_model_config_file function shall write the default model
        configuration to the MODEL_CONFIG_FILE in the current working directory
        if the output_dir argument is not provided or invalid.
"""

import os
import json
import logging
import yaml
import importlib
from typing import Dict, List 
from pathlib import Path

# Setting up a basic logger
logging.basicConfig(level=logging.INFO)

QUESTIONS_FILE = 'py2dataset_questions.json'
MODEL_CONFIG_FILE = 'py2dataset_model_config.yaml'
OUTPUT_DIR = 'datasets'


def get_default_questions() -> List[Dict]:
    """Returns default question list"""
    questions = [
        {
            "id": "file_dependencies",
            "text": "Dependencies of file: ({filename})?",
            "type": "file"
        },
        {
            "id": "internal_code_graph",
            "text": "Structural graph of the relationships between the functions and classes defined in file: ({filename})?",
            "type": "file"
        },
        {
            "id": "entire_code_graph",
            "text": "Structural graph of the relationships between the functions and classes defined and used in file: ({filename})?",
            "type": "file"
        },
        {
            "id": "file_functions",
            "text": "Funtions in file: ({filename})?",
            "type": "file"
        },      
        {
            "id": "file_classes",
            "text": "Classes in file: ({filename})?",
            "type": "file"
        },
        {
            "id": "file_control_flow",
            "text": "Control Flow in file: ({filename})?",
            "type": "file"
        },
        {
            "id": "function_inputs",
            "text": "Inputs to function: ({function_name}) in file: ({filename})?",
            "type": "function"
        },
        {
            "id": "function_docstring",
            "text": "Docstring of function: ({function_name}) in file: ({filename})?",
            "type": "function"
        },
        {
            "id": "function_calls",
            "text": "Calls in function: ({function_name}) in file: ({filename})?",
            "type": "function"
        },
        {
            "id": "function_variables",
            "text": "Variables in function: ({function_name}) in file: ({filename})?",
            "type": "function"
        }, 
        {
            "id": "function_returns",
            "text": "Returns from function: ({function_name}) in file: ({filename})?",
            "type": "function"
        },
        {
            "id": "class_methods",
            "text": "Methods in class: ({class_name}) in file: ({filename})?",
            "type": "class"
        },
        {
            "id": "class_docstring",
            "text": "Docstring of class: ({class_name}) in file: ({filename})?",
            "type": "class"
        },
        {
            "id": "class_attributes",
            "text": "Attributes of class: ({class_name}) in file: ({filename})?",
            "type": "class"
        },
        {
            "id": "class_variables",
            "text": "Variables in class: ({class_name}) in file: ({filename})?",
            "type": "class"
        },
        {
            "id": "class_inheritance",
            "text": "Inheritance of class: ({class_name}) in file: ({filename})?",
            "type": "class"
        },
        {
            "id": "method_inputs",
            "text": "Inputs to method: ({method_name}) in class: ({class_name}) in file: ({filename})?",
            "type": "method"
        },
        {
            "id": "method_docstring",
            "text": "Docstring of method: ({method_name}) in class: ({class_name}) in file: ({filename})?",
            "type": "method"
        },
        {
            "id": "method_calls",
            "text": "Calls in method: ({method_name}) in class: ({class_name}) in file: ({filename})?",
            "type": "method"
        },
        {
            "id": "method_returns",
            "text": "Returns from method: ({method_name}) in class: ({class_name}) in file: ({filename})?",
            "type": "method"
        },
        {   
            "id": "file_purpose",
            "text": "Purpose of file: ({filename})?",
            "type": "file"
        },
        {
            "id": "function_purpose",
            "text": "Purpose of function: ({function_name}) in file: ({filename})?",
            "type": "function"
        },
        {
            "id": "class_purpose",
            "text": "Purpose of class: ({class_name}) in file: ({filename})?",
            "type": "class"
        },
        {
            "id": "method_purpose",
            "text": "Purpose of method: ({method_name}) in class: ({class_name}) in file: ({filename})?",
            "type": "method"
        },
        {
            "id": "function_variable_purpose",
            "text": "Purpose of variable: ({function_variable}) in function: ({function_name}) in file: ({filename})?",
            "type": "function"
        },       
        {
            "id": "class_variable_purpose",
            "text": "Purpose of variable: ({class_variable}) in class: ({class_name}) in file: ({filename})?",
            "type": "class"
        }
    ]
    return questions


def get_default_model_config() -> Dict:
    """Returns default model config dict"""
    model_config = {
        "prompt_template": "Provide a concise and comprehensive Response to the Instruction considering the given Context and include your reasoning. \n### Context:\n{context}\n### Instruction:\n{query}\n### Response:",
        "inference_model": {
            "model_import_path": "ctransformers.AutoModelForCausalLM",
            "model_params": {
                "model_path": "TheBloke/WizardCoder-Guanaco-15B-V1.1-GGML",  
                "model_type": "starcoder",
                "local_files_only": False,
                "lib": "avx2",
                "threads": 30,
                "batch_size": 32,
                "max_new_tokens": 2048,
                "gpu_layers": 24,
                "reset": True
            }
        }
    }
    return model_config


def get_output_dir(output_dir: str='') -> str:
    """Returns the appropriate output directory."""   
    if output_dir: # Check if the directory exists and create it if not
        output_dir = os.path.abspath(output_dir)
    else: # Default to OUTPUT_DIR at the current working directory
        output_dir = os.path.join(os.getcwd(), OUTPUT_DIR)
    if not Path(output_dir).is_dir(): #create output_dir if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    logging.info(f'Using output directory: {output_dir}')
    return output_dir


def get_questions(questions_pathname: str) -> List[Dict]:
    """Get questions from file or default"""
    # check if questions_pathname is an empty string
    if not questions_pathname:
        questions_pathname = os.path.join(os.getcwd(), QUESTIONS_FILE)    

    # verify if questions_pathname is a valid file
    if not Path(questions_pathname).is_file():
        logging.info(f'Questions file {questions_pathname} not found. Using default questions')
        questions = get_default_questions()
        return questions

    # verify if questions_pathname is a valid json questions file
    try:
        with open(questions_pathname, 'r') as f:
            questions = json.load(f)
    except:
        logging.info(f'Questions file not valid: {questions_pathname} Using default questions')
        questions = get_default_questions()
        return questions  

    # check for 'id', 'text', and 'type' keys in each question
    for question in questions:
        if not all(key in question for key in ['id', 'text', 'type']):
            logging.info(f'Questions file not valid: {questions_pathname} Using default questions')
            questions = get_default_questions()
            return questions 

    logging.info(f'Using questions from file: {questions_pathname}')
    return questions


def instantiate_model(model_config: Dict) -> object:
    """
    Imports and instantiates a model based on the provided configuration.
    Args:
        model_config (dict): A dictionary containing the configuration for the
            model. It should include the import path for the model class and
            parameters for instantiation.
        user_config (dict): A dictionary containing user-provided configurations.
            If provided, these configurations will override the defaults.
    Returns:
        object: An instance of the specified model class, or None if there was
            an error.
    """
    model = None
    try:
        module_name, class_name = model_config['model_import_path'].rsplit('.', 1)
        module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Failed to import module {module_name}. Error: {e}")
        return model
    try:
        ModelClass = getattr(module, class_name)
    except AttributeError as e:
        print(f"Module {module_name} does not have a class named {class_name}. Error: {e}")
        return model
    
    model_params = model_config['model_params']
    try:
        model = ModelClass.from_pretrained(model_params.pop('model_path'), **model_params)
    except Exception as e:
        print(f"Failed to instantiate the model with the provided parameters. Error: {e}")
        return model

    return model


def get_model(model_config_pathname: str):
    """
    Agrs:
        model_config_pathname (str): The pathname of the model config file
    Returns:
        Tuple[object, str]: The instantiated model and prompt template 
    """
    # check if model_config_pathname is an empty string
    if not model_config_pathname:
        model_config_pathname = os.path.join(os.getcwd(), MODEL_CONFIG_FILE)
    
    # verify if model_config_pathname is a valid file
    if not Path(model_config_pathname).is_file():
        logging.info(f'Model config file not found: {model_config_pathname} Using default model config')
        model_config = get_default_model_config()
        return instantiate_model(model_config['inference_model']), model_config['prompt_template']

     # verify if model_config_pathname is a valid json model config file
    try:
        # Load model configuration from the YAML file
        with open(model_config_pathname, 'r') as config_file:
            model_config = yaml.safe_load(config_file)
    except:
        logging.info(f'Model config file not valid: {model_config_pathname} Using default model config')
        model_config = get_default_model_config()
        return instantiate_model(model_config['inference_model']), model_config['prompt_template']

    logging.info(f'Using model config from file: {model_config_pathname}')
    return instantiate_model(model_config['inference_model']), model_config['prompt_template']


def write_questions_file(output_dir: str='') -> None:
    """
    Writes the default questions to a file in JSON format.
    """
    questions = get_default_questions()
    if not output_dir or not Path(output_dir).is_dir():
        output_dir = os.getcwd()
    with open(os.path.join(output_dir, QUESTIONS_FILE), 'w') as file:
        json.dump(questions, file, indent=4)


def write_model_config_file(output_dir: str='') -> None:
    """
    Writes the default model config to a file in YAML format.
    """
    model_config = get_default_model_config()
    if not output_dir or not Path(output_dir).is_dir():
        output_dir = os.getcwd()
    with open(os.path.join(output_dir, MODEL_CONFIG_FILE), 'w') as file:
        yaml.dump(model_config, file)
