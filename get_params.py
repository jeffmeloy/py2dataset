import os
import json
import logging
import importlib
from typing import Dict, List
from pathlib import Path
import yaml

# Setting up a basic logger
logging.basicConfig(level=logging.INFO)

# defaults if provided inputs fail
QUESTIONS_FILE = "py2dataset_questions.json"
MODEL_CONFIG_FILE = "py2dataset_model_config.yaml"
OUTPUT_DIR = "datasets"


def get_default_questions() -> List[Dict]:
    """Return default question list"""
    questions = [
        {
            "id": "file_dependencies",
            "text": "Dependencies in Python file: `{filename}`?",
            "type": "file",
        },
        {
            "id": "entire_code_graph",
            "text": "Call code graph in Python file: `{filename}`?",
            "type": "file",
        },
        {
            "id": "file_functions",
            "text": "Functions in Python file: `{filename}`?",
            "type": "file",
        },
        {
            "id": "file_classes",
            "text": "Classes in Python file: `{filename}`?",
            "type": "file",
        },
        {
            "id": "function_inputs",
            "text": "Inputs to `{function_name}` in Python file: `{filename}`?",
            "type": "function",
        },
        {
            "id": "function_docstring",
            "text": "Docstring of `{function_name}` in Python file: `{filename}`?",
            "type": "function",
        },
        {
            "id": "function_calls",
            "text": "Calls made in `{function_name}` in Python file: `{filename}`?",
            "type": "function",
        },
        {
            "id": "function_variables",
            "text": "Variables in `{function_name}` in Python file: `{filename}`?",
            "type": "function",
        },
        {
            "id": "function_returns",
            "text": "Returns from `{function_name}` in Python file: `{filename}`?",
            "type": "function",
        },
        {
            "id": "class_methods",
            "text": "Methods in `{class_name}` in Python file: `{filename}`?",
            "type": "class",
        },
        {
            "id": "class_docstring",
            "text": "Docstring of `{class_name}` in Python file: `{filename}`?",
            "type": "class",
        },
        {
            "id": "class_attributes",
            "text": "Attributes of `{class_name}` in Python file: `{filename}`?",
            "type": "class",
        },
        {
            "id": "class_inheritance",
            "text": "Inheritance of `{class_name}` in Python file: `{filename}`?",
            "type": "class",
        },
        {
            "id": "method_inputs",
            "text": "Inputs to `{method_name}` in Python file: `{filename}`?",
            "type": "method",
        },
        {
            "id": "method_docstring",
            "text": "Docstring of `{method_name}` in Python file: `{filename}`?",
            "type": "method",
        },
        {
            "id": "method_calls",
            "text": "Calls made in `{method_name}` in Python file: `{filename}`?",
            "type": "method",
        },
        {
            "id": "method_variables",
            "text": "Variables in `{method_name}` in Python file: `{filename}`?",
            "type": "method",
        },
        {
            "id": "method_returns",
            "text": "Returns from `{method_name}` in Python file: `{filename}`?",
            "type": "method",
        },
        {
            "id": "file_purpose",
            "text": "I) Describe the Purpose and Processing Approach for Python file: `{filename}`; II) Define detailed Requirements, API Signatures, and Logic for all Functions and Class Methods; III) Explain the purpose of the inputs, variables, calls, and returns in the code.",
            "type": "file",
        },
    ]
    return questions


def get_default_model_config() -> Dict:
    """Return default model config dict"""
    model_config = {
        "system_prompt": "Lang: English. Output Format: unformatted, outline. Task: Create detailed software documentation for publication using this entire code module Context:\n'{context}'\n",
        "instruction_prompt": "Analyze Context considering these objects:\n'{code_objects}'\n to comply with this instruction:\n'{query}'\n",
        "prompt_template": "SYSTEM: {system_prompt} USER: {instruction_prompt} ASSISTANT:",
        "inference_model": {
            "model_import_path": "ctransformers.AutoModelForCausalLM",
            "model_inference_function": "from_pretrained",
            "model_params": {
                "model_path": "jeffmeloy/WestLake-7B-v2.Q8_0.gguf",
                "model_type": "mistral",
                "local_files_only": False,
                ## MODEL CONFIGURATION PARAMETERS (params set for model with this HW: GPU: 4090-24GB VRAM, CPU: 5950x-64GB RAM)
                # avx2 and gpu_layers are not compatible
                # "lib": "avx2",
                "threads": 16,
                "batch_size": 512,
                "context_length": 40000,
                "max_new_tokens": 20000,
                "gpu_layers": 100,
                "reset": True,
            },
        },
    }
    return model_config


def get_start_dir(start_dir: str = "") -> str:
    """Returns the appropriate start directory."""
    if start_dir and not Path(start_dir).is_dir():
        logging.info(f"Setting Start Dir : {start_dir}")
        start_dir = os.getcwd()
    else:
        start_dir = os.path.abspath(start_dir)
    return start_dir


def get_output_dir(output_dir: str = "") -> str:
    """Returns the appropriate output directory."""
    output_dir = os.path.abspath(output_dir or OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Using output directory: {output_dir}")
    return output_dir


def get_questions(questions_pathname: str) -> List[Dict]:
    """Get questions from file or default"""
    try:  # get questions from provided or default configuration file
        if not questions_pathname:
            questions_pathname = os.path.join(os.getcwd(), QUESTIONS_FILE)
        with open(questions_pathname, "r") as f:
            questions = json.load(f)
        logging.info(f"Using questions from file: {questions_pathname}")
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        logging.info(
            f"Questions file not valid: {questions_pathname} Using default questions"
        )
        questions = get_default_questions()
    return questions


def instantiate_model(model_config: Dict) -> object:
    """Imports and instantiates a model based on the provided configuration."""
    try:
        module_name, class_name = model_config["model_import_path"].rsplit(".", 1)
        ModelClass = getattr(importlib.import_module(module_name), class_name)
        model_params = model_config["model_params"]
        inference_function_name = model_config["model_inference_function"]
        if inference_function_name != "":
            inference_function = getattr(ModelClass, inference_function_name)
            llm = inference_function(model_params.pop("model_path"), **model_params)
        else:
            llm = ModelClass(model_params.pop("model_path"), **model_params)
        return llm
    except (ImportError, AttributeError, Exception) as e:
        logging.info(f"Failed to instantiate the model. Error: {e}")
        return None


def get_model(model_config_pathname: str) -> object:
    """Returns an instantiated model and prompt template based on the model configuration."""
    try:
        if not model_config_pathname:
            model_config_pathname = os.path.join(os.getcwd(), MODEL_CONFIG_FILE)
        with open(model_config_pathname, "r") as config_file:
            model_config = yaml.safe_load(config_file)
        logging.info(f"Using model config from file: {model_config_pathname}")
    except (FileNotFoundError, yaml.YAMLError):
        logging.info(
            f"Model config file not valid: {model_config_pathname} Using default model config"
        )
        model_config = get_default_model_config()
    model_config["model"] = instantiate_model(model_config["inference_model"])

    return model_config


def write_questions_file(output_dir: str = "") -> None:
    """Writes the default questions to a file in JSON format."""
    questions = get_default_questions()
    output_dir = output_dir if output_dir and Path(output_dir).is_dir() else os.getcwd()
    with open(os.path.join(output_dir, QUESTIONS_FILE), "w") as file:
        json.dump(questions, file, indent=4)


def write_model_config_file(output_dir: str = "") -> None:
    """Writes the default model config to a file in YAML format."""
    model_config = get_default_model_config()
    output_dir = output_dir if output_dir and Path(output_dir).is_dir() else os.getcwd()
    with open(os.path.join(output_dir, MODEL_CONFIG_FILE), "w") as file:
        yaml.dump(model_config, file)
