"""
For each Python file within given directory, generate, save, and return datasets that include responses to questions about the code.
Requirements:
[req00] The extract_python_data function shall:
    a. Accept parameters for the Python file path, base name, model configuration pathname, questions dictionary, use of LLM, and use of summary.
    b. Use the 'get_python_file_details' function to get the Python file details.
    c. If the use_llm parameter is True, instantiate the LLM using the 'get_model' function.
    d. Use the 'get_python_datasets' function to get the qa.json and instruct.json datasets.
    e. Return the file details, qa.json dataset, and instruct.json dataset.
[req01] The process_python_directories function shall:
    a. Accept parameters for the starting directory, output directory, model configuration pathname, questions dictionary, use of LLM, use of summary, graph generation, and HTML generation.
    b. Search for all Python files within the given directory and its subdirectories using a glob pattern.
    c. For each Python file, call the extract_python_data function to get the file details, qa.json dataset, and instruct.json dataset.
    d. For valid Python file datasets, call the save_python_data function to save the file details, qa.json dataset, and instruct.json dataset.
    e. Combine all of the qa.json and instruct.json files together using the 'combine_json_files' function.
    f. Return the combined datasets.
[req02] The py2dataset function shall:
    a. Accept parameters for the starting directory, output directory, questions pathname, model configuration pathname, use of LLM, use of summary, graph generation, quiet mode, and HTML generation.
    b. Determine the starting directory based on provided or default values.
    c. Adjust the logging level based on the quiet flag.
    d. Call the process_python_directories function to process the Python files and generate datasets.
    e. Return the datasets.
[req03] The main function shall:
    a. Accept and process command-line arguments.
    b. Determine the parameters for the py2dataset function based on the processed command-line arguments.
    c. Call the py2dataset function with the derived parameters.
"""
import os
import sys
import gc
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

from get_python_file_details import get_python_file_details
from get_python_datasets import get_python_datasets
from get_py2dataset_params import get_questions, get_model, get_output_dir
from save_py2dataset_output import combine_json_files, save_python_data

def extract_python_data(file_path: str, base_name: str, model_config_pathname: str, questions: Dict, 
                        use_llm: bool, use_summary: bool) -> Tuple[Union[Dict, Tuple], List[Dict], List[Dict]]:
    """
    Extracts data from a Python file.
    Args:
        file_path (str): Path to the Python file.
        base_name (str): Base name of the Python file.
        model_config_pathname (str): Path to the model configuration file.
        questions (Dict): Questions dictionary to answer about the Python file.
        use_llm (bool): If True, use the LLM model to generate answers for JSON.
        use_summary (bool): Use the code summary to reduce dataset context length.
    Returns:
        Tuple[Union[Dict, Tuple], List[Dict], List[Dict]]: File details dictionary or tuple of file details and None, qa.json dataset, instruct.json dataset.
    """
    file_details, qa_list, instruct_list = None, None, None    
    # use AST to get python file details
    file_details = get_python_file_details(file_path)
    if file_details is None or isinstance(file_details, tuple):
        return file_details, qa_list, instruct_list
    
    #instantiate llm and prompt if use_llm is True
    llm, prompt = get_model(model_config_pathname) if use_llm else (None, '')
    
    # get lists for qa.json and instruct.json for python file
    qa_list, instruct_list = get_python_datasets(file_path, file_details, base_name, questions, use_summary, use_llm, llm, prompt)
    
    del llm
    gc.collect()
    
    return file_details, qa_list, instruct_list


def process_python_directories(start_dir: str, output_dir: str, model_config_pathname: str, questions: Dict, 
                               use_llm: bool, use_summary: bool, graph: bool, html: bool) -> Dict[str, List[Dict]]:
    """
    Processes all Python files in the provided directory and subdirectories.
    Args:
        start_dir (str): Starting directory to search for Python files.
        output_dir (str): Directory to write the output files.
        model_config_pathname (str): Path to the model configuration file.
        questions (Dict): Questions dictionary to answer about each Python file.
        use_llm (bool): If True, use the LLM model to generate answers for JSON.
        use_summary (bool): Use the code summary to reduce dataset context length.
        graph (bool): Generate graphs for the code.
        html (bool): Generate HTML files from the JSON files.
    Returns:
        Dict[str, List[Dict]]: Datasets dictionary.
    """
    datasets = {}
    for pythonfile_path in Path(start_dir).rglob('[!_]*.py'):
        logging.info(f'Processing: {pythonfile_path}')
        relative_path = pythonfile_path.relative_to(start_dir)
        base_name = '.'.join(part for part in relative_path.parts)

        # get and save file_details, qa_list, and instruct_list
        file_details, qa_list, instruct_list = extract_python_data(pythonfile_path, base_name, model_config_pathname, questions, use_llm, use_summary)
        if file_details is None or isinstance(file_details, tuple):
            continue
        save_python_data(file_details, qa_list, instruct_list, relative_path, output_dir, graph, html)

    # combine all of the qa.json and instruct.json files together
    datasets = combine_json_files(output_dir, html)   
    return datasets


def py2dataset(start_dir: str = '', output_dir: str = '', questions_pathname: str = '', model_config_pathname: str = '', 
               use_llm: bool = False, use_summary: bool = False, graph: bool = False, 
               quiet: bool = False, html: bool = False) -> Dict[str, List[Dict]]:
    """
    Process Python files to generate question-answer pairs and instructions.
    Args:
        start_dir (str, optional): Starting directory to search for Python files. Defaults to current working directory.
        output_dir (str, optional): Directory to write the output files.
        questions_pathname (str, optional): Path to the questions file.
        model_config_pathname (str, optional): Path to the model configuration file.
        use_llm (bool, optional): If True, use a Large Language Model for generating JSON answers. Defaults to False.
        use_summary (bool, optional): Use code summary to reduce dataset context length. Defaults to False.
        graph (bool, optional): Generate graphs for the code. Defaults to False.
        quiet (bool, optional): Limit logging output. Defaults to False.
        html (bool, optional): Generate HTML files from the JSON files. Defaults to False.
    Returns:
        Dict[str, List[Dict]]: Datasets dictionary.
    """
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)
    sys.setrecursionlimit(3000)  # Increase the recursion limit for AST
    
    # if start dir is empty or not a valid directory, use current working directory
    if start_dir == '' :
        logging.info('No valid start path provided. Using current working directory.')
        start_dir = os.getcwd()    
    start_dir = os.path.abspath(start_dir)
    
    output_dir = get_output_dir(output_dir)
    questions = get_questions(questions_pathname)

    datasets = process_python_directories(start_dir, output_dir, model_config_pathname, questions, use_llm, use_summary, graph, html)
    return datasets


def main():
    """
    Command-line entry point for processing Python files and generating datasets.
    Args:
        --start_dir (str, optional): Starting directory to search for Python files. Defaults to the current working directory.
        --output_dir (str, optional): Directory to write the output files. Defaults to the 'datasets' directory in the current working directory.
        --questions_pathname (str, optional): Path to the questions file. If not provided, defaults defined in 'get_py2dataset_params.py' will be used.
        --model_config_pathname (str, optional): Path to the model configuration file. If not provided, defaults defined in 'get_py2dataset_params.py' will be used.
        --use_llm (bool, optional): Use a Large Language Model for generating JSON answers. Defaults to False.
        --use_summary (bool, optional): Use code summary to reduce dataset context length. Defaults to False.
        --graph (bool, optional): Generate graphs for the code. Defaults to False.
        --html (bool, optional): Generate HTML files from the JSON files. Defaults to False.
        --quiet (bool, optional): Limit logging output. If provided, only warnings and errors will be logged. Defaults to False.
    """
    arg_string = ' '.join(sys.argv[1:])
    start_dir = ''
    output_dir = ''
    questions_pathname = ''
    model_config_pathname = ''
    use_llm = False
    use_summary = False
    quiet = False
    graph = False
    html = False
    if '--start_dir' in arg_string:
        start_dir = arg_string.split('--start_dir ')[1].split(' ')[0]
        arg_string = arg_string.replace(f'--start_dir {start_dir}', '')
    if '--output_dir' in arg_string:
        output_dir = arg_string.split('--output_dir ')[1].split(' ')[0]
        arg_string = arg_string.replace(f'--output_dir {output_dir}', '')
    if '--model_config_pathname' in arg_string:
        model_config_pathname = arg_string.split('--model_config_pathname ')[1].split(' ')[0]
        arg_string = arg_string.replace(f'--model_config_pathname {model_config_pathname}', '')
    if '--questions_pathname' in arg_string:
        questions_pathname = arg_string.split('--questions_pathname ')[1].split(' ')[0]
        arg_string = arg_string.replace(f'--questions_pathname {questions_pathname}', '') 
    if '--use_llm' in arg_string:
        use_llm = True
        arg_string = arg_string.replace('--use_llm', '')
    if '--use_summary' in arg_string:
        use_summary = True
        arg_string = arg_string.replace('--use_summary', '')
    if '--quiet' in arg_string:
        quiet = True
        arg_string = arg_string.replace('--quiet', '')
    if '--graph' in arg_string:
        graph = True
        arg_string = arg_string.replace('--graph', '')
    if '--html' in arg_string:
        html = True
        arg_string = arg_string.replace('--html', '')

    py2dataset(start_dir, output_dir, questions_pathname, model_config_pathname, use_llm, use_summary, graph, quiet, html)

if __name__ == "__main__":
    main()