"""
For each Python file within given directory, generate, save, and return datasets
that include responses to questions about the code.
Requirements:
[req00] The process_single_file function shall:
    a. Accept parameters for Python file path, start directory, model config 
    pathname, questions dict, use of LLM, and output dir.
    b. If 'use_llm' is True, use 'get_model' to instantiate LLM model config.
    c. Use 'get_python_file_details' to retrieve Python file info.
    d. Use 'get_python_datasets' to acquire instruct.json datasets.
    e. Use 'save_python_data' to store file details and instruct.json data.
[req01] The py2dataset function shall:
    a. Accept parameters for start directory, output dir, questions path, model
    config path, use of LLM, and quiet mode.
    b. Adjust logging level based on 'quiet'.
    c. Use current working dir if no valid start dir is provided.
    d. Get output dir with 'get_output_dir'.
    e. Retrieve questions dict with 'get_questions'.
    f. Search for Python files using 'rglob', excluding those starting with "_".
    g. For each Python file, spawn a child process with 'process_single_file'
    to get file details and instruct.json data, if single_process is False.
    h. Combine instruct.json files with 'combine_json_files'.
    i. Return datasets.
[req02] The main function shall:
    a. Accept and process command-line args.
    b. Determine py2dataset parameters based on processed arguments.
    c. Call py2dataset with derived parameters.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List
from multiprocessing import Process

from get_python_file_details import get_python_file_details
from get_python_datasets import get_python_datasets
from get_py2dataset_params import get_questions, get_model, get_output_dir
from save_py2dataset_output import combine_json_files, save_python_data

def process_single_file(pythonfile_path: str, start_dir: str, model_config_pathname: str,
                        questions: Dict, use_llm: bool, output_dir: str,
                        model_config: Dict = None, single_process: bool = False) -> None:
    """
    Process a single Python file to generate question-answer pairs and instructions.
    Args:
        pythonfile_path (str): Path to the Python file.
        start_dir (str): Starting directory to search for Python files.
        model_config_pathname (str): Path to the model configuration file.
        questions (Dict): Questions dictionary to answer about the Python file.
        use_llm (bool): If True, use a Large Language Model for generating JSON answers.
        output_dir (str): Directory to write the output files.
        model_config (Dict): Model configuration dictionary for the LLM.
        single_process (bool, optional): Set True to use single process. Defaults to False.
    Returns:
        none
    """
    logging.info(f'Processing: {pythonfile_path}')
    relative_path = pythonfile_path.relative_to(start_dir)
    base_name = '.'.join(part for part in relative_path.parts)

    if not single_process:
        # Instantiate llm and prompt if use_llm is True for each file to avoid
        # multiprocessing pickling problem
        model_config = get_model(model_config_pathname) if use_llm else (None, '', 0)

    # Use AST to get python file details
    file_details = get_python_file_details(pythonfile_path)
    if file_details is None or isinstance(file_details, tuple):
        return

    # Get lists for instruct.json for python file
    instruct_list = get_python_datasets(
        pythonfile_path,
        file_details,
        base_name,
        questions,
        model_config
        )
    if instruct_list is None:
        return

    save_python_data(file_details, instruct_list, relative_path, output_dir)

def py2dataset(start_dir: str = '', output_dir: str = '', questions_pathname: str = '',
               model_config_pathname: str = '', use_llm: bool = False, quiet: bool = False,
               single_process: bool = False) -> Dict[str, List[Dict]]:
    """
    Process Python files to generate question-answer pairs and instructions.
    Args:
        start_dir (str, optional): Starting directory for Python files.
        Defaults to current working directory.
        output_dir (str, optional): Directory to write the output files.
        questions_pathname (str, optional): Path to the questions file.
        model_config_pathname (str, optional): Path to the model
        configuration file.
        use_llm (bool, optional): If True, use a Large Language Model
        for generating JSON answers. Defaults to False.
        quiet (bool, optional): Limit logging output. Defaults to False.
        single_process(bool, optional): If True, only a single process 
        will be used to process Python files. Defaults to False. Set to True to
        instantiate LLM once before processing all files.
    Returns:
        Dict[str, List[Dict]]: Datasets dictionary.
    """
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)
    sys.setrecursionlimit(3000)  # Increase the recursion limit for AST

    # If start dir is empty or not a valid directory, use current working directory
    if not start_dir:
        logging.info('No valid start path provided. Using current working directory.')
        start_dir = os.getcwd()
    start_dir = os.path.abspath(start_dir)
    output_dir = get_output_dir(output_dir)
    questions = get_questions(questions_pathname)

    # If single_process is True, load model config here
    if single_process:
        model_config = get_model(model_config_pathname) if use_llm else (None, '', 0)

    for pythonfile_path in Path(start_dir).rglob('[!_]*.py'):
        if pythonfile_path.is_dir():
            continue
        if single_process:
            process_single_file(
                pythonfile_path,
                start_dir,
                model_config_pathname,
                questions,
                use_llm,
                output_dir,
                model_config,
                single_process)
            continue
        # Spawn a new child process to manage python memory leaks
        proc = Process(
            target=process_single_file,
            args=(
                pythonfile_path,
                start_dir,
                model_config_pathname,
                questions,
                use_llm,
                output_dir)
            )
        proc.start()
        proc.join()

    # Combine all of the instruct.json files together
    datasets = combine_json_files(output_dir)
    return datasets

def main():
    """
    Command-line entry point for processing Python files and generating datasets.
    """
    arg_string = ' '.join(sys.argv[1:])
    start_dir = ''
    output_dir = ''
    questions_pathname = ''
    model_config_pathname = ''
    use_llm = False
    quiet = False
    single_process = False

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
    if '--quiet' in arg_string:
        quiet = True
        arg_string = arg_string.replace('--quiet', '')
    if '--single_process' in arg_string:
        single_process = True
        arg_string = arg_string.replace('--single_process', '')

    py2dataset(
        start_dir,
        output_dir,
        questions_pathname,
        model_config_pathname,
        use_llm,
        quiet,
        single_process)

if __name__ == "__main__":
    main()
