"""
For each Python file within given directory, generate, save, and return datasets
that include responses to questions about the code.
Requirements:
[req01] The `process_python_directories` function shall:
        a. Accept parameters for the starting directory, output directory, questions dictionary, LLM, prompt, use of LLM, use of summary, graph generation, and HTML generation.
        b. Search for all Python files within the given directory and its subdirectories.
        c. Process each Python file using the 'get_python_file_details' and 'get_python_datasets' functions.
        d. Write the resulting datasets to the specified output directory.
        e. Return the datasets.
[req02] The `py2dataset` function shall:
        a. Accept parameters for the starting directory, output directory, questions pathname, model configuration pathname, use of LLM, use of summary, graph generation, quiet mode, and HTML generation.
        b. Determine the appropriate parameters for the `process_python_directories` function based on provided or default values.
        c. Call the `process_python_directories` function.
        d. Adjust the logging level based on the quiet flag.
        e. Return the datasets.
[req03] The `main` function shall:
        a. Accept command-line arguments.
        b. Process the command-line arguments to determine the parameters for the `py2dataset` function.
        c. Call the `py2dataset` function with the derived parameters.
"""
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Union
from get_python_file_details import get_python_file_details
from get_python_datasets import get_python_datasets
from get_py2dataset_params import get_questions, get_model, get_output_dir
from save_py2dataset_output import read_file, write_file, combine_json_files, create_code_graph

def process_python_directories(start_dir: str, output_dir: str, questions: Dict, llm, prompt, use_llm: bool, use_summary: bool, graph: bool, html: bool) -> Dict[str, List[Dict]]:
    """
    Processes all Python files in the provided directory and subdirectories.
    Args:
        start_dir (str): Starting directory to search for Python files.
        output_dir (str): Directory to write the output files.
        questions (Dict): Questions dictionary to answer about each Python file.
        llm: Large Language Model to use for generating answers.
        prompt (str): Prompt to provide to the language model.
        use_llm (bool): If True, use the LLM model to generate answers for JSON.
        use_summary (bool): Use the code summary to reduce dataset context length.
        graph (bool): Generate graphs for the code.
        html (bool): Generate HTML files from the JSON files.
    Returns:
        Dict[str, List[Dict]]: Datasets dictionary.
    """
    python_files = [p for p in Path(start_dir).rglob('[!_]*.py') if p.is_file()]

    for file_path in python_files:
        logging.info(f'Processing: {file_path}')
        relative_path = Path(file_path).relative_to(start_dir)
        base_name = '.'.join(part for part in relative_path.parts)

        # use AST to get python file details
        file_details = get_python_file_details(file_path)
        if file_details is None or isinstance(file_details, tuple):
            continue

        # get lists for qa.json and intruct.json for python file
        qa_list, instruct_list = get_python_datasets(file_path, file_details, base_name, questions, llm, prompt, use_llm, use_summary)
        if not qa_list:
            continue

        output_subdir = Path(output_dir) / relative_path.parts[0]
        output_subdir.mkdir(parents=True, exist_ok=True)

        # write qa.json and instrunct.json files
        file_names = [f'{base_name}.qa.json', f'{base_name}.instruct.json', f'{base_name}.details.yaml']
        contents = [qa_list, instruct_list, file_details]
        for file_name, content in zip(file_names, contents):
            write_file(content, output_subdir / file_name)

        # Create code graph images
        if graph:
            # add error handling if anything goes wrong with creating or saving the graph
            try:
                create_code_graph(file_details, base_name, output_subdir)
            except:
                logging.info(f'Error creating graph for {file_path}')
                continue

    # combine all of the qa.json and instruct.json files together
    datasets = combine_json_files(output_dir, html)   
    return datasets


def py2dataset(start_dir: str='', output_dir: str='', questions_pathname: str='', model_config_pathname: str='', use_llm: bool=False, use_summary: bool=False, graph: bool=False, quiet: bool=False, html: bool=False) -> Dict[str, List[Dict]]:
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
    
    llm = None
    model_config = None
    prompt = ''
    if use_llm:
        llm, prompt = get_model(model_config_pathname)

    datasets = process_python_directories(start_dir, output_dir, questions, llm, prompt, use_llm, use_summary, graph, html)
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