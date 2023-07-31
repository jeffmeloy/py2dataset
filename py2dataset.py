"""
For each Python files within given directory, generate qa and instruct json
files that address the questions in the questions.json file. Combine these to 
create a composite qa.json and instruct.json file that includes all of the 
data filues stored in the output_dir (./datasets by default)
Requirements:
[req01] The read_file function shall accept a file path as an argument and
        return its contents as a dictionary.
[req02] The write_file function shall accept a dictionary and a file path as
        arguments, and write the dictionary to the file in JSON or YAML format.
[req03] The combine_files function shall accept a directory path as an 
        argument, merge all JSON files in the directory into 'qa.json' and 
        'instruct.json', remove duplicates, and replace duplicate inputs with
        an empty string.
[req04] The process_python_directories function shall accept a directory path,
        a dictionary of questions, a boolean flag indicating whether to use a
        large language model (LLM), and an output directory path as arguments.
[req05] The process_python_directories function shall analyze all Python files
        in the given directory and its subdirectories, generate a summary of
        each Python file's contents, generate question-answer pairs and 
        instructions for each Python file, and write the summaries, 
        question-answer pairs, and instructions to JSON and YAML files in the
        specified output directory.
[req06] The process_python_directories function shall call the combine_files
        function to merge all JSON files in the output directory after 
        processing all Python files in the directory.
[req07] The python_code_to_dataset function shall accept a directory path, a
        boolean flag indicating whether to use a large language model (LLM),
        and an output directory path as arguments.
[req08] The python_code_to_dataset function shall read questions from a JSON
        file named 'questions.json', call the process_python_directories
        function to analyze all Python files in the given directory and its
        subdirectories, and increase the Python recursion limit to handle 
        large files.
[req09] The command line interface of the script shall accept five arguments:
        the directory of Python files to analyze, a flag to indicate the use 
        of the large language model (LLM), a flag to suppress info logging
        messages, the output directory for generated files, and a flag to 
        clean input data.
[req10] The command line interface shall prompt the user for a directory if
        no directory argument is provided.
[req11] The command line interface shall raise a ValueError if the provided
        directory argument does not exist.
[req12] The logging level shall be set to WARNING if the '--quiet' flag is set,
        and to INFO otherwise by the command line interface.
[req13] The python_code_to_dataset function shall be called with the directory,
        'use_llm' flag, and output directory provided in the command line
        arguments.
[req14] The create_code_graph function shall accept file details, a base name,
        an output subdirectory, and a graph type as arguments. It shall 
        generate a graph from the file details and save
"""
import argparse
import sys
import os
import re
import json
import logging
import yaml
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import Dict, List, Union
from get_python_file_details import get_python_file_details
from get_python_json import get_python_json

def read_file(file_path: Path) -> Dict:
    """
    Reads a JSON or YAML file and returns its contents as a dictionary.
    Args:
        file_path (Path): The path to the file.
    Returns:
        The contents of the file as a dictionary.
    """
    file_type = file_path.suffix[1:]
    with file_path.open() as f:
        if file_type == 'json':
            return json.load(f)
        elif file_type == 'yaml':
            return yaml.load(f)


def write_file(data: Dict, file_path: Path) -> None:
    """
    Writes a dictionary to a JSON or YAML file. 
    Args:
        data (Dict): The data to write to the file.
        file_path (Path): The path to the file.
    """
    file_type = file_path.suffix[1:]
    with file_path.open('w') as f:
        if file_type == 'json':
            json.dump(data, f, indent=4)
        elif file_type == 'yaml':
            yaml.SafeDumper.ignore_aliases = lambda *args: True
            yaml.dump(data, f, Dumper=yaml.SafeDumper, sort_keys=False)


def combine_json_files(directory) -> None:
    """
    Combine all JSON files in the output directory into 'qa.json' and 
    'instruct.json', and then remove duplicates.
    Args:
        directory (str): The directory where the output files are located.
    """
    file_names = ['qa.json', 'instruct.json']
    keys = ['question', 'instruction']
    combined_data = []
    for file in file_names:
        file_path = Path(directory) / file
        if file_path.exists():
            combined_data = read_file(file_path)
        for json_file in Path(directory).rglob(f'*.{file}'):
            combined_data.extend(read_file(json_file))
        combined_data = list({i[keys[file_names.index(file)]]: i for i in combined_data}.values())
        write_file(combined_data, file_path)
        if file == 'instruct.json':
            instruct_combined_data = combined_data.copy()
        combined_data = []  

    # remove duplicate inputs from instruct.json to make a cleaned_instruct.json
    seen_inputs = set()
    for item in instruct_combined_data:
        if item['input'] in seen_inputs:
            item['input'] = ''
        else:
            seen_inputs.add(item['input'])
    cleaned_instruct_file_path = Path(directory) / 'cleaned_instruct.json'
    write_file(instruct_combined_data, cleaned_instruct_file_path)


def create_code_graph(file_details: Dict, base_name: str, output_subdir: Path) -> None:
    """
    Generate graphs from the file_details and save them as PNG images.
    Args:
        file_details (dict): The details extracted from the Python file.
        base_name (str): The base name of the output files.
        output_subdir (Path): The subdirectory where the output files will be
            saved.
    """
    for graph_type in ['internal_code_graph', 'entire_code_graph']:
        # Create graphs
        output_file = output_subdir / f'{base_name}.{graph_type}.png'  
        G = nx.DiGraph()
        for node_name in file_details['file_info'][graph_type]['nodes']:
            G.add_node(node_name)
        # Add edges
        for edge in file_details['file_info'][graph_type]['edges']:
            source = edge['source']
            target = edge['target']
            if source in G.nodes and target in G.nodes:
                edge_data = {}
                if 'target_input' in edge:
                    edge_data['target_input'] = edge['target_input']
                if 'target_returns' in edge:
                    edge_data['target_returns'] = edge['target_returns']
                G.add_edge(source, target, **edge_data)
            
        # Save code graph as png
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, font_weight='bold', font_size = 8, node_shape='s', node_size=500, width=1, arrowsize=12)
        edge_labels = {}
        for edge in G.edges(data=True):
            label = []
            if 'target_input' in edge[2] and edge[2]['target_input']:
                label.append(f"Inputs: {', '.join(edge[2]['target_input'])}")
            if 'target_returns' in edge[2] and edge[2]['target_returns']:
                label.append(f"\nReturns: {', '.join(edge[2]['target_returns'])}")
            edge_labels[(edge[0], edge[1])] = '\n'.join(label)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        plt.savefig(output_file)
        plt.close()  # Close the figure


def process_python_directories(start_path: str, questions: Dict[str, Union[str, Dict]], use_llm: bool=False, graph: bool=False, output_dir: str=None, model_config: dict = None) -> None:
    """
    Processes all Python files in a given directory and its subdirectories.
    Args:
        start_path (str): The directory to start the search for Python files.
        questions (Dict): The set of questions to answer about each Python 
            file.
        use_llm (bool): Whether to use the LLM model to generate answers for
            json.
        output_dir (str): The directory where the output files should be
            written. If not provided, the function writes the files to the
            'python_json_and_yaml' directory in the current working directory.
    """
    python_files = list(Path(start_path).rglob('[!_]*.py'))
    for file_path in python_files:
        logging.info(f'Processing: {file_path}')
        relative_path = Path(file_path).relative_to(start_path)
        base_name = '.'.join(part for part in relative_path.parts)

        # use AST to get python file details
        file_details = get_python_file_details(file_path)
        if file_details is None or isinstance(file_details, tuple):
            continue

        # get lists for qa.json and intruct.json for python file
        
        qa_list, instruct_list = get_python_json(file_path, file_details, base_name, questions, use_llm, model_config)
        if not qa_list:
            continue

        # create output directory if needed
        if output_dir is None:
            output_dir = './datasets/'
        output_subdir = Path(output_dir) / relative_path.parts[0]
        output_subdir.mkdir(parents=True, exist_ok=True)

        # write qa.json and instrunct.json files
        file_names = [f'{base_name}.qa.json', f'{base_name}.instruct.json', f'{base_name}.details.yaml']
        contents = [qa_list, instruct_list, file_details]
        for file_name, content in zip(file_names, contents):
            write_file(content, output_subdir / file_name)

        # Create code graph images
        if graph:
            create_code_graph(file_details, base_name, output_subdir)

    # combine all of the qa.json and instruct.json files together
    combine_json_files(output_dir)


def py2dataset(start_path: str, use_llm: bool=False, graph: bool=False, output_dir: str=None, model_config_path: str=None) -> None:
    sys.setrecursionlimit(3000)  # Increase the recursion limit for AST
    questions = read_file(Path('questions.json'))
    model_config = None
    if model_config_path:
        model_config = read_file(Path(model_config_path))
    process_python_directories(start_path, questions, use_llm, graph, output_dir, model_config)

if __name__ == "__main__":
    arg_string = ' '.join(sys.argv[1:])
    use_llm = False
    quiet = False
    graph = False
    output_dir = None
    model_config_path = None
    if '--use_llm' in arg_string:
        use_llm = True
        arg_string = arg_string.replace('--use_llm', '')
    if '--quiet' in arg_string:
        quiet = True
        arg_string = arg_string.replace('--quiet', '')
    if '--graph' in arg_string:
        graph = True
        arg_string = arg_string.replace('--graph', '')
    if '--output_dir' in arg_string:
        output_dir = arg_string.split('--output_dir ')[1].split(' ')[0]
        arg_string = arg_string.replace(f'--output_dir {output_dir}', '')
    if '--model_config' in arg_string:
        model_config_path = arg_string.split('--model_config ')[1].split(' ')[0]
        arg_string = arg_string.replace(f'--model_config {model_config_path}', '')

    directory = arg_string.strip()
    if directory.endswith('"'):
        directory = directory[:-1]
    if not os.path.isdir(directory):
        raise ValueError(f"The directory {directory} does not exist.")

    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)

    py2dataset(directory, use_llm, graph, output_dir, model_config_path)