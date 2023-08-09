"""
For each Python files within given directory, generate qa and instruct json
files that address the questions in the questions.json file. Combine these to 
create a composite qa.json and instruct.json file that includes all of the 
data filues stored in the output_dir (./datasets by default)
Requirements:
[req01] The read_file function shall accept a file path as an argument and
        return its contents as a dictionary. This function requires the 'json'
        and 'yaml' libraries for reading JSON and YAML files respectively.
[req02] The write_file function shall accept a dictionary and a file path as
        arguments, and write the dictionary to the file in JSON or YAML format.
        This function requires the 'json' and 'yaml' libraries for writing JSON
        and YAML files respectively.
[req03] The combine_json_files function shall accept a directory path as an 
        argument, merge all JSON files in the directory into 'qa.json' and 
        'instruct.json', remove duplicates, and replace duplicate inputs with
        an empty string. This function requires the 'json' and 'os' libraries.
[req04] The create_code_graph function shall accept a directory path, a
        dictionary of questions, a boolean flag indicating whether to use a
        large language model (LLM), and an output directory path as arguments.
        This function requires the 'matplotlib.pyplot' and 'networkx' libraries
        to create and display graphs.
[req05] The process_python_directories function shall accept a directory path,
        a dictionary of questions, a boolean flag indicating whether to use a
        large language model (LLM), and an output directory path as arguments.
        This function requires the 'os', 'json', 'yaml', 
        'get_python_file_details', and 'get_python_datasets' libraries.
[req06] The main function shall call the py2dataset function with appropriate
        arguments. This function requires the 'sys' library to access command
        line arguments.
"""
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
from get_python_datasets import get_python_datasets
from get_py2dataset_params import get_questions, get_model, get_output_dir

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


def combine_json_files(directory) -> Dict[str, List[Dict]]:
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

    # create a qa_purpose.json, qa_instruct.jaon, and qa_cleaned_instruct.json
    file_names = ['qa.json', 'instruct.json', 'cleaned_instruct.json']
    keys = ['question', 'instruction', 'instruction']
    for file in file_names:
        purpose_data = []
        nquestion = 0
        dataset = read_file(Path(directory) / file)
        for item in dataset:
            if item[keys[file_names.index(file)]].startswith('Purpose of'):
                purpose_data.append(item)
                nquestion += 1
        if nquestion > 0:
            purpose_filepath = Path(directory) / f'{file.split(".")[0]}_purpose.json'
            write_file(purpose_data, purpose_filepath)
        if file == 'qa.json':
            qa_list = dataset.copy()
        if file == 'instruct.json':
            instruct_list = dataset.copy()
    return {'qa_list': qa_list, 'instruct_list': instruct_list}
       

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
                if 'target_inputs' in edge:
                    edge_data['target_inputs'] = edge['target_inputs']
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
            if 'target_inputs' in edge[2] and edge[2]['target_inputs']:
                label.append(f"Inputs: {', '.join(edge[2]['target_inputs'])}")
            if 'target_returns' in edge[2] and edge[2]['target_returns']:
                label.append(f"\nReturns: {', '.join(edge[2]['target_returns'])}")
            edge_labels[(edge[0], edge[1])] = '\n'.join(label)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        plt.savefig(output_file)
        plt.close()  # Close the figure


def process_python_directories(start_dir: str, output_dir: str, questions: Dict, llm, prompt, use_llm: bool, use_summary: bool, graph: bool) -> Dict[str, List[Dict]]:
    """
    Processes all Python files in a given directory and its subdirectories.
    Args:
        start_dir (str): The directory to start the search for Python files.
        output_dir (str): The directory where the output files should be
            written.
        questions (Dict): The set of questions to answer about each Python 
            file.
        model_config (Dict): The configuration for the model.
        use_llm (bool): Whether to use the LLM model to generate answers for
            json.
        use_summary (bool): Whether to use the summary of the code to reduce 
            dataset context length
        graph (bool): Whether to generate graphs for the code.
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
    datasets = combine_json_files(output_dir)
    return datasets


def py2dataset(start_dir: str='', output_dir: str='', questions_pathname: str='', model_config_pathname: str='', use_llm: bool=False, use_summary: bool=False, graph: bool=False, quiet: bool=False) -> Dict[str, List[Dict]]:
    """
    Process Python files within the specified directory and its 
    subdirectories, to generating question-answer pairs and instructions for
    each file. The results are written to JSON and YAML files in the specified
    output directory.
    Args:
        start_dir (str, optional): directory to start the search for Python
            files.
        use_llm (bool, optional): If True, use a large language model to
            generate answers for JSON. Defaults to False.
        graph (bool, optional): If True, generate graphs from the file details. 
              Defaults to False.
        output_dir (str, optional): Path to the directory where the output
            files should be written. 
        model_config_pathname (str, optional): Path to the model configuration
            file. 
    Raises:
        ValueError: If the provided directory does not exist.
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

    datasets = process_python_directories(start_dir, output_dir, questions, llm, prompt, use_llm, use_summary, graph)
    return datasets


def main():
    """
    Command line function called function to process Python files within the 
    specified directory and its subdirectories, to generating question-answer
    pairs and instructions for each file. The results are written to JSON and
    YAML files in the specified output directory.
    Args:
        start_dir (str, optional): directory to start the search for Python
            files.
        use_llm (bool, optional): If True, use a large language model to
            generate answers for JSON. Defaults to False.
        graph (bool, optional): If True, generate graphs from the file details.
                Defaults to False.
        output_dir (str, optional): Path to the directory where the output
            files should be written. If not provided, writes the files to the
            'datasets' directory in the current working directory.
        model_config_pathname (str, optional): Path to the model configuration file.
            If not provided, defaults tO local 'py2dataset_model_config.yaml'
        questions_pathname (str, optional): Path to the questions file.
    Raises: ValueError: If the provided directory does not exist.
    """
    arg_string = ' '.join(sys.argv[1:])
    start_dir = ''
    use_llm = False
    use_summary = False
    quiet = False
    graph = False
    output_dir = ''
    questions_pathname = ''
    model_config_pathname = ''
    if '--start_dir' in arg_string:
        start_dir = arg_string.split('--start_dir ')[1].split(' ')[0]
        arg_string = arg_string.replace(f'--start_dir {start_dir}', '')
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
    if '--output_dir' in arg_string:
        output_dir = arg_string.split('--output_dir ')[1].split(' ')[0]
        arg_string = arg_string.replace(f'--output_dir {output_dir}', '')
    if '--model_config_pathname' in arg_string:
        model_config_pathname = arg_string.split('--model_config_pathname ')[1].split(' ')[0]
        arg_string = arg_string.replace(f'--model_config_pathname {model_config_pathname}', '')
    if '--questions_pathname' in arg_string:
        questions_pathname = arg_string.split('--questions_pathname ')[1].split(' ')[0]
        arg_string = arg_string.replace(f'--questions_pathname {questions_pathname}', '') 

    py2dataset(start_dir, output_dir, questions_pathname, model_config_pathname, use_llm, use_summary, graph, quiet)

if __name__ == "__main__":
    main()