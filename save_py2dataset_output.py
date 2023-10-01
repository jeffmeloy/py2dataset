"""
Utility functions for reading the input and saving the output of the py2dataset script.
Requirements:
[req01] The `read_file` function shall:
        a. Accept a file path as an argument.
        b. Read and return the contents of a JSON or YAML file as a dictionary.
[req02] The `write_file` function shall:
        a. Accept a dictionary and a file path as arguments.
        b. Write the dictionary to a file in either JSON or YAML format.
[req03] The `convert_json_to_html` function shall:
        a. Convert JSON files within a given directory to HTML format.
        b. Save each converted file with a .html extension.
        c. Preserve spacing and tabs for the 'input' field.
[req04] The `combine_json_files` function shall:
        a. Accept a directory path as an argument.
        b. Merge all JSON files in the directory.
        c. Remove duplicates from the combined JSON files.
        d. Write the combined data to 'instruct.json' files.
        e. Convert the merged JSON files to HTML format.
        f. Return the 'instruct_list' datasets.
[req05] The `create_code_graph` function shall:
        a. Accept details of a Python file, a base name, and an output directory as arguments.
        b. Generate code graphs based on the provided file details.
        c. Save the graphs as PNG images in the specified output directory.
[req06] The `save_python_data` function shall:
        a. Accept details of a Python file, a base name, and an output directory as arguments.
        b. Save the details of the Python file as a YAML file.
        c. Save the instruction data as JSON files.
        d. Generate and save code graphs.
"""
import sys
import os
import re
import json
import logging
import yaml
import matplotlib.pyplot as plt
import networkx as nx
from html import escape
from pathlib import Path
from typing import Dict, List, Union


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
    Returns:
        None
    """
    file_type = file_path.suffix[1:]
    with file_path.open('w') as f:
        if file_type == 'json':
            json.dump(data, f, indent=4)
        elif file_type == 'yaml':
            yaml.SafeDumper.ignore_aliases = lambda *args: True
            yaml.dump(data, f, Dumper=yaml.SafeDumper, sort_keys=False)


def convert_json_to_html(directory: str) -> None:
    """
    Convert JSON files within a given directory to HTML format.
    Args:
        directory (str): The directory where the JSON files are located.
    Returns:
        None    
    """
    def preserve_spacing(text: str, tab_width: int = 4) -> str:
        """Preserve spaces and tabs in the provided text."""
        return text.replace(" ", "&nbsp;").replace("\t", "&nbsp;" * tab_width)

    for json_file in Path(directory).rglob('*.json'):
        dataset = read_file(json_file)
        if not dataset:
            continue

        html_file = json_file.with_suffix('.html')
        html_content = """
        <html>
        <head>
            <style>
                table {border-collapse: collapse; width: 100%; table-layout: fixed;}
                th, td {
                    border: 1px solid black;
                    padding: 8px;
                    text-align: left;
                    white-space: pre-line;
                    vertical-align: top;
                    word-wrap: break-word;
                }
            </style>
        </head>
        <body>
            <table>
                <thead>
                    <tr>
        """
        column_count = len(dataset[0].keys())
        column_width = 100 / column_count  # Calculate the width for each column based on the number of columns
        for key in dataset[0].keys():
            html_content += f"<th style='width: {column_width}%;'>{key}</th>"
        html_content += """
                    </tr>
                </thead>
                <tbody>
        """
        for entry in dataset:
            html_content += "<tr>"
            for key in entry:
                # Convert \n to HTML line breaks
                value = escape(str(entry[key]))
                value = preserve_spacing(value)
                value = value.replace('\n', '<br/>')
                html_content += f"<td>{value}</td>"
            html_content += "</tr>"

        html_content += """
                </tbody>
            </table>
        </body>
        </html>
        """
        html_file_path = json_file.with_suffix('.html')
        try:   
            with open(html_file_path, 'w', encoding='utf-8') as file:
                file.write(html_content)
        except:
            logging.save(logging.info(f'Failed saving: {html_file_path}'))


def combine_json_files(directory: str) -> Dict[str, List[Dict]]:
    """
    Combine all JSON files in the output directory into 'instruct.json', and then remove duplicates.
    Args:
        directory (str): The directory where the output files are located.
    Returns:
        A dictionary containing the 'instruct_list' datasets.
    """
   
    def remove_duplicate_dataset_entries(dataset: List[Dict], key1: str, key2: str) -> List[Dict]:
        """
        Remove duplicate entries from the provided dataset based on the provided keys.
        Args:
            dataset (List[Dict]): The dataset to remove duplicates from.
            key1 (str): The first key to check for duplicates.
            key2 (str): The second key to check for duplicates.
        Returns:
            A dataset without duplicate entries.
        """
        seen = set()
        result = []
        for item in dataset:
            if (item[key1], item[key2]) not in seen:
                seen.add((item[key1], item[key2]))
                result.append(item)
        return result

    instruct_data = []
    for file_name in ['instruct.json']:
        file_path = Path(directory) / file_name
        combined_data = []
        for json_file in Path(directory).rglob(f'*.{file_name}'):
            json_file_data = read_file(json_file)
            combined_data.extend(json_file_data)
            combined_data = remove_duplicate_dataset_entries(combined_data, 'instruction', 'output')
            instruct_data = combined_data.copy()
            # gen training datasets that contains purpose and graph data formatted as follow for each item in the dataset:
            purpose_data = [item for item in combined_data if item['instruction'].startswith('1) Describe the purpose')]
            graph_data = [item for item in combined_data if item['instruction'].startswith('What is the call code graph')]
            code_output = []
            graph_output = []
            for item in purpose_data:
                code_output.append({'instruction': 'Define the Python code file that is described as follows:\n'+ item['output'], 'output': item['input']})
            for item in graph_data:
                graph_output.append({'instruction': 'Define the call code graph for Python file:\n' + item['input'], 'output': item['output']})
            code_graph_output = code_output + graph_output
            write_file(code_graph_output, Path(directory) / 'training.json')

        write_file(combined_data, file_path)

    # Save html file for each json file in the output directory
    convert_json_to_html(directory)
    return {'instruct_list': instruct_data}


def create_code_graph(file_details: Dict, base_name: str, output_subdir: Path) -> None:
    """
    Generate graphs from the file_details and save them as PNG images.
    Args:
        file_details (dict): The details extracted from the Python file.
        base_name (str): The base name of the output files.
        output_subdir (Path): The subdirectory where the output files will be saved.
    Returns:
        None
    """
    graph_type = 'entire_code_graph'
    output_file = output_subdir / f'{base_name}.{graph_type}.png'

    # Create graphs, add nodes, and add edges
    G = nx.DiGraph()
    G.add_nodes_from(file_details['file_info'][graph_type]['nodes'])
    for edge in file_details['file_info'][graph_type]['edges']:
        source, target = edge['source'], edge['target']
        if source in G.nodes and target in G.nodes:
           G.add_edge(source, target, **{k: v for k, v in edge.items() if k in ['target_inputs', 'target_returns']})
    # Draw graphs
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
    plt.savefig(output_file) # Save the figure
    plt.close()  # Close the figure


def save_python_data(file_details: dict, instruct_list: list, relative_path: Path, output_dir: str) -> None:
    """
    Save Python file details as a YAML file, the instruction data as a JSON file, and code graphs.
    Args:
        file_details (dict): The details extracted from the Python file.
        instruct_list (list): The instruction data extracted from the Python file.
        relative_path (Path): The relative path to the Python file.
        output_dir (str): The directory where the output files will be saved.
    Returns:
        None
    """
    output_subdir = Path(output_dir) / relative_path.parts[0]
    output_subdir.mkdir(parents=True, exist_ok=True)
    base_name = '.'.join(part for part in relative_path.parts)

    # write instrunct.json files
    file_names = [f'{base_name}.instruct.json', f'{base_name}.details.yaml']
    contents = [instruct_list, file_details]

    for file_name, content in zip(file_names, contents):
        write_file(content, output_subdir / file_name)

    try:
        create_code_graph(file_details, base_name, output_subdir)
    except:
        logging.info(f'Error creating graph for {base_name}')
