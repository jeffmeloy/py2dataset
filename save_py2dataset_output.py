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
        d. Write the combined data to 'qa.json' and 'instruct.json' files.
        e. Generate and save 'qa_purpose.json', 'qa_instruct.json', and 'qa_cleaned_instruct.json' files.
        f. Convert the merged JSON files to HTML format if the html flag is set to True.
        g. Return the 'qa_list' and 'instruct_list' datasets.
[req05] The `create_code_graph` function shall:
        a. Accept details of a Python file, a base name, and an output directory as arguments.
        b. Generate code graphs based on the provided file details.
        c. Save the graphs as PNG images in the specified output directory.
[req06] The `save_python_data` function shall:
        a. Accept details of a Python file, a base name, and an output directory as arguments.
        b. Save the details of the Python file as a YAML file.
        c. Save the QA and instruction data as JSON files.
        d. Generate and save code graphs if the graph flag is set to True.
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
                table {border-collapse: collapse;width: 100%;}
                th, td {border: 1px solid black; padding: 8px; text-align: left; white-space: pre-line;}
                td:nth-child(2) { width: 50%; overflow-wrap: anywhere;}
            </style>
        </head>
        <body>
            <table>
                <thead>
                    <tr>
        """
        for key in dataset[0].keys():
            html_content += f"<th>{key}</th>"
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
                if key == "input":  
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


def combine_json_files(directory: str, html: bool) -> Dict[str, List[Dict]]:
    """
    Combine all JSON files in the output directory into 'qa.json' and 'instruct.json', and then remove duplicates.
    Args:
        directory (str): The directory where the output files are located.
        html (bool): Whether to convert JSON files to HTML format.
    Returns:
        A dictionary containing the 'qa_list' and 'instruct_list' datasets.
    """
    qa_data = []
    instruct_data = []
    for file_name in ['qa.json', 'instruct.json']:
        file_path = Path(directory) / file_name
        combined_data = []
        for json_file in Path(directory).rglob(f'*.{file_name}'):
            combined_data.extend(read_file(json_file))
        combined_data = list({i['question' if file_name == 'qa.json' else 'instruction']: i for i in combined_data}.values())
        write_file(combined_data, file_path)
        if file_name == 'qa.json':
            qa_data = combined_data.copy()
        else:
            instruct_data = combined_data.copy()
            # Create purpose-specific file
            purpose_data = [item for item in combined_data if item['instruction'].startswith('What is the purpose')]
            if purpose_data:
                purpose_file_path = file_path.with_name(file_path.stem + '_purpose.json')
                write_file(purpose_data, purpose_file_path)
            # Remove duplicate "input" fields in instruct.json and instruct_purpose.json
            dataset = read_file(file_path)
            seen_inputs = set()
            for item in dataset:
                if item['input'] in seen_inputs:
                    item['input'] = ''
                else:
                    seen_inputs.add(item['input'])
            write_file(dataset, Path(directory) / ('cleaned_'+ file_name))

    # Save html file for each json file in the output directory
    if html:
        convert_json_to_html(directory)

    return {'qa_list': qa_data, 'instruct_list': instruct_data}


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
    for graph_type in ['internal_code_graph', 'entire_code_graph']:
        # Create graphs
        output_file = output_subdir / f'{base_name}.{graph_type}.png'  
        G = nx.DiGraph()
        # Add nodes
        G.add_nodes_from(file_details['file_info'][graph_type]['nodes'])
        # Add edges
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


def save_python_data(file_details: dict, qa_list: list, instruct_list: list, relative_path: Path, output_dir: str, graph: bool, html: bool) -> None:
    """
    Save Python file details as a YAML file, the QA and instruction data as JSON files, and code graphs if the graph flag is set to True.
    Args:
        file_details (dict): The details extracted from the Python file.
        qa_list (list): The QA data extracted from the Python file.
        instruct_list (list): The instruction data extracted from the Python file.
        relative_path (Path): The relative path to the Python file.
        output_dir (str): The directory where the output files will be saved.
        graph (bool): Whether to generate code graphs.
        html (bool): Whether to convert JSON files to HTML format.
    Returns:
        None
    """
    output_subdir = Path(output_dir) / relative_path.parts[0]
    output_subdir.mkdir(parents=True, exist_ok=True)
    base_name = '.'.join(part for part in relative_path.parts)

    # write qa.json and instrunct.json files
    file_names = [f'{base_name}.qa.json', f'{base_name}.instruct.json', f'{base_name}.details.yaml']
    contents = [qa_list, instruct_list, file_details]

    for file_name, content in zip(file_names, contents):
        write_file(content, output_subdir / file_name)

    if graph: # Create code graph images
        try:
            create_code_graph(file_details, base_name, output_subdir)
        except:
            logging.info(f'Error creating graph for {base_name}')
