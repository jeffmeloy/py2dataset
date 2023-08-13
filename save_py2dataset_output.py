"""
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
    """
    file_type = file_path.suffix[1:]
    with file_path.open('w') as f:
        if file_type == 'json':
            json.dump(data, f, indent=4)
        elif file_type == 'yaml':
            yaml.SafeDumper.ignore_aliases = lambda *args: True
            yaml.dump(data, f, Dumper=yaml.SafeDumper, sort_keys=False)


def convert_json_to_html(directory: str) -> None:

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
        if file == 'qa.json':
            qa_data = combined_data.copy()
        if file == 'instruct.json':
            instruct_data = combined_data.copy()
        
        # Create purpose-specific file
        purpose_data = []
        for item in combined_data:
            if item[keys[file_names.index(file)]].startswith('What is the purpose'):
                purpose_data.append(item)
        if purpose_data:
            purpose_file_path = file_path.with_name(file_path.stem + '_purpose.json')
            write_file(purpose_data, purpose_file_path)

        # Reset combined_data for the next iteration
        combined_data = []

    # remove duplicate "input" info in the instruct.json and instruct_purpose.json information 
    # to make a cleaned_instruct.json cleaned_instruct_purpose_json
    file_names = ['instruct.json', 'instruct_purpose.json']
    for file in file_names:
        seen_inputs = set()
        file_path = Path(directory) / file
        if not file_path.exists():
            continue
        dataset = read_file(file_path)
        if not dataset:
            continue
        for item in dataset:
            if item['input'] in seen_inputs:
                item['input'] = ''
            else:
                seen_inputs.add(item['input'])
        cleaned_instruct_file_path = Path(directory) / ('cleaned_'+ file) 
        write_file(dataset, cleaned_instruct_file_path)

    # save html file for each json file in the output directory
    if html:
        convert_json_to_html(directory)

    return {'qa_list': qa_data, 'instruct_list': instruct_data}


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