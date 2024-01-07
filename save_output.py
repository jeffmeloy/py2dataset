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
import json
import logging
from html import escape
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import networkx as nx
import yaml


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
        if file_type == "json":
            return json.load(f)
        if file_type == "yaml":
            return yaml.load(f, Loader=yaml.SafeLoader)
        return {}


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
    with file_path.open("w") as f:
        if file_type == "json":
            json.dump(data, f, indent=4)
        elif file_type == "yaml":
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

    for json_file in Path(directory).rglob("*.json"):
        try:
            dataset = read_file(json_file)
            if not dataset:
                continue
        except Exception:
            continue

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
        column_width = (
            100 / column_count
        ) 
        for key in dataset[0].keys():
            html_content += f"<th style='width: {column_width}%;'>{key}</th>"
        html_content += """
                    </tr>
                </thead>
                <tbody>
        """
        html_rows = []  
        for entry in dataset:
            row_parts = ["<tr>"]
            for key in entry:
                value = escape(str(entry[key]))
                value = preserve_spacing(value)
                value = value.replace("\n", "<br/>")
                row_parts.append(f"<td>{value}</td>")
            row_parts.append("</tr>")
            html_rows.append("".join(row_parts)) 
        html_content += "".join(html_rows)

        html_content += """
                </tbody>
            </table>
        </body>
        </html>
        """
        html_file_path = json_file.with_suffix(".html")
        try:
            with open(html_file_path, "w", encoding="utf-8") as file:
                file.write(html_content)
        except Exception:
            logging.info(f"Failed saving: {html_file_path}")


def combine_json_files(directory: str, html: bool = False) -> Dict[str, List[Dict]]:
    """
    Combine all JSON files in the output directory into 'instruct.json', and
    then remove duplicates. Also generate a 'training.json' file.
    Args:
        directory (str): The directory where the output files are located.
    Returns:
        A dictionary containing the 'instruct_list' datasets.
    """
    logging.info(f"Combining JSON files in {directory}")

    # Generate instruct.json file
    combined_data = []
    seen = set()
    for json_file in Path(directory).rglob("*.json"):
        if json_file.name == "training.json":
            continue

        try:
            file_data = read_file(json_file)
            if file_data:
                combined_data.extend(file_data)
        except Exception:
            logging.info(f"Failed reading: {json_file}")

    combined_data = [
        item
        for item in combined_data
        if (item["instruction"], item["output"]) not in seen
        and not seen.add((item["instruction"], item["output"]))
    ]
    write_file(combined_data, Path(directory) / "instruct.json")

    # Generate training.json file
    purpose_data = [
        item
        for item in combined_data
        if item["instruction"].startswith("1) Describe the Purpose")
    ]
    code_output = [
        {
            "instruction": f"Define a Python code file that is described as follows:\n{item['output']}",
            "output": item["input"],
        }
        for item in purpose_data
    ]
    write_file(code_output, Path(directory) / "training.json")

    if html:
        logging.info("Converting JSON files to HTML")
        convert_json_to_html(directory)

    return {"instruct_list": combined_data}


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
    # create graph
    graph_type = "entire_code_graph"
    G = nx.DiGraph()
    G.add_nodes_from(file_details["file_info"][graph_type]["nodes"])
    for edge in file_details["file_info"][graph_type]["edges"]:
        source, target = edge["source"], edge["target"]
        if source in G.nodes and target in G.nodes:
            G.add_edge(
                source,
                target,
                **{
                    k: v
                    for k, v in edge.items()
                    if k in ["target_inputs", "target_returns"]
                },
            )

    # draw graph
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        font_weight="bold",
        font_size=8,
        node_shape="s",
        node_size=500,
        width=1,
        arrowsize=12,
    )
    edge_labels = {}
    for edge in G.edges(data=True):
        label = []
        if "target_inputs" in edge[2] and edge[2]["target_inputs"]:
            label.append(f"Inputs: {', '.join(edge[2]['target_inputs'])}")
        if "target_returns" in edge[2] and edge[2]["target_returns"]:
            label.append(f"\nReturns: {', '.join(edge[2]['target_returns'])}")
        edge_labels[(edge[0], edge[1])] = "\n".join(label)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    # save graph
    output_file = output_subdir / f"{base_name}.{graph_type}.png"
    plt.savefig(output_file)
    plt.close()


def save_python_data(
    file_details: dict,
    instruct_list: list,
    relative_path: Path,
    output_dir: str,
) -> None:
    """
    Save file_details as .yaml, the instruct_list as 'json, and code graph as .png
    Args:
        file_details (dict): The details extracted from the Python file.
        instruct_list (list): The instruction data extracted from the Python file.
        relative_path (Path): The relative path to the Python file.
        output_dir (str): The directory where the output files will be saved.
    Returns:
        None
    """
    output_subdir = Path(output_dir) / relative_path.parent
    output_subdir.mkdir(parents=True, exist_ok=True)
    base_name = relative_path.name
    file_names = [f"{base_name}.instruct.json", f"{base_name}.details.yaml"]
    contents = [instruct_list, file_details]
    for file_name, content in zip(file_names, contents):
        write_file(content, output_subdir / file_name)

    try:
        create_code_graph(file_details, base_name, output_subdir)
    except Exception as e:
        logging.info(f"Error creating graph for {base_name}: {e}")
