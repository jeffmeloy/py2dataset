import json
import logging
from html import escape
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import networkx as nx
import yaml


def read_file(file_path: Path) -> Dict:
    """Read a JSON or YAML file and return its contents as a dictionary."""
    file_type = file_path.suffix[1:]
    with file_path.open() as f:
        if file_type == "json":
            return json.load(f)
        if file_type == "yaml":
            return yaml.load(f, Loader=yaml.SafeLoader)
        return {}


def write_file(data: Dict, file_path: Path) -> None:
    """Write a dictionary to a JSON or YAML file."""
    file_type = file_path.suffix[1:]
    with file_path.open("w", encoding="utf-8") as f:
        if file_type == "json":
            json.dump(data, f, indent=4)
        elif file_type == "yaml":
            yaml.SafeDumper.ignore_aliases = lambda *args: True
            yaml.dump(
                data,
                f,
                Dumper=yaml.SafeDumper,
                width=float("inf"),
                sort_keys=False,
                default_flow_style=False,
                indent=4,
                allow_unicode=True,
                encoding="utf-8",
            )


def convert_json_to_html(directory: str) -> None:
    """Convert JSON files within given directory to HTML format and save it."""

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
        column_width = round(100 / column_count, 2)
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


def combine_json_files(
    directory: str, html: bool, questions: Dict
) -> Dict[str, List[Dict]]:
    """Create instruct, training, sharegpt, instruct, document_code json and code_details yaml files."""
    logging.info(f"Combining JSON files in {directory}")

    # Save instruct.json file
    combined_data, code_filename = [], []
    skip_files = {"instruct.json", "sharegpt.json", "document_code.json"}
    for json_file in Path(directory).rglob("*.json"):
        if json_file.name in skip_files:
            continue
        try:
            file_data = read_file(json_file)
            if file_data:
                combined_data.extend(file_data)
                cleaned_name = (
                    json_file.relative_to(directory)
                    .with_suffix("")
                    .as_posix()
                    .replace(".instruct", "")
                )
                code_filename.append(cleaned_name)
        except Exception as e:
            logging.info(f"Failed reading: {json_file}. Error: {e}")
    write_file(combined_data, Path(directory) / "instruct.json")

    # Generate document_code.json file
    logging.info(f"Generating document_code.json file in {directory}")
    purpose_question = [
        item["text"] for item in questions if item["id"] == "file_purpose"
    ][0]
    purpose_question = purpose_question.split("{filename}")[0]
    purpose_data = [
        item
        for item in combined_data
        if item["instruction"].startswith(purpose_question)
    ]
    document_code = [
        {
            "document": item["output"],
            "code": item["input"],
        }
        for item in purpose_data
    ]
    for i, item in enumerate(document_code):
        item["code filename"] = code_filename[i]
    write_file(document_code, Path(directory) / "document_code.json")

    # save sharegpt.json file
    logging.info(f"Generating sharegpt.json file in {directory}")
    system_value = "Use the provided documentation to output the corresponding Python code."

    sharegpt = [
        {
            "conversation": [
                {
                    "from": "system",
                    "value": system_value,
                },
                {
                    "from": "human",
                    "value": f"Create Python code based on this documentation: {item['document']}",
                },
                {"from": "gpt", "value": item["code"]},
            ],
            "nbytes": "0",
            "source": item["code filename"],
        }
        for item in document_code
    ]
    for item in sharegpt:
        nbytes = 0
        for conv in item["conversation"]:
            nbytes += len(conv["value"].encode("utf-8"))
        item["nbytes"] = nbytes
    write_file(sharegpt, Path(directory) / "sharegpt.json")

    # save code_details.yaml file
    code_details = []
    logging.info(f"Combining *.code_details.yaml files in {directory}")
    for yaml_file in Path(directory).rglob("*.code_details.yaml"):
        try:
            with open(yaml_file, "r") as f:
                file_data = f.read()
                code_details.append(file_data)
        except Exception as e:
            logging.info(f"Failed reading: {yaml_file}. Error: {e}")
    with open(Path(directory) / "code_details.yaml", "w") as f:
        f.write("\n".join(code_details))

    if html:
        logging.info("Converting JSON files to HTML")
        convert_json_to_html(directory)

    return {"instruct_list": combined_data}


def create_code_graph(file_details: Dict, base_name: str, output_subdir: Path) -> None:
    """Generate graphs from the file_details and save them as PNG images."""
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

    try:  # save graph
        output_file = output_subdir / f"{base_name}.{graph_type}.png"
        plt.savefig(output_file)
        plt.close()
    except Exception as e:
        logging.error(f"Error saving graph for {base_name}: {e}", exc_info=True)


def save_python_data(
    file_details: dict, instruct_list: list, relative_path: Path, output_dir: str
) -> None:
    """Save Python file details as a YAML file, the instruction data as JSON files,
    and generate and save code graphs.
    """
    output_subdir = Path(output_dir) / relative_path.parent
    output_subdir.mkdir(parents=True, exist_ok=True)
    base_name = relative_path.name
    write_file(instruct_list, output_subdir / f"{base_name}.instruct.json")
    write_file(file_details, output_subdir / f"{base_name}.details.yaml")
    output_text = file_details["file_info"]["code_qa_response"]
    with open(output_subdir / f"{base_name}.code_details.yaml", "w") as f:
        f.write(output_text)

    try:
        create_code_graph(file_details, base_name, output_subdir)
    except Exception as e:
        logging.error(f"Error creating graph for {base_name}: {e}", exc_info=True)
