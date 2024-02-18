"""
Requirements
[req01] The code_graph function performs the following:
        a. Extracts the function and class method details from the file summary.
        b. Creates a lookup dictionary for function and class method details.
        c. Creates a directed graph with nodes and edges representing the relationships in the code.
        d. Adds edges for function and class method calls.
        e. Adds edge data to edges.
        f. Returns a dictionary with nodes and edges representing the relationships in the code.
[req02] The extract_control_flow_tree function performs the following:
        a. Extracts control flow tree from AST.
        b. Returns control flow tree.
[req03] The reorganize_control_flow function performs the following:
        a. Gets starting points from the code graph.
        b. Reorganizes the control flow structure recursively.
        c. Returns reorganized control flow structure.
[req04] The get_plantUML_element function performs the following:
        a. Gets plantUML code for each element.
        b. Returns plantUML code for each element.
[req05] The get_plantUML function performs the following: 
        a. Gets plantUML code for entire file.
        b. Returns plantUML code for entire file.
[req06] The get_code_graph function performs the following:
        a. Gets the entire code graph
        b. Gets the control flow structure
        c. Gets the plantUML code
        d. Returns entire code graph, control flow structure and plantUML code
"""
import ast
import json
from typing import Dict, List, Optional, Union
import networkx as nx


def code_graph(
    file_summary: Dict[str, Union[Dict, str]],
) -> Dict[str, Union[List[str], Dict[str, List[str]]]]:
    """
    Create a dictionary representation of file details.
    Args:
        file_summary: Dict[str, Union[Dict, str]]: The details extracted from the file.
    Returns:
        dict: A dictionary with nodes and edges representing the relationships in the code.
    """
    G = nx.DiGraph()

    # Create lookup dictionaries for function and class method details
    function_details_lookup = {}
    for function_def in file_summary["function_defs"]:
        function_details_lookup.update(function_def)
    class_method_details_lookup = {}
    for class_def in file_summary["class_defs"]:
        for (
            class_name,
            class_details,
        ) in class_def.items():  # Extract class name and details
            G.add_node(class_name)  # Add class as a graph node
            for method_name, method_details in class_details["method_defs"].items():
                qualified_method_name = (
                    f"{class_name}.{method_name}"  # Create method fully qualified name
                )
                G.add_node(qualified_method_name)  # Add method as a graph node
                class_method_details_lookup[
                    qualified_method_name
                ] = method_details  # Store method details
                G.add_edge(
                    class_name, qualified_method_name
                )  # Add edge from class to method

    # Helper function to extract edge data from target details
    def get_edge_data_from_details(
        target_details: dict, source_details: dict, target: str
    ) -> dict:
        edge_data = {}
        if target_details:
            edge_data["target_inputs"] = target_details.get("inputs")
            edge_data["target_returns"] = list(set(target_details.get("returns", [])))
        if (
            source_details
            and "call_inputs" in source_details
            and target in source_details["call_inputs"]
        ):
            edge_data["target_inputs"] = source_details["call_inputs"][target]
        return edge_data

    # Helper function to add edge with data
    def add_edge_with_data(
        source: str, target: str, init_method: Optional[str] = None
    ) -> None:
        target_details = class_method_details_lookup.get(
            init_method or target
        ) or function_details_lookup.get(target)
        source_details = function_details_lookup.get(
            source
        ) or class_method_details_lookup.get(source)
        G.add_edge(
            source,
            target,
            **get_edge_data_from_details(target_details, source_details, target),
        )

    # Helper function to add edges for function or class method calls
    def add_edges_for_calls(source_name, calls):
        class_names = [
            list(class_def.keys())[0] for class_def in file_summary["class_defs"]
        ]
        for called in calls:
            called_class_name = called.split(".")[0]
            if called.startswith("self."):
                method_name = called.replace("self.", "")
                fully_qualified_name = f"{source_name.split('.')[0]}.{method_name}"
                if fully_qualified_name in class_method_details_lookup:
                    add_edge_with_data(source_name, fully_qualified_name)
                    continue
            if (
                called in function_details_lookup
                or called in class_method_details_lookup
                or f"{source_name.split('.')[0]}.{called}"
                in class_method_details_lookup
            ):
                add_edge_with_data(source_name, called)
            elif called_class_name in class_names:
                init_method = None
                init_method_name = f"{called}.__init__"
                if init_method_name in class_method_details_lookup:
                    init_method = init_method_name
                add_edge_with_data(source_name, called, init_method)
            else:
                G.add_node(called)
                add_edge_with_data(source_name, called)

    # Add function nodes to graph and edges for function calls
    for function_name in function_details_lookup:
        G.add_node(function_name)
    for func_name, details in function_details_lookup.items():
        add_edges_for_calls(func_name, details["calls"])

    # Add edges for method calls
    for qualified_method_name, details in class_method_details_lookup.items():
        add_edges_for_calls(qualified_method_name, details["calls"])

    # Add edge data to edges and create node and edges to return
    for edge in G.edges:
        source, target = edge
        target_details = function_details_lookup.get(
            target
        ) or class_method_details_lookup.get(target)
        source_details = function_details_lookup.get(
            source
        ) or class_method_details_lookup.get(source)
        edge_data = get_edge_data_from_details(target_details, source_details, target)
        G[source][target].update(edge_data)
    nodes = list(G.nodes)
    edges = [
        {"source": edge[0], "target": edge[1], **edge[2]} for edge in G.edges.data()
    ]

    # remove any nodes that are not either a source of an edge or a target of an edge
    nodes_to_remove = []
    for node in nodes:
        if node not in [edge["source"] for edge in edges] and node not in [
            edge["target"] for edge in edges
        ]:
            nodes_to_remove.append(node)

    return {"nodes": nodes, "edges": edges}


def extract_control_flow_tree(nodes: List[ast.AST]) -> List[Union[str, dict]]:
    """
    Extract control flow tree from AST.
    Args:
        nodes: AST nodes
    Returns:
        control_flow_tree: control flow tree
    """
    control_flow_tree = []
    node_keywords_map = {
        ast.FunctionDef: "def",
        ast.AsyncFunctionDef: "def",
        ast.ClassDef: "class",
        ast.While: "while",
        ast.For: "for",
        ast.AsyncFor: "for",
        ast.With: "with",
        ast.AsyncWith: "with",
        ast.Return: "return",
        ast.If: "if",
        ast.Try: "try",
    }

    for node in nodes:
        node_type = type(node)
        if node_type not in node_keywords_map:
            control_flow_tree.append(ast.unparse(node))
            continue
        keyword = node_keywords_map[node_type]

        if keyword == "def":
            args_str = ", ".join([ast.unparse(arg) for arg in node.args.args])
            key = f"def {node.name}({args_str})"
            value = extract_control_flow_tree(node.body)
            control_flow_tree.append({key: value})
        elif keyword == "class":
            key = f"class {node.name}"
            value = extract_control_flow_tree(node.body)
            control_flow_tree.append({key: value})
        elif keyword == "while":
            key = f"while {ast.unparse(node.test)}"
            value = extract_control_flow_tree(node.body)
            control_flow_tree.append({key: value})
        elif keyword == "for":
            key = f"for {ast.unparse(node.target)} in {ast.unparse(node.iter)}"
            value = extract_control_flow_tree(node.body)
            control_flow_tree.append({key: value})
        elif keyword == "with":
            key = f"with {', '.join([ast.unparse(item) for item in node.items])}"
            value = extract_control_flow_tree(node.body)
            control_flow_tree.append({key: value})
        elif keyword == "return":
            key = "return"
            value = [ast.unparse(node.value)] if node.value is not None else []
            control_flow_tree.append({key: value})
        elif keyword == "if":
            key = f"if {ast.unparse(node.test)}"
            value = extract_control_flow_tree(node.body)
            if_block = {key: value}
            orelse = node.orelse
            while orelse and isinstance(orelse[0], ast.If):
                key = f"elif {ast.unparse(orelse[0].test)}"
                value = extract_control_flow_tree(orelse[0].body)
                if_block[key] = value
                orelse = orelse[0].orelse
            if orelse:
                key = "else"
                value = extract_control_flow_tree(orelse)
                if_block[key] = value
            control_flow_tree.append(if_block)
        elif keyword == "try":
            try_block = extract_control_flow_tree(node.body)
            except_block = []
            for handler in node.handlers:
                h_type = ast.unparse(handler.type) if handler.type is not None else ""
                h_name = (
                    ast.unparse(handler.name)
                    if isinstance(handler.name, ast.Name)
                    else ""
                )
                key = f"except {h_type} as {h_name}:"
                value = extract_control_flow_tree(handler.body)
                except_block.append({key: value})
            control_flow_dict = {"try": try_block, "except": except_block}
            if node.orelse:
                else_block = extract_control_flow_tree(node.orelse)
                control_flow_dict["else"] = else_block
            if node.finalbody:
                finally_block = extract_control_flow_tree(node.finalbody)
                control_flow_dict["finally"] = finally_block
            control_flow_tree.append(control_flow_dict)
        else:
            control_flow_tree.append(ast.unparse(node))

    return control_flow_tree


def reorganize_control_flow(code_graph, control_flow_structure):
    """
    Reorganize control flow structure to match the code graph.
    Args:
        file_details: file details
        control_flow_structure: control flow structure
    Returns:
        reorganized_control_flow_structure: reorganized control flow structure
    """
    # Get starting points from the code graph as those
    targets = [edge["target"] for edge in code_graph["edges"]]
    starting_points = [node for node in code_graph["nodes"] if node not in targets]

    # Define a function to reorganize the structure recursively
    def reorganize_structure(structure, start_points):
        organized, seen = [], set()

        # Iterate through each start point and find matching elements in structure
        for start in start_points:
            for element in structure:
                if isinstance(element, dict):
                    key = next(iter(element))  # Get the first key of the dictionary
                    if (
                        start in key
                    ):  # Add element if it matches the start point and hasn't been seen
                        element_id = json.dumps(element)
                        if element_id not in seen:
                            organized.append(element)
                            seen.add(element_id)
                elif (
                    isinstance(element, str) and start in element
                ):  # Handle string elements
                    organized.append(element)

        # Append elements not included in the organized list
        remaining = [elem for elem in structure if json.dumps(elem) not in seen]
        organized.extend(remaining)
        return organized

    # Reorganize the control flow structure recursively
    return reorganize_structure(control_flow_structure, starting_points)


def get_plantUML_element(element, indentation=""):
    """
    Get plantUML code for each element.
    Args:
        element: element
        indentation: current indentation level
    Returns:
        plantuml_str: plantUML code for each element
    """
    plantuml_str = ""
    if isinstance(element, dict):
        key = next(iter(element))
        value = element[key]
        if key.startswith("def ") or key.startswith("async def "):
            plantuml_str += f"{indentation}def {value} {{\n"
            inner_indentation = indentation + "  "
        elif key.startswith("class "):
            plantuml_str += f"{indentation}class {value} {{\n"
            inner_indentation = indentation + "  "
        elif key.startswith("if "):
            plantuml_str += f"{indentation}if ({value}) {{\n"
            inner_indentation = indentation + "  "
        elif (
            key.startswith("for ")
            or key.startswith("while ")
            or key.startswith("asyncfor ")
        ):
            plantuml_str += f"{indentation}while ({value}) {{\n"
            inner_indentation = indentation + "  "
        elif key.startswith("try "):
            plantuml_str += f"{indentation}try {{\n"
            inner_indentation = indentation + "  "
        elif key.startswith("except "):
            plantuml_str += f"{indentation}catch ({value}) {{\n"
            inner_indentation = indentation + "  "
        else:
            plantuml_str += f"{indentation}:{key};\n"
            inner_indentation = indentation

        if isinstance(value, list):
            for child in value:
                plantuml_str += get_plantUML_element(child, inner_indentation)

        if key.startswith("def ") or key.startswith("class "):
            plantuml_str += f"{indentation}}}\n"
        elif (
            key.startswith("if ")
            or key.startswith("for ")
            or key.startswith("while ")
            or key.startswith("asyncfor ")
            or key.startswith("try ")
            or key.startswith("except ")
        ):
            plantuml_str += f"{indentation}}}\n"

    elif isinstance(element, str):
        plantuml_str += f"{indentation}:{element};\n"

    return plantuml_str


def get_plantUML(control_flow_structure):
    """
    Get plantUML activity diagram code for entire file.
    Args:
        file_details: file details
    Returns:
        plantuml_str: plantUML code for entire file
    """
    plantuml_str = "@startuml\n"
    for element in control_flow_structure:
        plantuml_str += get_plantUML_element(element, "  ")
    plantuml_str += "end\n@enduml"
    return plantuml_str


def get_code_graph(file_summary, file_ast):
    """
    Add code graph and control flow to file details.
    Args:
        file_details: file details
    Returns:
        file_details: file details
    """
    try:
        entire_code_graph = code_graph(file_summary)
        control_flow_tree = extract_control_flow_tree(file_ast.body)
        control_flow_structure = reorganize_control_flow(
            entire_code_graph, control_flow_tree
        )
        plantUML = get_plantUML(control_flow_structure)
    except Exception as e:
        control_flow_structure = [str(e)]
        plantUML = str(e)

    #print('file_summary:', file_summary)
    #print('entire_code_graph:', entire_code_graph)
    #print('control_flow_structure:', control_flow_structure)
    #print('plantUML:', plantUML)

    return entire_code_graph, control_flow_structure, plantUML
