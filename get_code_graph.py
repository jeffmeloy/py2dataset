import ast
from typing import Dict, List, Optional, Union
import networkx as nx


def code_graph(
    file_summary: Dict[str, Union[Dict, str]],
) -> Dict[str, Union[List[str], Dict[str, List[str]]]]:
    """Create a dictionary representation of file details."""
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
                class_method_details_lookup[qualified_method_name] = (
                    method_details  # Store method details
                )
                G.add_edge(
                    class_name, qualified_method_name
                )  # Add edge from class to method

    def get_edge_data_from_details(
        target_details: dict, source_details: dict, target: str
    ) -> dict:
        """Extract edge data from target details."""
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


    def add_edge_with_data(
        source: str, target: str, init_method: Optional[str] = None
    ) -> None:
        """Helper function to add edge with data"""
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

    def add_edges_for_calls(source_name, calls):
        """Helper function to add edges for function or class method calls"""
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

    return {"nodes": nodes, "edges": edges}


def extract_control_flow_tree(nodes: List[ast.AST]) -> List[Union[str, dict]]:
    """Extract control flow tree from AST."""
    # todo1: make each function, class, and method a partition
    # todo2: add visual indicators or styles to distinguish between functions, classes, and methods
    # todo3: when calling an internal function, class, or method, add the line and arrow to the partition
    # todo4: use different arrow styles or colors to indicate the type of call (e.g., function call, method call, class instantiation)
    # todo5: make try/except like other conditional logic loops (maybe treat more like if loop somehow)
    # todo6: add visual indicators to distinguish between the try, except, else, and finally sections
    # todo7: make grouping better to keep all elements within the same box for imports and non-node_keywords_map elements within each function or method
    # todo8: add visual separators or spacing between different types of elements (e.g., imports, non-keyword elements, control flow statements)
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
            # todo11: add support for visualizing function or method arguments and return values
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
    """Reorganize control flow structure to match the code graph."""
    # todo9: break up long strings to prevent plantuml errors
    # todo10: add line breaks or truncation indicators to visually represent the continuation of long strings
    targets = [edge["target"] for edge in code_graph["edges"]]
    starting_points = [node for node in code_graph["nodes"] if node not in targets]
    visited = set()

    def segregate_control_flow(control_flow_structure):
        imports_globals = [element for element in control_flow_structure if isinstance(element, str)]
        rest_structure = [element for element in control_flow_structure if isinstance(element, dict)]
        return imports_globals, rest_structure

    def dfs_order(graph, start, visited=None, order=None):
        if visited is None:
            visited = set()
        if order is None:
            order = []
        visited.add(start)
        for edge in graph["edges"]:
            if edge["source"] == start and edge["target"] not in visited:
                dfs_order(graph, edge["target"], visited, order)
        order.append(start)  # This appends the node after all its descendants have been handled
        return order

    def map_to_control_flow(execution_order, rest_structure):
        mapped_structure = []
        for node in execution_order:
            for element in rest_structure:
                if (isinstance(element, dict) and node in next(iter(element))) or (isinstance(element, str) and node in element):
                    if element not in mapped_structure:  # Ensure unique elements
                        mapped_structure.append(element)
        return mapped_structure

    # Segregate imports and globals from the rest of the control flow structure
    imports_globals, rest_structure = segregate_control_flow(control_flow_structure)
    # Determine execution order from the code graph
    execution_order = []
    for start in starting_points:
        if start not in visited:
            temp_order = dfs_order(code_graph, start, visited, order=[])
            execution_order.extend(temp_order)

    # Reverse execution order to match logical flow
    execution_order.reverse()
    # Map execution order back to the control flow elements
    organized_rest_structure = map_to_control_flow(execution_order, rest_structure)
    # Combine imports_globals and organized_rest_structure for the final organized structure
    organized_structure = imports_globals + organized_rest_structure

    return organized_structure

def get_plantUML_element(element: dict, indentation="") -> str:
    """Get PlantUML element from control flow structure."""
    plantuml_str = ""

    if isinstance(element, dict):
        key = next(iter(element))
        value = element[key]

        if "def" in key:
            plantuml_str += (
                f'{indentation}:{key};\n'
            )
            inner_indentation = indentation + "    "
            for item in value:
                plantuml_str += get_plantUML_element(item, inner_indentation)

        elif "class" in key or "partition" in key:
            partition_name = key.split(" ")[1]
            plantuml_str += (
                f'{indentation}partition "{partition_name}" {{\n'
            )
            inner_indentation = indentation + "    "
            for item in value:
                plantuml_str += get_plantUML_element(item, inner_indentation)
            plantuml_str += (
                f'{indentation}}}\n'
            )

        elif "if" in key or "while" in key or "for" in key:
            condition = key.split(" ", 1)[1]
            plantuml_str += (
                f'{indentation}if ({condition}) then;\n'
            )
            inner_indentation = indentation + "    "
            for item in value:
                plantuml_str += get_plantUML_element(item, inner_indentation)
            plantuml_str += (
                f'{indentation}endif;\n'
            )

        elif "try" in element:
            plantuml_str += (
                f'{indentation}partition "try" {{\n'
            )
            inner_indentation = indentation + "    "
            for item in element["try"]:
                plantuml_str += get_plantUML_element(item, inner_indentation)
            plantuml_str += (
                f'{indentation}}}\n'
            )

            if "except" in element:
                except_blocks = element["except"]
                if not isinstance(except_blocks, list):
                    except_blocks = [except_blocks]
                for except_block in except_blocks:
                    except_key = next(iter(except_block))
                    except_value = except_block[except_key]
                    except_key = except_key.split(" ", 1)[1]
                    plantuml_str += (
                        f'{indentation}partition "{except_key}" {{\n'
                    )
                    inner_indentation = indentation + "    "
                    for item in except_value:
                        plantuml_str += get_plantUML_element(item, inner_indentation)
                    plantuml_str += (
                        f'{indentation}}}\n'
                    )

            if "else" in element:
                plantuml_str += (
                    f'{indentation}else;\n'
                )
                inner_indentation = indentation + "    "
                for item in element["else"]:
                    plantuml_str += get_plantUML_element(item, inner_indentation)

            if "finally" in element:
                plantuml_str += (
                    f'{indentation}finally;\n'
                )
                inner_indentation = indentation + "    "
                for item in element["finally"]:
                    plantuml_str += get_plantUML_element(item, inner_indentation)

        else:
            plantuml_str += (
                f"{indentation}:{key};\n"
            )

    elif isinstance(element, str):
        plantuml_str += (
            f"{indentation}:{element};\n"
        )

    return plantuml_str

def get_plantUML(control_flow_structure: List[Union[str, dict]]) -> str:
    """Get PlantUML from control flow structure."""
    plantuml_str = "@startuml\n"
    plantuml_str += "start\n"
    for element in control_flow_structure:
        plantuml_str += get_plantUML_element(element, "")
    plantuml_str += "stop\n"
    plantuml_str += "@enduml"
    return plantuml_str


def get_code_graph(file_summary: dict, file_ast: ast.AST) -> (dict, list, str):
    """Add code graph and control flow to file details."""
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

    #print('**********************')
    #print(plantUML)
    #print('**********************')

    return entire_code_graph, control_flow_structure, plantUML
