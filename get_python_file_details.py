"""
Use AST to extract details from a Python file, determine the code call graph and return it as a dictionary.
Requirements:
[req01] The get_all_calls function shall:
        a. Accept a node as an argument.
        b. Recursively find all function calls in the subtree rooted at the node.
        c. Return a list of all function calls in the subtree rooted at the node.
[req02] The `CodeVisitor` class shall:
        a. Accept the source code as input
        b. Use the AST to extract details about the code.
        c. Inherit from `ast.NodeVisitor`.
        d. Implement the `visit_FunctionDef` method to gather details about functions.
        e. Implement the `visit_ClassDef` method to gather details about classes.
        f. Implement the `extract_details` method to parse information about a given node.
        g. Implement the `analyze` method to traverse the AST, list all nodes within the current file, and populate the 'file_info' attribute with comprehensive file details.
[req03] The `code_graph` function shall:
        a. Accept the file summary and file details as input.
        b. Construct a dictionary with nodes and edges that illustrate code relationships.
        c. Define elements such as function nodes, class nodes, and method nodes.
        d. Specify edges to represent relationships like function calls, method calls, and class inheritance.
        e. Return a dictionary representation of the code graph, aiding in understanding the code's structure and inter-relationships.
[req04] The `get_python_file_details` function shall:
        a. Accept a file path as an argument.
        b. Extract detailed information from the specified Python file using the AST and the `CodeVisitor` class.
        c. Include the entire file graph in the returned details.
        d. Return a dictionary encompassing the extracted file details.
"""
import ast
import re
import json
import logging
import networkx as nx
from typing import Dict, List, Optional, Union

def get_all_calls(node):
    """
    Recursively find all function calls in the subtree rooted at `node`.
    Args:
        node: ast.AST: The node to start the search from.
    Returns:
        list: A list of all function calls in the subtree rooted at `node`.
    """
    calls = {}
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Call):
            calls[ast.unparse(child.func)] = [ast.unparse(arg) for arg in child.args]
        calls.update(get_all_calls(child))
    return calls

class CodeVisitor(ast.NodeVisitor):
    """
    Visitor class for traversing an AST (Abstract Syntax Tree) and extracting details about the code.
    Attributes:
        code (str): The source code.
        functions(Dict): details about functions in the code.
        classes (Dict): details about classes in the code.
        file_info (Dict): details about the file.
    Methods:
        visit_FunctionDef(node: ast.FunctionDef) -> None: 
            Extract details about a function.
        visit_ClassDef(node: ast.ClassDef) -> None: 
            Extract details about a class.
        extract_details(node: ast.AST, node_type: str) -> Dict[str, Union[str, List[str]]]: 
            Extract details about a node.
        analyze(node: ast.AST) -> None: 
            Populate file_info with details about the file.
    """
    def __init__(self, code: str):
        """
        Initialize a new instance of the class.
        Args:
            code: str: The source code.
        Returns:
            None
        """
        self.code: str = code
        self.functions: Dict[str, Dict[str, Union[str, List[str]]]] = {}
        self.classes: Dict[str, Dict[str, Union[str, List[str]]]] = {}
        self.file_info: Dict[str, Union[str, List[str]]] = {}
        self.current_class: str = None
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Extract details about a function.
        Args:
            node: ast.FunctionDef: The node to visit.
        Returns:
            None
        """
        details = self.extract_details(node, 'method' if self.current_class else 'function')
        if self.current_class:
            self.classes[self.current_class][f'class_method_{node.name}'] = details
        else:
            self.functions[node.name] = details
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        Extract details about a class.
        Args:
            node: ast.ClassDef: The node to visit.
        Returns:
            None
        """
        self.classes[node.name] = self.extract_details(node, 'class') # populate class dictionary when class definition found in AST
        self.current_class = node.name  # set current_class to indicate inside a class
        self.generic_visit(node) # continue AST traversal to the next node
        self.current_class = None  # reset current_class when finished with this class
    
    def extract_details(self, node: ast.AST, node_type: str) -> Dict[str, Union[str, List[str]]]:
        """
        Extract details about a node.
        Args:
            node: ast.AST: The node to extract details from.
            node_type: str: The type of node.
        Returns:
            Dict[str, Union[str, List[str]]]: The details extracted from the node.
        """
        node_walk = list(ast.walk(node))
        call_data = get_all_calls(node)
        details = {
            f"{node_type}_name": node.name, 
            f"{node_type}_code": ast.unparse(node),
            f"{node_type}_ast": ast.dump(node, include_attributes=True), 
            f"{node_type}_docstring": ast.get_docstring(node),
            f"{node_type}_inputs": [arg.arg for arg in node.args.args] if node_type in ['function', 'method'] else None,
            f"{node_type}_defaults": [ast.unparse(d) for d in node.args.defaults] if node_type in ['function', 'method'] else None,
            f"{node_type}_returns": [ast.unparse(subnode.value) if subnode.value is not None else "None" for subnode in node_walk if isinstance(subnode, ast.Return)],
            f"{node_type}_calls": list(call_data.keys()),
            f"{node_type}_call_inputs": call_data, 
            f"{node_type}_variables": list({ast.unparse(target) for subnode in node_walk if isinstance(subnode, ast.Assign) for target in subnode.targets if isinstance(target, ast.Name)}),
            f"{node_type}_decorators": list({ast.unparse(decorator) for decorator in node.decorator_list} if node.decorator_list else set()),
            f"{node_type}_annotations": list({ast.unparse(subnode.annotation) for subnode in node_walk if isinstance(subnode, ast.AnnAssign) and subnode.annotation is not None}),
            f"{node_type}_properties": list({ast.unparse(subnode) for subnode in node_walk if isinstance(subnode, ast.Attribute) and isinstance(subnode.ctx, ast.Store)}),
        }  
        if node_type in ['class', 'method']:
            if node_type == 'method' and self.current_class: # find attributes defined as self.attribute
                attributes = [target.attr for subnode in node_walk if isinstance(subnode, ast.Assign) for target in subnode.targets if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self']
                if attributes: # if this class already has some attributes, add to them
                    if "class_attributes" in self.classes[self.current_class]:
                        self.classes[self.current_class]["class_attributes"].extend(attributes)
                    else: # otherwise, start a new list of attributes for this class
                        self.classes[self.current_class]["class_attributes"] = attributes
            if node_type == 'class':
                details.update({
                    "class_attributes": [target.attr for subnode in node.body if isinstance(subnode, ast.Assign) for target in subnode.targets if isinstance(target, ast.Attribute)],
                    "class_methods": [subnode.name for subnode in node.body if isinstance(subnode, ast.FunctionDef) and subnode.name != "__init__"],
                    "class_inheritance": [ast.unparse(base) for base in node.bases] if node.bases else [],
                    "class_static_methods": [subnode.name for subnode in node.body if isinstance(subnode, ast.FunctionDef) and subnode.name != "__init__" and any(isinstance(decorator, ast.Name) and decorator.id == "staticmethod" for decorator in subnode.decorator_list)],
                    })
        return details

    def analyze(self, node: ast.AST) -> None:
        """
        Traverse the AST rooted at 'node', create a list of all nodes within the current file, and populate 'file_info' with file details
        Args:
            node: ast.AST: The node to analyze.
        Returns:
            None
        """
        node_walk = list(ast.walk(node))
        self.visit(node)
        self.file_info = {
            "file_code": self.code,
            "file_ast" : ast.dump(node),
            "file_dependencies": list({alias.name for subnode in node_walk if isinstance(subnode, ast.Import) for alias in subnode.names} | {subnode.module for subnode in node_walk if isinstance(subnode, ast.ImportFrom)}),
            "file_functions": list(self.functions.keys()),
            "file_classes": list(self.classes.keys()),
        }
        
        # add file_summary to file_info
        function_defs = [{func_name: {"inputs": details["function_inputs"], "calls": details["function_calls"], "call_inputs": details["function_call_inputs"], "returns": details["function_returns"]}} for func_name, details in self.functions.items()]
        class_defs = []
        for class_name, class_details in self.classes.items():
            method_defs = {}
            for method_name, details in class_details.items():
                if method_name.startswith('class_method_'):
                    method_defs[method_name[len('class_method_'):]] = {"inputs": details["method_inputs"], "calls": details["method_calls"], "call_inputs": details["method_call_inputs"], "returns": details["method_returns"]}
            class_defs.append({class_name: {"method_defs": method_defs}})
        self.file_info["file_summary"] = { 'dependencies': self.file_info["file_dependencies"], 'function_defs' : function_defs, 'class_defs' : class_defs}

def code_graph(file_summary: Dict[str, Union[Dict, str]]) -> Dict[str, Union[List[str], Dict[str, List[str]]]]:
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
    for function_def in file_summary['function_defs']:
        function_details_lookup.update(function_def)
    class_method_details_lookup = {}
    for class_def in file_summary['class_defs']:
        for class_name, class_details in class_def.items(): # Extract class name and details
            G.add_node(class_name) # Add class as a graph node
            for method_name, method_details in class_details['method_defs'].items():
                qualified_method_name = f'{class_name}.{method_name}' # Create method fully qualified name
                G.add_node(qualified_method_name) # Add method as a graph node
                class_method_details_lookup[qualified_method_name] = method_details  # Store method details 
                G.add_edge(class_name, qualified_method_name) # Add edge from class to method

    # Helper function to extract edge data from target details
    def get_edge_data_from_details(target_details: dict, source_details: dict, target: str) -> dict:
        edge_data = {}
        if target_details:
            edge_data['target_inputs'] = target_details.get('inputs')
            edge_data['target_returns'] = list(set(target_details.get('returns', [])))
        if source_details and 'call_inputs' in source_details and target in source_details['call_inputs']:
            edge_data['target_inputs'] = source_details['call_inputs'][target]     
        return edge_data

    # Helper function to add edge with data
    def add_edge_with_data(source: str, target: str, init_method: Optional[str] = None) -> None:
        target_details = class_method_details_lookup.get(init_method or target) or function_details_lookup.get(target)
        source_details = function_details_lookup.get(source) or class_method_details_lookup.get(source)
        G.add_edge(source, target, **get_edge_data_from_details(target_details, source_details, target))

    # Helper function to add edges for function or class method calls
    def add_edges_for_calls(source_name, calls):
        class_names = [list(class_def.keys())[0] for class_def in file_summary['class_defs']]
        for called in calls:
            called_class_name = called.split('.')[0]
            if called.startswith("self."):
                method_name = called.replace("self.", "")
                fully_qualified_name = f"{source_name.split('.')[0]}.{method_name}"
                if fully_qualified_name in class_method_details_lookup:
                    add_edge_with_data(source_name, fully_qualified_name)
                    continue
            if (
                called in function_details_lookup or 
                called in class_method_details_lookup or 
                f"{source_name.split('.')[0]}.{called}" in class_method_details_lookup
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
    for function_name in function_details_lookup.keys():
        G.add_node(function_name)
    for func_name, details in function_details_lookup.items():
        add_edges_for_calls(func_name, details['calls'])

    # Add edges for method calls
    for qualified_method_name, details in class_method_details_lookup.items():
        add_edges_for_calls(qualified_method_name, details['calls'])

    # Add edge data to edges and create node and edges to return
    for edge in G.edges:
        source, target = edge
        target_details = function_details_lookup.get(target) or class_method_details_lookup.get(target)
        source_details = function_details_lookup.get(source) or class_method_details_lookup.get(source)
        edge_data = get_edge_data_from_details(target_details, source_details, target)
        G[source][target].update(edge_data)
    nodes = list(G.nodes)
    edges = [{"source": edge[0], "target": edge[1], **edge[2]} for edge in G.edges.data()]
    return {"nodes": nodes, "edges": edges}


def get_python_file_details(file_path: str) -> Dict[str, Union[Dict, str]]:
    """
    Extract details from a Python file.
    Args:
        file_path: str: The path to the Python file.
    Returns:
        Dict[str, Union[Dict, str]]: The details extracted from the file.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
            code = f.read()
            tree = ast.parse(code)
    except:
        logging.warning(f"Permission denied or syntax error in file: {file_path}")
        return None 

    visitor = CodeVisitor(code)
    visitor.analyze(tree)
    file_details = {'file_info': visitor.file_info, 'functions': visitor.functions, 'classes': visitor.classes}
    file_details['file_info']['entire_code_graph'] = code_graph(file_details['file_info']['file_summary'])
    file_details['file_info']['file_summary'] = json.dumps(file_details['file_info']['file_summary']).replace('\"','')
    return file_details