"""
Return file_details dictionary for python source code.
Requirements:
[req00] The `remove_docstring` function shall:
        a. Accept a Python code string as an argument.
        b. Remove docstrings and comments from the provided code.
        c. Return the sanitized code string.
[req01] The get_all_calls function shall:
        a. Accept a node of type ast.AST as an argument.
        b. Recursively find all function calls in the subtree rooted at the node.
        c. Return a dictionary of all function calls in the subtree rooted at the node.
[req02] The `CodeVisitor` class shall:
        a. Accept the source code as input when instantiated.
        b. Use the AST to extract details about the code.
        c. Inherit from `ast.NodeVisitor`.
        d. Implement `visit_FunctionDef` method to gather details about functions.
        e. Implement `visit_ClassDef` method to gather details about classes.
        f. Implement `extract_details` method to parse information about a given node.
        g. Implement `analyze` method to traverse the AST and 'file_info'. 
        h. Maintain a current class context using the attribute 'current_class'.
[req03] The `code_graph` function shall:
        a. Accept the file summary as input.
        b. Construct a directed graph with nodes and edges using networkx library.
        c. Define elements such as function nodes, class nodes, and method nodes.
        d. Specify edges to represent relationships. 
        e. Return a dictionary representation of the code call graph. 
[req04] The `get_python_file_details` function shall:
        a. Accept a file path as an argument.
        b. Extract info from Python file using the AST and the `CodeVisitor` class.
        c. Include the entire code graph in the returned details.
        d. Return a dictionary encompassing the extracted file details. 
"""
import ast
import logging
import json
from typing import Dict, List, Union
from get_code_graph import get_code_graph


def remove_docstring(code: str, tree: ast.AST) -> str:
    """
    Remove docstrings from the provided Python code.
    This includes top-level module docstrings and docstrings in
    functions, classes, and async functions.
    Args:
        code (str): The source code from which to remove docstrings.
    Returns:
        str: The source code with docstrings removed.
    """
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Str)
    ):
        tree.body.pop(0)

    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
            and node.body
        ):
            first_body_node = node.body[0]
            if isinstance(first_body_node, ast.Expr) and isinstance(
                first_body_node.value, ast.Str
            ):
                node.body.pop(0)

    return ast.unparse(tree)


def get_all_calls(node: ast.AST, calls=None) -> Dict[str, List[str]]:
    """
    Recursively find all function calls in the subtree rooted at `node`,
    including those in class attributes, list comprehensions, and lambda functions.
    Args:
        node (ast.AST): The root node to start the search from.
        calls (List[Tuple[str, List[str]]], optional): Accumulator for function calls found.
    Returns:
        Dict[str, List[str]]: dictionary mapping function calls (as strings) to lists of their arguments.
    """
    if calls is None:
        calls = []
    if isinstance(node, ast.Call):
        calls.append((ast.unparse(node.func), [ast.unparse(arg) for arg in node.args]))
    elif isinstance(node, ast.ClassDef):
        for body_item in node.body:
            if isinstance(body_item, ast.Assign) and isinstance(
                body_item.targets[0], ast.Name
            ):
                get_all_calls(body_item.value, calls)

    for child in ast.iter_child_nodes(node):
        get_all_calls(child, calls)

    # Build the dictionary after traversal
    calls_dict = {}
    for func, args in calls:
        if func not in calls_dict:
            calls_dict[func] = []
        calls_dict[func].extend(args)

    return calls_dict


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

    def __init__(self, code: str, tree: ast.AST) -> None:
        """
        Initialize a new instance of the class.
        Args:
            code: str: The source code.
        """
        self.code: str = code
        self.functions: Dict[str, Dict[str, Union[str, List[str]]]] = {}
        self.classes: Dict[str, Dict[str, Union[str, List[str]]]] = {}
        self.file_info: Dict[str, Union[str, List[str]]] = {}
        self.current_class: str = None
        self.constants: List[str] = []
        self.tree = tree

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Extract details about a function.
        Args:
            node: ast.FunctionDef: The node to visit.
        """
        details = self.extract_details(
            node, "method" if self.current_class else "function"
        )
        if self.current_class:
            self.classes[self.current_class][f"class_method_{node.name}"] = details
        else:
            self.functions[node.name] = details
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        Extract details about a class.
        Args:
            node: ast.ClassDef: The node to visit.
        """
        self.classes[node.name] = self.extract_details(
            node, "class"
        )  # populate class dictionary when class definition found in AST
        self.current_class = node.name  # set current_class to indicate inside a class
        self.generic_visit(node)  # continue AST traversal to the next node
        self.current_class = None  # reset current_class when finished with this class

    def generic_visit(self, node):
        """
        Called if no explicit visitor function exists for a node.
        Args:
            node: ast.AST: The node to visit.
        """
        for child in ast.iter_child_nodes(node):
            child.parent = node
            self.visit(child)

    def visit_Assign(self, node: ast.Assign):
        """
        Get self.constants
        Args:
            node: ast.Assign: The node to visit.
        """
        if isinstance(node.parent, ast.Module):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    value = node.value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        value_repr = f"'{value.value}'"
                    else:
                        value_repr = ast.unparse(value)
                    constant_assignment = f"{target.id}={value_repr}"
                    self.constants.append(constant_assignment)
        self.generic_visit(node)

    def extract_details(
        self, node: ast.AST, node_type: str
    ) -> Dict[str, Union[str, List[str]]]:
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
        node_inputs = (
            [arg.arg for arg in node.args.args]
            if node_type in ["function", "method"]
            else None
        )
        node_variables = list(
            {
                ast.unparse(target)
                for subnode in node_walk
                if isinstance(subnode, ast.Assign)
                for target in subnode.targets
                if isinstance(target, ast.Name)
            }
        )
        if node_inputs:
            node_variables = list(set(node_inputs + node_variables))

        details = {
            f"{node_type}_name": node.name,
            f"{node_type}_code": ast.unparse(node),
            f"{node_type}_docstring": next(
                (
                    n.value.s
                    for n in node_walk
                    if isinstance(n, ast.Expr) and isinstance(n.value, ast.Str)
                ),
                None,
            ),
            f"{node_type}_inputs": node_inputs,
            f"{node_type}_defaults": [ast.unparse(d) for d in node.args.defaults]
            if node_type in ["function", "method"]
            else None,
            f"{node_type}_returns": [
                ast.unparse(subnode.value) if subnode.value is not None else "None"
                for subnode in node_walk
                if isinstance(subnode, ast.Return)
            ],
            f"{node_type}_calls": list(call_data.keys()),
            f"{node_type}_call_inputs": call_data,
            f"{node_type}_variables": node_variables,
            f"{node_type}_decorators": list(
                {ast.unparse(decorator) for decorator in node.decorator_list}
                if node.decorator_list
                else set()
            ),
            f"{node_type}_annotations": list(
                {
                    ast.unparse(subnode.annotation)
                    for subnode in node_walk
                    if isinstance(subnode, ast.AnnAssign)
                    and subnode.annotation is not None
                }
            ),
            f"{node_type}_properties": list(
                {
                    ast.unparse(subnode)
                    for subnode in node_walk
                    if isinstance(subnode, ast.Attribute)
                    and isinstance(subnode.ctx, ast.Store)
                }
            ),
        }
        if node_type in ["class", "method"]:
            if (
                node_type == "method" and self.current_class
            ):  # find attributes defined as self.attribute
                attributes = [
                    target.attr
                    for subnode in node_walk
                    if isinstance(subnode, ast.Assign)
                    for target in subnode.targets
                    if isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ]
                class_attributes = self.classes[self.current_class].setdefault(
                    "class_attributes", []
                )
                class_attributes += attributes
            if node_type == "class":
                details.update(
                    {
                        "class_attributes": [
                            target.attr
                            for subnode in node.body
                            if isinstance(subnode, ast.Assign)
                            for target in subnode.targets
                            if isinstance(target, ast.Attribute)
                        ],
                        "class_methods": [
                            subnode.name
                            for subnode in node.body
                            if isinstance(subnode, ast.FunctionDef)
                        ],
                        "class_inheritance": [ast.unparse(base) for base in node.bases]
                        if node.bases
                        else [],
                        "class_static_methods": [
                            subnode.name
                            for subnode in node.body
                            if isinstance(subnode, ast.FunctionDef)
                            and any(
                                isinstance(decorator, ast.Name)
                                and decorator.id == "staticmethod"
                                for decorator in subnode.decorator_list
                            )
                        ],
                    }
                )
        return details

    def analyze(self, node: ast.AST) -> None:
        """
        Traverse the AST, list all nodes, and populate 'file_info'
        Args:
            node: ast.AST: The node to analyze.
        """
        # Traverse the AST to capture various code details
        self.visit(node)
        node_walk = list(ast.walk(node))

        file_dependencies = {
            alias.name
            for subnode in node_walk
            if isinstance(subnode, ast.Import)
            for alias in subnode.names
        } | {
            subnode.module
            for subnode in node_walk
            if isinstance(subnode, ast.ImportFrom)
        }

        function_defs = [
            {
                func_name: {
                    "inputs": details["function_inputs"],
                    "calls": details["function_calls"],
                    "call_inputs": details["function_call_inputs"],
                    "returns": details["function_returns"],
                }
            }
            for func_name, details in self.functions.items()
        ]

        class_defs = [
            {
                class_name: {
                    "method_defs": {
                        method_name[len("class_method_") :]: {
                            "inputs": details["method_inputs"],
                            "calls": details["method_calls"],
                            "call_inputs": details["method_call_inputs"],
                            "returns": details["method_returns"],
                        }
                        for method_name, details in class_details.items()
                        if method_name.startswith("class_method_")
                    }
                }
            }
            for class_name, class_details in self.classes.items()
        ]

        self.file_info = {
            "file_code": self.code,
            "file_ast": node,
            "file_dependencies": list(file_dependencies),
            "file_functions": list(self.functions.keys()),
            "file_classes": list(self.classes.keys()),
            "file_constants": self.constants,
            "file_summary": {
                "dependencies": list(file_dependencies),
                "function_defs": function_defs,
                "class_defs": class_defs,
            },
            "file_code_simplified": remove_docstring(ast.unparse(node), self.tree),
        }


def get_python_file_details(file_path: str) -> Dict[str, Union[Dict, str]]:
    """
    Extract details from a Python file.
    Args:
        file_path: str: The path to the Python file.
    Returns:
        Dict[str, Union[Dict, str]]: The details extracted from the file.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
            tree = ast.parse(code)
    except (PermissionError, SyntaxError, IOError) as e:
        logging.warning(f"{e} error in file: {file_path}")
        return None

    visitor = CodeVisitor(code, tree)
    visitor.analyze(tree)
    file_details = {
        "file_info": visitor.file_info,
        "functions": visitor.functions,
        "classes": visitor.classes,
    }

    # get code graph items and add to file_details
    file_summary = file_details["file_info"]["file_summary"]
    file_ast = file_details["file_info"]["file_ast"]
    entire_code_graph, control_flow_structure, plant_uml = get_code_graph(
        file_summary, file_ast
    )
    file_details["file_info"]["entire_code_graph"] = entire_code_graph
    file_details["file_info"]["control_flow_structure"] = control_flow_structure
    file_details["file_info"]["plant_uml"] = plant_uml
    file_details["file_info"]["file_summary"] = json.dumps(file_summary).replace(
        '"', ""
    )
    del file_details["file_info"]["file_ast"]

    return file_details
