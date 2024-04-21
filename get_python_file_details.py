import ast
import logging
import json
from typing import Dict, List, Union
from get_code_graph import get_code_graph


def remove_docstring(tree: ast.AST) -> str:
    """Remove docstrings from the provided Python code."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Str)
            ):
                node.body.pop(0)
    return ast.unparse(tree)


def get_all_calls(node: ast.AST, calls: List[tuple] = None) -> Dict[str, List[str]]:
    """Recursively find all function calls in the subtree rooted at `node`."""
    if calls is None:
        calls = []

    if isinstance(node, ast.Call):
        calls.append((ast.unparse(node.func), [ast.unparse(arg) for arg in node.args]))
    elif isinstance(node, ast.ClassDef):
        for body_item in node.body:
            if isinstance(body_item, ast.Assign):
                get_all_calls(body_item.value, calls)

    for child in ast.iter_child_nodes(node):
        get_all_calls(child, calls)

    return {func: args for func, args in calls}


class CodeVisitor(ast.NodeVisitor):
    """Visitor class for traversing an AST and extracting details about the code."""

    def __init__(self, code: str, tree: ast.AST) -> None:
        """Initialize a new instance of the class."""
        self.code = code
        self.tree = tree
        self.functions: Dict[str, Dict[str, Union[str, List[str]]]] = {}
        self.classes: Dict[str, Dict[str, Union[str, List[str]]]] = {}
        self.file_info: Dict[str, Union[str, List[str]]] = {}
        self.current_class: str = None
        self.constants: List[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Extract details about a function.
        Args:
            node (ast.FunctionDef): The function definition node.
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
        """Extract details about a class."""
        self.classes[node.name] = self.extract_details(node, "class")
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = None

    def visit_Assign(self, node: ast.Assign) -> None:
        """Get constants defined in the global scope or class attributes."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                value = node.value
                value_repr = (
                    f"'{value.value}'"
                    if isinstance(value, ast.Constant) and isinstance(value.value, str)
                    else ast.unparse(value)
                )
                self.constants.append(f"{target.id}={value_repr}")
        self.generic_visit(node)

    def extract_details(
        self, node: ast.AST, node_type: str
    ) -> Dict[str, Union[str, List[str]]]:
        """Extract details about a node."""
        node_walk = list(ast.walk(node))
        call_data = get_all_calls(node)
        node_inputs = (
            [arg.arg for arg in node.args.args]
            if node_type in ["function", "method"]
            else None
        )
        node_variables = [
            ast.unparse(target)
            for subnode in node_walk
            if isinstance(subnode, ast.Assign)
            for target in subnode.targets
            if isinstance(target, ast.Name)
        ]
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
            f"{node_type}_decorators": [
                ast.unparse(decorator) for decorator in node.decorator_list
            ]
            if node.decorator_list
            else [],
            f"{node_type}_annotations": [
                ast.unparse(subnode.annotation)
                for subnode in node_walk
                if isinstance(subnode, ast.AnnAssign) and subnode.annotation is not None
            ],
            f"{node_type}_properties": [
                ast.unparse(subnode)
                for subnode in node_walk
                if isinstance(subnode, ast.Attribute)
                and isinstance(subnode.ctx, ast.Store)
            ],
        }

        if node_type in ["class", "method"]:
            if node_type == "method" and self.current_class:
                attributes = [
                    target.attr
                    for subnode in node_walk
                    if isinstance(subnode, ast.Assign)
                    for target in subnode.targets
                    if isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ]
                self.classes[self.current_class].setdefault(
                    "class_attributes", []
                ).extend(attributes)
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

    def analyze(self) -> None:
        """Traverse the AST and populate 'file_info' with details about the file."""
        self.visit(self.tree)
        node_walk = list(ast.walk(self.tree))

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
            "file_ast": self.tree,
            "file_dependencies": list(file_dependencies),
            "file_functions": list(self.functions.keys()),
            "file_classes": list(self.classes.keys()),
            "file_constants": self.constants,
            "file_summary": {
                "dependencies": list(file_dependencies),
                "function_defs": function_defs,
                "class_defs": class_defs,
            },
            "file_code_simplified": remove_docstring(self.tree),
        }


def get_python_file_details(file_path: str) -> Dict[str, Union[Dict, str]]:
    """Extract details from a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            code = file.read()
    except (PermissionError, IOError) as e:
        logging.warning(f"{e} error in file: {file_path}")
        return None

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        logging.warning(f"{e} error in file: {file_path}")
        return None

    visitor = CodeVisitor(code, tree)
    visitor.analyze()
    file_details = {
        "file_info": visitor.file_info,
        "functions": visitor.functions,
        "classes": visitor.classes,
    }

    file_summary = file_details["file_info"]["file_summary"]
    file_ast = file_details["file_info"]["file_ast"]
    entire_code_graph, control_flow_structure, plant_uml = get_code_graph(file_summary, file_ast)
    file_details["file_info"]["entire_code_graph"] = entire_code_graph
    file_details["file_info"]["control_flow_structure"] = control_flow_structure
    file_details["file_info"]["plant_uml"] = plant_uml
    file_details["file_info"]["file_summary"] = json.dumps(file_summary).replace(
        '"', ""
    )
    del file_details["file_info"]["file_ast"]

    return file_details
