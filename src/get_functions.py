import os
import inspect
import importlib
from enum import Enum
from pathlib import Path


def is_enum(cls):
    return issubclass(cls, Enum)


def get_classes_from_module(module):
    classes = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        # if not is_enum(obj) and obj.__module__ == module.__name__:
        classes.append(name)
    return classes


def crawl_functions_folder(functions_folder):
    for root, _, files in os.walk(functions_folder):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                module_name = os.path.splitext(file)[0]
                module_path = os.path.relpath(root, start=functions_folder).replace(
                    os.sep, "."
                )
                full_module_name = (
                    f"{module_path}.{module_name}" if module_path else module_name
                )
                try:
                    module = importlib.import_module(
                        f"{functions_folder}.{full_module_name}"
                    )
                    classes = get_classes_from_module(module)
                    if classes:
                        print(f"Module: {full_module_name}")
                        for cls in classes:
                            print(f"  - Class: {cls}")
                except ImportError as e:
                    print(f"Could not import module {full_module_name}: {e}")


if __name__ == "__main__":
    functions_folder = "functions"
    crawl_functions_folder(functions_folder)
