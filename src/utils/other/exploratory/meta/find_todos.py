
import re
import os

def find_todos_and_notes_in_codebase(directory: str) -> None: # Clear structure printing
    todo_pattern = re.compile(r'#\s*TODO[:\s]*(.*)', re.IGNORECASE)
    note_pattern = re.compile(r'#\s*NOTE[:\s]*(.*)', re.IGNORECASE)

    print("\n")
    print("=" * 80)
    print("Searching for TODO comments...\n")
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_number, line in enumerate(f, start=1):
                        match = todo_pattern.search(line)
                        if match:
                            todo_comment = match.group(1).strip()
                            print(f"TODO: {todo_comment} \n(file: {file_path}, line: {line_number})\n")

    print("\n")
    print("=" * 80)
    print("Searching for NOTE comments...\n")
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_number, line in enumerate(f, start=1):
                        match = note_pattern.search(line)
                        if match:
                            note_comment = match.group(1).strip()
                            print(f"NOTE: {note_comment} \n(file: {file_path}, line: {line_number})\n")

if __name__ == "__main__":
    main_dir = "/Users/stijnvanseveren/PythonProjects/MASTERPROEF"

    find_todos_and_notes_in_codebase(main_dir)
