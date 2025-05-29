import re

# Ensure this path is correct relative to where the script runs
file_path = "src/agents/graphs/basic_agent_graph.py"

try:
    with open(file_path, encoding="utf-8") as f:
        content = f.read()
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit(1)
except Exception as e:
    print(f"Error reading file: {e}")
    exit(1)

# Remove all ASCII control characters except for newlines (\n, \r) and tabs (\t)
# \x00-\x08: Null to Backspace
# \x0b-\x0c: Vertical Tab, Form Feed
# \x0e-\x1f: Shift Out to Unit Separator
# \x7f: Delete
cleaned_content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", content)

try:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(cleaned_content)
    print(f"Successfully cleaned control characters from {file_path}")
except Exception as e:
    print(f"Error writing to file: {e}")
    exit(1)
