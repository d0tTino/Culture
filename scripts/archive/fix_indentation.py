import re

# Read the file
with open('tests/integration/test_memory_usage_tracking.py', 'r') as f:
    content = f.read()

# Pattern to find: 'elif 'id' in mem:' followed by a newline and unindented 'retrieved_ids.append(mem['id'])'
pattern = r"(\s+)elif 'id' in mem:(\r?\n)\1retrieved_ids\.append\(mem\['id'\]\)"

# Replacement with proper indentation (add 4 more spaces before retrieved_ids.append)
replacement = r"\1elif 'id' in mem:\2\1    retrieved_ids.append(mem['id'])"

# Replace the pattern
fixed_content = re.sub(pattern, replacement, content)

# Write the fixed file
with open('tests/integration/test_memory_usage_tracking.py', 'w') as f:
    f.write(fixed_content)

print("Fixed indentation in tests/integration/test_memory_usage_tracking.py") 