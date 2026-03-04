import json

def split_jsonl_file(input_file, output_prefix, lines_per_file=6000):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i in range(0, len(lines), lines_per_file):
        batch_lines = lines[i:i + lines_per_file]
        output_file = f"{output_prefix}_part{i // lines_per_file + 1}.jsonl"

        with open(output_file, 'w', encoding='utf-8') as out_file:
            for line in batch_lines:
                out_file.write(line)

input_file = ""
output_prefix = ""
split_jsonl_file(input_file, output_prefix, lines_per_file=6000)
