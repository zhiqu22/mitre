from language_id_map import translate_language_code
import sys
import count_pairs_tool

function_name = sys.argv[1]
if function_name == "get_total_pairs":
    print(count_pairs_tool.get_total_pairs())
elif function_name == "get_partial_pairs":
    pairs = count_pairs_tool.get_total_pairs()
    num_parts = int(sys.argv[2])
    idx_parts = int(sys.argv[3]) - 1
    chunk_size = len(pairs) // num_parts
    remainder = len(pairs) % num_parts
    start = idx_parts * chunk_size + min(idx_parts, remainder)
    end = start + chunk_size + (1 if idx_parts < remainder else 0)
    print(pairs[start:end])
elif function_name == "get_bidirection_pairs":
    print(count_pairs_tool.get_bidirection_pairs())
elif function_name == "get_languages":
    print(count_pairs_tool.languages)
elif function_name == "write_langauges":
    languages = count_pairs_tool.languages
    path = sys.argv[2]
    with open(path, 'w', encoding='utf-8') as f:
        for language in languages:
            f.write(language + '\n')
elif function_name == "write_pairs":
    bidirection_pairs = count_pairs_tool.get_bidirection_pairs()
    path = sys.argv[2]
    with open(path, 'w', encoding='utf-8') as f:
        f.write(",".join(bidirection_pairs))
elif function_name == "translate_language_code":
    input_str = sys.argv[2]
    source_type = sys.argv[3]
    output_type = sys.argv[4]
    print(translate_language_code(input_str, source_type, output_type))
