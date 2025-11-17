import argparse
from tokenizers import Tokenizer
import os
import pandas as pd
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import json
import shutil # Import for file copying

# --- (The combine_tokenizers and adjust_config functions remain the same) ---

def combine_tokenizers(old_tokenizer, new_tokenizer, save_dir):
    # Load both the json files, take the union, and store it
    # Note: XTTS uses a slightly different vocab structure in its config, but this code assumes
    # standard BPE vocab.json format (token -> id).
    json1 = json.load(open(os.path.join(old_tokenizer, 'vocab.json'), encoding='utf-8'))
    json2 = json.load(open(os.path.join(new_tokenizer, 'vocab.json'), encoding='utf-8'))

    # Create a new vocabulary
    new_vocab = {}
    idx = 0
    # Add words from first tokenizer (XTTS)
    for word in json1.keys():
        if word not in new_vocab.keys():
            new_vocab[word] = idx
            idx += 1

    # Add words from second tokenizer (Persian)
    for word in json2.keys():
        if word not in new_vocab.keys():
            new_vocab[word] = idx
            idx += 1

    # Make the directory if necessary
    os.makedirs(save_dir, exist_ok=True)

    # Save the vocab
    with open(os.path.join(save_dir, 'vocab.json'), 'w', encoding='utf-8') as fp:
        json.dump(new_vocab, fp, ensure_ascii=False)

    # Merge the two merges file.
    # Concatenate them, but ignore the first line of the second file
    os.system('cat {} > {}'.format(os.path.join(old_tokenizer, 'merges.txt'), os.path.join(save_dir, 'merges.txt')))
    os.system('tail -n +2 -q {} >> {}'.format(os.path.join(new_tokenizer, 'merges.txt'), os.path.join(save_dir, 'merges.txt')))


# --- The key modified function ---
def extend_tokenizer(args):
    
    root = os.path.join(args.output_path, "XTTS_v2.0_original_model_files/")
    # 1. Save existing XTTS tokenizer parts (using the old vocab.json)
    existing_tokenizer = Tokenizer.from_file(os.path.join(root, "vocab.json"))
    old_tokenizer_path = os.path.join(root, "old_tokenizer/")
    os.makedirs(old_tokenizer_path, exist_ok=True)
    existing_tokenizer.model.save(old_tokenizer_path)

    # Define the path where you've downloaded the mshojaei77 files
    # >>> CHANGE THIS PATH TO WHERE YOUR tokenizer.json IS LOCATED <<<
    PERSIAN_TOKENIZER_SOURCE = "/path/to/mshojaei77_download_folder/tokenizer.json" 
    
    # 2. LOAD PRE-TRAINED PERSIAN TOKENIZER AND EXPORT IT
    
    # Path to temporarily store the Persian vocab.json and merges.txt
    new_tokenizer_path = os.path.join(root, "new_tokenizer_persian/")
    os.makedirs(new_tokenizer_path, exist_ok=True)

    print(f" > Loading pre-trained Persian tokenizer from: {PERSIAN_TOKENIZER_SOURCE}")
    
    # Load the single tokenizer.json file
    persian_tokenizer = Tokenizer.from_file(PERSIAN_TOKENIZER_SOURCE)
    
    # Export the BPE model components (vocab.json and merges.txt)
    # The output directory is new_tokenizer_path
    persian_tokenizer.model.save(new_tokenizer_path) 
    
    # Rename the exported vocab.json to match the expected name if necessary, 
    # though model.save usually names it vocab.json
    
    # Optional: Copy special tokens map if needed for future use
    shutil.copy(
        os.path.join(os.path.dirname(PERSIAN_TOKENIZER_SOURCE), 'special_tokens_map.json'),
        os.path.join(new_tokenizer_path, 'special_tokens_map.json')
    )

    # 3. Combine the tokenizers
    merged_tokenizer_path = os.path.join(root, "merged_tokenizer/")
    print(" > Combining XTTS and Persian tokenizers...")
    combine_tokenizers(
        old_tokenizer_path,
        new_tokenizer_path, # Now contains the exported vocab.json and merges.txt
        merged_tokenizer_path
    )

    # 4. Load the merged files and save the final XTTS vocab.json
    tokenizer = Tokenizer.from_file(os.path.join(root, "vocab.json"))
    tokenizer.model = tokenizer.model.from_file(os.path.join(merged_tokenizer_path, 'vocab.json'), os.path.join(merged_tokenizer_path, 'merges.txt'))
    
    # Add the language token for the new language
    tokenizer.add_special_tokens([f"[{args.language}]"])

    # Final step: Overwrite the original XTTS vocab.json with the merged one
    tokenizer.save(os.path.join(root, "vocab.json"))
    print(f" > Successfully merged and saved tokenizer to: {os.path.join(root, 'vocab.json')}")

    # Cleanup temporary directories
    os.system(f'rm -rf {old_tokenizer_path} {new_tokenizer_path} {merged_tokenizer_path}')

# --- (The adjust_config and main functions remain the same) ---
def adjust_config(args):
    config_path = os.path.join(args.output_path, "XTTS_v2.0_original_model_files/config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    if args.language not in config["languages"]:
        config["languages"] += [args.language]
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_path", type=str, required=True, help="Base output directory for XTTS files.")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to the training metadata CSV (not used in this version but kept for compatibility).")
    parser.add_argument("--language", type=str, required=True, help="Language code for the new language (e.g., 'fa').")
    parser.add_argument("--extended_vocab_size", default=2000, type=int, help="Kept for compatibility, but not used since we are loading a pre-trained tokenizer.")

    args = parser.parse_args()

    # Ensure you replace the placeholder path in extend_tokenizer before running!
    extend_tokenizer(args)
    adjust_config(args)
