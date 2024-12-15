import os
import logging
import zipfile
import count_pairs_tool

logging.basicConfig(filename=os.path.join("logs","preprocess", "unzip.log"), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def count_lines(file_path):
    with open(file_path, 'r') as file:
        lines = 0
        for _ in file:
            lines += 1
    return lines

total_pairs = count_pairs_tool.get_total_pairs()
unsuccessful_pairs = []
for pair in total_pairs:
    src, tgt = pair.split("-")
    if src == "ns" or tgt == "ns": continue
    zip_dir= os.path.join("raw_data", "zips")
    unzip_dir = os.path.join("raw_data", "unzips")
    
    unzip_path = os.path.join(unzip_dir, f"{src}-{tgt}")
    os.makedirs(unzip_path, exist_ok=True)
    
    inversed_flag = False
    zip_path = os.path.join(zip_dir, f"{src}-{tgt}.txt.zip")
    
    if not os.path.exists(zip_path):
        inversed_flag = True
        zip_path = os.path.join(zip_dir, f"{tgt}-{src}.txt.zip")
    print(zip_path)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logging.info(f"Finished: {pair}")
    except Exception as e:
        unsuccessful_pairs.append(pair)
        logging.error(f"Unsuccessful unzip: {pair}")
        logging.error(f"Reason is: {e}")
        logging.info(f"Current unsuccessful_pairs: {unsuccessful_pairs}")
        os.rmdir(unzip_path)
        continue
    
    # check length
    src_path = os.path.join(unzip_dir, f"{src}-{tgt}", f"NLLB.{src}-{tgt}.{src}") if not inversed_flag else \
               os.path.join(unzip_dir, f"{src}-{tgt}", f"NLLB.{tgt}-{src}.{src}")
    tgt_path = os.path.join(unzip_dir, f"{src}-{tgt}", f"NLLB.{src}-{tgt}.{tgt}") if not inversed_flag else \
               os.path.join(unzip_dir, f"{src}-{tgt}", f"NLLB.{tgt}-{src}.{tgt}")
    src_length = count_lines(src_path)
    tgt_length = count_lines(tgt_path)
    if src_length == tgt_length:
        logging.info(f"{pair} passed length checker, the size is {src_length}")
    else:
        logging.error(f"{pair} have differnt lengths, the size of {src} is {src_length} ; the size of {tgt} is {tgt_length}")
        unsuccessful_pairs.append(pair)
        logging.info(f"Current unsuccessful_pairs: {unsuccessful_pairs}")
        continue
    
    os.rename(src_path, os.path.join(unzip_dir, f"{src}-{tgt}", f"{src}-{tgt}.{src}"))
    os.rename(tgt_path, os.path.join(unzip_dir, f"{src}-{tgt}", f"{src}-{tgt}.{tgt}"))

    
    




