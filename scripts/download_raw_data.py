import os
import logging
import requests
import count_pairs_tool
from language_id_map import translate_language_code

logging.basicConfig(filename=os.path.join("logs", "preprocess", "download.log"), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

total_pairs = count_pairs_tool.get_total_pairs()

unsuccessful_pairs = []
for pair in total_pairs:
    src, tgt = pair.split("-")
    try:
        logging.info(f"Downloading: {pair}")
        response = requests.get(f"https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/{src}-{tgt}.txt.zip", stream=True)
        response.raise_for_status()
        with open(os.path.join("raw_data", "zips", f"{src}-{tgt}.txt.zip"), 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        logging.info(f"Finished: {pair}")
    except requests.exceptions.RequestException as e_1:
        try:
            logging.info(f"Trying inversed pair: {tgt}-{src}, because {pair} is unscuccessful.")
            response = requests.get(f"https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/{tgt}-{src}.txt.zip", stream=True)
            response.raise_for_status()
            with open(os.path.join("raw_data", "zips", f"{tgt}-{src}.txt.zip"), 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            logging.info(f"Finished: {pair}")
        except requests.exceptions.RequestException as e_2:  
            unsuccessful_pairs.append(pair)
            src_lang, tgt_lang = translate_language_code(src, "m2m", "language"), translate_language_code(tgt, "m2m", "language")
            logging.error(f"Unsuccessful: {src_lang} to {tgt_lang}")
            logging.info(f"Reason of forward downloading is: {e_1}")
            logging.info(f"Reason of inversed downloading is: {e_2}")
            logging.info(f"Current unsuccessful pairs are: {unsuccessful_pairs}")
