from ftlangdetect import detect
import sys
import os

def _read_txt_strip_(url):
    file = open(url, 'r', encoding='utf-8')
    lines = file.readlines()
    file.close()
    return [line.strip() for line in lines]


save_path = sys.argv[1]
checkpoint_name = sys.argv[2]
src_language_sequence = sys.argv[3]

src_languages = src_language_sequence.split(",")

supported_languages = [
    "en", "bg", "da", "es", "uk", "hi", "ro", "de", "cs", "pt",
    "nl", "mr", "ur", "sv", "gu", "ar", "fr", "ru", "it", "pl",
    "he", "kn", "bn", "be", "mt", "am", "is", "sd"
]

writing_list = []
for src in src_languages:
    for tgt in supported_languages:
        if src == tgt : continue
        nominator = 0
        results = _read_txt_strip_(os.path.join(save_path, checkpoint_name, "{}-{}.detok.h".format(src, tgt)))
        denominator = len(results)
        for sentence in results:
            result = detect(text=sentence, low_memory=False)["lang"]
            if result != tgt:
                nominator += 1
        score = round((nominator / denominator  * 100), 2)
        writing_list.append(f"{src}-{tgt}\n")
        writing_list.append(f"Ratio: {score} \n")

file = open(os.path.join(save_path, "{}.offtarget".format(str(checkpoint_name))), 'a', encoding='utf-8')
file.writelines(writing_list)
file.close()

