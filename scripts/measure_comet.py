from comet import download_model, load_from_checkpoint
import sys
import os
import count_pairs_tool

supported_languages = count_pairs_tool.languages

def _read_txt_strip_(url):
    file = open(url, 'r', encoding='utf-8')
    lines = file.readlines()
    file.close()
    return [line.strip() for line in lines]

save_path = sys.argv[1]
checkpoint_name = sys.argv[2]
tgt  = sys.argv[3]
model_path = download_model("Unbabel/wmt22-comet-da")

model = load_from_checkpoint(model_path)

writing_list = []
for src in supported_languages:
    if src == tgt: continue
    refs = _read_txt_strip_(os.path.join(save_path, checkpoint_name, "{}-{}.r".format(src, tgt)))
    hypos = _read_txt_strip_(os.path.join(save_path, checkpoint_name, "{}-{}.h".format(src, tgt)))
    srcs = _read_txt_strip_(os.path.join(save_path, checkpoint_name, "{}-{}.s".format(src, tgt)))
    data = [{"src": src_text, "mt": mt_text, "ref": ref_text} for src_text, mt_text, ref_text in zip(srcs, hypos, refs)]
    model_output = model.predict(data, batch_size=100, gpus=1)
    score = round(model_output.system_score * 100, 2)
    writing_list.append(f"{src}-{tgt}\n")
    writing_list.append(f"Score: {score} \n")

file = open(os.path.join(save_path, "{}.comet".format(str(checkpoint_name))), 'a', encoding='utf-8')
file.writelines(writing_list)
file.close()
