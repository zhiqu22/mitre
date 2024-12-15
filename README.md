# MITRE

Registering Source Tokens to Target Language Spaces in Multilingual Neural Machine Translation

This is the repository for reproducing the data mining and pre-training described in our paper.

If consider using our pre-trained model only, please refer to the HuggingFace page (in implementing, coming soon). 

### BEGIN: Dirs tree

```markdown
root_path
|-- MITRE
|   |-- scripts
|   |-- raw_data
|   |   |--floresp-v2.0-rc.3
|   |-- dicts
|   |-- README.md
# follows are created by build_data.sh
|   |-- mitre-bin
|   |-- test-bin
|   |-- results
|   |-- tables
|   |-- ...
|-- fairseq
|   |-- models
|   |   |--mitre
|-- mosesdecoder
```

**Note**:  
1. Please manually download [Flores+](https://github.com/openlanguagedata/flores/tags/v2.0-rc.3). 
2. Extract the subfiles of Flores+, and place them in `raw_data/floresp-v2.0-rc.3`.
3. Please manually download `mosesdecoder`.

### BEGIN: Codes Introduction

MITRE is a decoder-only model.

In order to reuse the MNMT training tools of Fairseq, 
we save the encoder-decoder architecture in training to reduce the cost of implementing data collection, batching, and loss computation.
Specifically, we simply set the encoder layer to **0** to keep the feature of decoder-only.

```markdown
mitre
|-- models
|   |-- __init__.py
|   |-- transformer_encoder_register.py
|   |-- transformer_decoder_register.py
|   |-- transformer_register.py
|-- modules
|   |-- __init__.py
|   |-- mutihead_attention_register.py
|   |-- transformer_layer_register.py
|-- __init__.py
```

### Environment Init

```bash
conda create -n mitre python=3.10
conda activate mitre

conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# if there raise any error about cython or spicy in installing fairseq, 
# please consider mannully edit pyproject.toml like this:
# vi pyproject.toml
#requires = [...
#  "cython", --> "cython<3.0.0"
#  "numpy>=1.21.3", --> "numpy==1.22.4"
#  "torch>=1.10", --> "torch>=2.0.1"
#]

git clone https://github.com/NVIDIA/apex
cd apex
git checkout 6943fd26e04c59327de32592cf5af68be8f5c44e
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./

cd ..
mkdir fairseq/models
mv mitre fairseq/models/mitre

# build pre-training dataset for mitre
# 1. SentencePiece codes (L83-104) are masked because bpe model and vocab are provided.
#    if you want to run it by your self, the cost of memory is around 700G
# 2. In L18-19, there are two parameters using in sharding bpe and binarizing data
#    both denominator and nominator are initiailized as 1
#    if you want to speed up this process, you can set denominator to n
#    to divide the language pairs into n parts, then mannully set nominator to m,
#    which means to process the m-th part pairs.
bash build_data.sh {root_path}
```

### Train

If you want to reproduce the pre-training of MITRE, please run
```bash
bash train.sh {root_path}
```
**Note**:  
1. You have to manually update and confirm params in L3~14, which are used to distributed training.  
2. The type of 400M model is transformer_register_big; the type of 900M model is transformer_register_massive


### Evaluation

You can evaluate spBLEU, chrF++, and COMET scores by run
```bash
# a tool to save cost in loading model when using the inference of fairseq.
mv MITRE/scripts/multilingual_generate.py fairseq/fairseq_cli/multilingual_generate.py
# libs for spbleu and chrf
pip install sacrebleu
pip install "sacrebleu[ja]"
pip install "sacrebleu[ko]"
# libs for comet
pip install unbabel-comet
# libs for making table
pip install openpyxl

bash evaluation.sh {root_path} {EXPERIMENT_NAME} {EXPERIMENT_ID} {PT_ID}
```
**Note**:  
1. PT_ID can be a single pt name, "averaged" or "all".
2. This script supports running with multiple gpus, please manually update params.

### Experiments on EC-40

Scripts used in EC-40, which have a style similar to the MITRE's main scripts, are saved in MITRE/ec_40_scripts.

When you want to reproduce the experiments on EC-40, please download the training data in the [repository of EC-40](https://github.com/Smu-Tan/ZS-NMT-Variations/tree/main).

Do not forget to `mv {root_path}/MITRE/ec_40_scripts {root_path}/ec_40_scripts`.

Additionally, when you want to measure off-target ratio, you have to run `pip install ftlangdetect` at first.