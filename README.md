# MITRE

Registering Source Token into the Target Language Spaces in Multilingual Neural Machine Translation

This is the repository for reproducing the training described in our paper.

If consider use only, please refer to the HuggingFace page (implementing). 

### BEGIN: Dirs tree

```markdown
parent_dir
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
2. Please manually download `mosesdecoder`.

### BEGIN: Codes Introduction 
```markdown
mitre
|-- models
|   |-- __init__.py
|   |-- scripts
|   |-- raw_data
|-- modules
|   |-- dicts
|-- __init__.py

### Envirnment Init

```bash
conda create -n mitre python=3.10
conda activate mitre

conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# if there raise any error about cython in installing fairseq, 
# please consider mannully edit pyproject.toml like this:
# vi pyproject.toml
#requires = [...
#  "cython", --> "cython<3.0.0"
#  "numpy>=1.21.3", --> "numpy==1.22.4" # the minimum requirement for spicy, higher version will raise error
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
bash build_data.sh {parent_path}
```

