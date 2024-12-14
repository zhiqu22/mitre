# MITRE

Registering Source Token into the Target Language Spaces in Multilingual Neural Machine Translation

This is the repository for reproducing the training described in our paper.

If consider use only, please refer to the HuggingFace page (implementing). 

### Begin

```markdown
parent_dir
|-- MITRE
|   |-- raw_data
|   |   |--floresp-v2.0-rc.3
|   |-- README.md
|-- fairseq
|   |-- models
|   |   |--mitre
|-- mosesdecoder
```

**Note**:  
1. Please manually download [Flores+](https://github.com/openlanguagedata/flores/tags/v2.0-rc.3). 
2. Extract the subfiles of Flores+, and place them in `raw_data/floresp-v2.0-rc.3`.
2. Please manually download `mosesdecoder`.

---

### Envirnment Init

```bash
conda create -n mitre python=3.10
conda activate mitre

conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121




