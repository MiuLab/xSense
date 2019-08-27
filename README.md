## xSense; PyTorch Implementation
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![PyTorch 1.1.0](https://img.shields.io/badge/pytorch-1.1.0-green.svg?style=plastic)

This repository contains the official PyTorch implementation of the following paper:

> **xSense: Learning Sense-Separated Sparse Representations and Textual Definitions for Explainable Word Sense Networks**<br>
> *Ting-Yun Chang, *Ta-Chung Chi, *Shang-Chi Tsai, Yun-Nung Chen<br>
> https://arxiv.org/pdf/1809.03348.pdf
>
> **Abstract:** *Word embeddings are difficult to interpret due to their dense vector nature.
This paper focuses on interpreting embeddings for various aspects, including sense separation in the vector dimensions and definition generation.
Specifically, our algorithm projects the target word embedding to a high-dimensional sparse vector and picks the specific dimensions that can best explain the semantic meaning of the target word by the encoded contextual information, where the sense of the target word can be indirectly inferred.
Then an RNN generates the textual definition of the target word in the human-readable form, enabling direct interpretation of the corresponding word embedding. 
This paper also introduces a large and high-quality context-definition dataset that consists of sense definitions together with multiple example sentences per 
polysemous word, which is a valuable resource for definition modeling and word sense disambiguation.*

### Training
```bash
$ bash run.sh
```

### Evaluation
```bash
$ bash eval.sh
```

### Reference:

Main paper to be cited

```
@article{chang2018xSense,
  title={xSense: Learning Sense-Separated Sparse Representations and Textual Definitions for Explainable Word Sense Networks},
  author={Chang, Ting-Yun and Chi, Ta-Chung and Tsai, Shang-Chi and Chen, Yun-Nung},
  journal={arXiv preprint https://arxiv.org/abs/1809.03348},
  year={2018}
}
```

