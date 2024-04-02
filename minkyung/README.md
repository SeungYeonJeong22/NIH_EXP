# Causal Medical Image Classification

In this project, I propose a new way to explore the integration of two investigations that yielded promising results: (i) my previous **Causality-Driven CNNs** (e.g. _Mulcat_, see below), and (ii) causal intervention/**backdoor adjustment** by [this](https://github.com/zc2024/Causal_CXR)).

To learn more about **causality-driven CNNs**, take a look at my two papers [**Journal** paper (under review)](https://arxiv.org/abs/2309.10399) and [**ICCV 2023** paper](https://openaccess.thecvf.com/content/ICCV2023W/CVAMD/html/Carloni_Causality-Driven_One-Shot_Learning_for_Prostate_Cancer_Grading_from_MRI_ICCVW_2023_paper.html), and at the **code** I have made public in this repository: [causality_conv_nets](https://github.com/gianlucarloni/causality_conv_nets).

#### Notes
(Present) :construction: This project repo is still a _work in progress_: the code still needs to be cleaned and tidied. If something about my implementation is not clear to you, please leave me an [email](mailto:gianluca.carloni@isti.cnr.it).

(November 2023) At the moment, I haven't integrated causality-driven CNNs in this code, leaving it as a future step. However, I have coded from scratch some missing functionalities and fixed the main issues associated with the [Causal_CXR code](https://github.com/zc2024/Causal_CXR)), which prevented it from being reproducible. This required me to implement certain functionalities from scratch, even at the PyTorch library level. Unfortunately, pushing the "modified torch" library to this GitHub repository was not feasible, therefore I will only describe the modified functions in detail below.

(October 2023) Modified the original code to work smoothly with the `torch` Distributed Data Parallel settings (multi-GPUs) on my Institute's NVIDIA DGX system.

## Modify PyTorch to have a custom Transformer network

Given a regular PyTorch installation with parameters:
`__version__ = '1.13.1'`, `debug = False`, `cuda = '11.6'`, `git_version = '49444c3e546bf240bed24a101e747422d1f8a0ee'`, and `hip = None`,
I customized specific functions and modules  to accomplish the following goal: the **Transformer** network, which uses multi-head attention, needs to yield not one but two sets of feature sets. In the paper, they are associated with _Q_ and _Q*_ and represent the _causal_ and _confounding_ features (disentanglement).

To do so, we modified `torch/nn/functional.py` and `torch/nn/modules/activation.py` as described below:

### 1) Modify `torch/nn/functional.py`

we created a new version of the `multi_head_attention_forward()` function, called `multi_head_attention_forward_disentangled()`, that returns -> `Tuple[Tensor, Tensor, Optional[Tensor]]`. In there, once we obtained the regular `attn_output_weights`, we calculated the confounding/trivial features by inverting the attention weights. Intuitively, if the softmaxed attention weights indicate the locations that influence the class, by taking `1-softmax()` we get the complement. The resulting object has the same shape and size as the actual attn_output:

```
attn_output_inverted = torch.bmm((1-attn_output_weights), v)
attn_output_inverted = attn_output_inverted.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
attn_output_inverted = linear(attn_output_inverted, out_proj_weight, out_proj_bias)
attn_output_inverted = attn_output_inverted.view(tgt_len, bsz, attn_output_inverted.size(1))
```
    
Then, whenever the code was `attn_output = attn_output.squeeze(1)`, we added our variant: `attn_output_inverted = attn_output_inverted.squeeze(1)`, and ultimately returned `return attn_output, attn_output_inverted, attn_output_weights` or `return attn_output, attn_output_inverted, None` depending on the flags.

### 2) Modify `torch/nn/modules/activation.py`

we created a new version of the `MultiheadAttention` class, called `MultiheadAttentionDisentangled`, that utilizes the previously defined `multi_head_attention_forward_disentangled()` in place of the usual `multi_head_attention_forward()`.

```
        attn_output, attn_output_inverted, attn_output_weights = F.multi_head_attention_forward_disentangled(...)
```
and returns `return attn_output.transpose(1, 0), attn_output_inverted.transpose(1, 0), attn_output_weights` or `return attn_output, attn_output_inverted, attn_output_weights`, depending on the flag used.

## Prepare the datasets in pickle format

In [prepare_dataset_pickle.py](https://github.com/gianlucarloni/causal_medimg/blob/main/prepare_dataset_pickle.py) I implemented a simple script that creates the pickle version of the Chest X-Ray dataset. In fact, this was somehow assumed to exist in the original version of the code, without giving any indication of how to obtain it.
