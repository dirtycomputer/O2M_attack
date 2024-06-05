# Cross-Modality Jailbreak and Mismatched Attacks on Medical Multimodal Large Language Models

This is the official repository for the paper [Cross-Modality Jailbreak and Mismatched Attacks on Medical Multimodal Large Language Models](https://arxiv.org/pdf/2405.20775).

## Abstract

Security concerns related to Large Language Models (LLMs) have been extensively explored, yet the safety implications for Multimodal Large Language Models (MLLMs), particularly in medical contexts (MedMLLMs), remain insufficiently studied. This paper delves into the underexplored security vulnerabilities of MedMLLMs, especially when deployed in clinical environments where the accuracy and relevance of question-and-answer interactions are critically tested against complex medical challenges. By combining existing clinical medical data with atypical natural phenomena, we redefine two types of attacks: mismatched malicious attack (2M-attack) and optimized mismatched malicious attack (O2M-attack).

Using our own constructed voluminous 3MAD dataset, which covers a wide range of medical image modalities and harmful medical scenarios, we conduct a comprehensive analysis and propose the MCM optimization method, which significantly enhances the attack success rate on MedMLLMs. Evaluations with this dataset and novel attack methods, including white-box attacks on LLaVA-Med and transfer attacks on four other state-of-the-art models, indicate that even MedMLLMs designed with enhanced security features are vulnerable to security breaches. Our work underscores the urgent need for a concerted effort to implement robust security measures and enhance the safety and efficacy of open-source MedMLLMs, particularly given the potential severity of jailbreak attacks and other malicious or clinically significant exploits in medical settings.

## Code and Dataset

Our code is available at [GitHub Repository](https://github.com/dirtycomputer/O2M_attack.git).

## Warning

Medical large model jailbreaking may generate content that includes unverified diagnoses and treatment recommendations. Always consult professional medical advice.

## Citation

If you find our work helpful, please consider citing the following paper:

```
@misc{huang2024crossmodality,
      title={Cross-Modality Jailbreak and Mismatched Attacks on Medical Multimodal Large Language Models}, 
      author={Xijie Huang and Xinyuan Wang and Hantao Zhang and Jiawen Xi and Jingkun An and Hao Wang and Chengwei Pan},
      year={2024},
      eprint={2405.20775},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```

# Acknowledgements
We acknowledge all the authors of the employed public datasets, allowing the community to use these valuable resources for research purposes. We also thank the authors of LLaVA-Med, GCG, and PGD for their significant research contributions.


