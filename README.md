eBay University Machine Learning Competition 2025 — Project Overview
This project was developed as part of the eBay 2025 University Machine Learning Competition, a challenge focused on solving real-world multi-label text classification problems using eBay product listing data. The competition centers on accurately predicting multiple relevant product categories (labels) from free-text inputs such as titles and descriptions. The evaluation metric is the Averaged Fβ-score, a strict yet informative measure that emphasizes both precision and recall across all predicted labels.

Approach
My approach begins with thorough data preprocessing, including text normalization, tokenization, and handling of class imbalance. I leverage both classical and transformer-based NLP techniques, with an emphasis on high-recall multi-label classification architectures. Multiple models and loss functions were benchmarked, including Binary Cross-Entropy with Logits, Focal Loss, and various threshold optimization strategies to maximize the Fβ-score.

I conducted extensive experiments using pretrained language models such as BERT and RoBERTa, fine-tuned for multi-label output. To enhance performance, techniques such as dynamic threshold tuning, label correlation modeling, and inference-time ensembling were explored. Model performance was continuously validated using stratified k-fold cross-validation, ensuring robustness and generalizability across product domains.

This repository includes the complete training pipeline, evaluation framework, and submission logic, all designed to optimize predictive accuracy under the competition's constraints. My goal throughout was not only to achieve a high score, but also to maintain a clean, modular codebase that could be easily extended and adapted to similar classification challenges.

GLiNER Credits:
@misc{zaratiana2023gliner,
      title={GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer}, 
      author={Urchade Zaratiana and Nadi Tomeh and Pierre Holat and Thierry Charnois},
      year={2023},
      eprint={2311.08526},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
