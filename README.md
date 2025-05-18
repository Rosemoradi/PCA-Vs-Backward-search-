# PCA-Vs-Backward-search-
Compares PCA and Backward Search for dimensionality reduction on synthetic 10D data. Visualizes impact on classification error using LDA. Which keeps accuracy higher while reducing complexity? Discover which method performs best—interpretability vs variance!
#  Dimensionality Reduction: PCA vs Backward Feature Elimination

This project compares two powerful dimensionality reduction techniques—**Principal Component Analysis (PCA)** and **Backward Feature Elimination (BFE)**—to evaluate their effect on classification performance using **Linear Discriminant Analysis (LDA)** on synthetic 10D data.

---

##  Project Overview

We simulate two classes of data with overlapping Gaussian distributions across 10 correlated features and apply:

-  **PCA**: Projects data onto principal components that preserve maximum variance
- ✂ **Backward Search**: Greedily removes the least useful features for classification

Each method is evaluated on how classification error changes as dimensionality is reduced.

---

##  Key Components

###  Dataset
- 2 synthetic classes (1000 samples each)
- 10 continuous features with block-structured covariance
- Ground truth labels: 0 (Class 1), 1 (Class 2)

###  Algorithms Used

| Method | Approach | Evaluation |
|--------|----------|------------|
| PCA | Unsupervised, variance-preserving | Reconstruction MSE & LDA accuracy |
| Backward Search | Supervised, greedy feature removal | Classification error minimization |

---

##  Results

###  PCA
- Retains variance but shows fluctuations in classification error
- Accuracy improves with dimensionality, but not always optimal

###  Backward Search
- More stable and effective in reducing classification error, especially in low dimensions
- Removes features with least contribution to LDA

 **Conclusion**:  
Backward Search yields better accuracy when fewer dimensions are retained, while PCA is faster and still useful when preserving structure.


👩‍💻 Author
Razieh Moradi Graduate Student, McMaster University 📫 moradr1@mcmaster.ca


