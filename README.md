# HANU MLA Lab Solutions

Source code and the required input data for the Machine Learning Applications
(MLA) tutorials at Hanoi University, academic year 2026–2027.

## Tutorials

- [Week 1: K-Nearest Neighbours](week-1/Lab1_KNN_Starter/)
- [Week 2: Decision Trees](week-2/Lab2_Starter/)

Each lab directory contains the Python source and its `data/` directory. Run
the commands below from the corresponding lab directory so the scripts can
resolve `./data` correctly.

```bash
cd week-1/Lab1_KNN_Starter
python knn.py
```

```bash
cd week-2/Lab2_Starter
python decision_tree.py
```

Install the shared dependencies with:

```bash
python -m pip install numpy scipy pandas scikit-learn matplotlib reportlab pymupdf
```

Reports, plots, prediction arrays, virtual environments, and Python caches
are generated artifacts and are excluded by `.gitignore`.
