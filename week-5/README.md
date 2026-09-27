# Week 5 — SVM and regularization

Open `MLA_Week5_SVM_Regularisation_Lab_Solved.ipynb` in Jupyter or Google Colab
and run all cells in order. The notebook contains the completed Parts A–J,
embedded outputs and plots, and an AI-assisted worked reflection. The original
`Student` notebook is preserved.

The data are generated locally with seed 42; no external dataset is required.
The solution uses the starter's scikit-learn 1.x API (`penalty=None`, `l1`,
and `l2`). Its setup output records the versions used for verification.

Implementation notes:

- The scratch SVM records both objective terms after each parameter update.
- Part C retains the `LinearSVC` comparison and adds a linear `SVC` comparison
  because `LinearSVC` also penalizes the intercept.
- Part H recalculates `C = 1 / (lambda * N_fold)` for every training fold,
  selects lambda without the test data, refits on training plus validation,
  and evaluates the test set once.
- Part I applies the unit change consistently to validation features and checks
  that training-only standardization removes its effect on predictions.
- Part J distinguishes its hypothetical dropped feature from the actual L1 fit.

The group report is `report_artifacts/Group05_CLC02_Lab5.pdf`. It follows
notebook Parts A–J and uses the previous reports as formatting references,
as requested by the submitting member. It includes four figures and the complete
code appendix. Rebuild it with `report_artifacts/build_report.py` using ReportLab;
the generator reads the executed notebook and the repository report metadata.
