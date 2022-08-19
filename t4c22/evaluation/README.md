Traffic4cast 2022 Evaluation
============================

Create submissions, test sets and evaluate submissions. Currently only for the core competition (congestion classes `cc`), code for the extended competition (
super-segment average speeds `eta`) forthcoming.


The submission zip for the core competition must have the following file structure:
```
london/labels/cc_labels_test.parquet
madrid/labels/cc_labels_test.parquet
melbourne/labels/cc_labels_test.parquet
```


`create_submission_template.py`
------------------------------
Contains dummy usage examples to create submissions for models both using `T4c22Dataset` or `T4c22GeometricDataset`.

`create_submission.py`
-------------------------
Contains helpers to create submissions for models both using `T4c22Dataset` or `T4c22GeometricDataset`.
See `test_create_submission.py` for usage example to create submissions from model checkpoints.

`generate_tests_sets.py`
-------------------------
This is the code we used to generate the test sets.


`scorecomp.py`
--------------
This is the code we use for evaluation in the leaderboard.
