# Selective OCR router example

This dependency-free Python example accompanies the article **Building a Selective OCR Router Without Test-Set Leakage**.

It demonstrates four evaluation boundaries:

- deterministic folds assigned by source group;
- out-of-fold risk predictions for calibration;
- a frozen abstention threshold;
- label-free routing of sealed test candidates.

Requirements: Python 3.11 or newer.

```powershell
python -m unittest -v
```

Expected result: five tests pass.

The example is educational and is not the production CXT-Select implementation. The related public protocol and reviewer artifact are available at [Zenodo](https://doi.org/10.5281/zenodo.21947556) under CC BY 4.0.

