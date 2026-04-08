# Models

This directory holds third-party model weights used for evaluation.
It is gitignored — copy files here manually.

## Gravity Spy CNN

**File:** `sidd-cqg-paper-O3-model.h5`

This is the O3-era Gravity Spy classifier from the `gravityspy` package
(see `weights/pytorch/README.md` for context).

It lives on CIT at:
```
/home/meesde.boer/gw_learn/GravitySpy/models/sidd-cqg-paper-O3-model.h5
```

Copy it here with:
```bash
rsync -avz --progress \
  cit:/home/meesde.boer/gw_learn/GravitySpy/models/sidd-cqg-paper-O3-model.h5 \
  models/
```

Alternatively, if the `gravityspy` package is installed it may bundle its own
model path — check `gravityspy.__file__` to locate it.
