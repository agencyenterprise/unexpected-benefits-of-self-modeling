# Run MNIST experiments

## Pre-requisites

Ensure you have the requirements installed:

```bash
pip install -r requirements.txt
```

Run the experiment scripts:

```bash
python run_experiment.py
python run_rlct_calculation.py
```

Then extract the tensorboard events:

```bash
python extract_logs_to_csv.py
```