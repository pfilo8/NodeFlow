import itertools
import subprocess
import sys

import pyaml

datasets_option = int(sys.argv[1])

datasets = {
    0: [
        "uci_boston",
        "uci_concrete",
        "uci_energy"
    ],
    1: [
        "uci_wine_quality",
        "uci_yacht"
    ],
}

depths = [2, 3, 4]
num_trees = [100, 200, 300, 400, 500]
context_dims = [40, 80, 100]
hidden_dims = [[40], [80], [100], [80, 40], [40, 40]]
data = datasets[datasets_option]

for p, d, t, c, h in itertools.product(data, depths, num_trees, context_dims, hidden_dims):
    params = {
        "run": {
            "params": {
                "n_epochs": 100,
                "batch_size": 512,
                "num_samples": 100,
                "random_seed": 42,
                "tree_params":
                    {
                        "max_depth": d,
                        "num_trees": t,
                        "loss_function": "RMSEWithUncertainty",
                        "silent": True
                    },
                "flow_params":
                    {
                        "input_dim": 1,
                        "context_dim": c,
                        "hidden_dims": h
                    }
            },
            "pipeline": p

        }

    }

    filename = f"/tmp/params_{p}_{d}_{t}_{c}_{h}.yml"

    with open(filename, 'w') as f:
        pyaml.dump(params, f)

    command = [
        "kedro", "run", "--config", filename
    ]
    print(command)
    subprocess.run(command)
