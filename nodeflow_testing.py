from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from src.probabilistic_flow_boosting.extras.datasets.uci_dataset import UCIDataSet
from src.probabilistic_flow_boosting.nodeflow import NodeFlow
from src.probabilistic_flow_boosting.pipelines.reporting.nodes import calculate_metrics_nodeflow

x_train = UCIDataSet(
    filepath_data="data/01_raw/UCI/wine-quality-red/data.txt",
    filepath_index_columns="data/01_raw/UCI/wine-quality-red/index_features.txt",
    filepath_index_rows="data/01_raw/UCI/wine-quality-red/index_train_0.txt"
).load()
y_train = UCIDataSet(
    filepath_data="data/01_raw/UCI/wine-quality-red/data.txt",
    filepath_index_columns="data/01_raw/UCI/wine-quality-red/index_target.txt",
    filepath_index_rows="data/01_raw/UCI/wine-quality-red/index_train_0.txt"
).load()

x_test = UCIDataSet(
    filepath_data="data/01_raw/UCI/wine-quality-red/data.txt",
    filepath_index_columns="data/01_raw/UCI/wine-quality-red/index_features.txt",
    filepath_index_rows="data/01_raw/UCI/wine-quality-red/index_test_0.txt"
).load()

y_test = UCIDataSet(
    filepath_data="data/01_raw/UCI/wine-quality-red/data.txt",
    filepath_index_columns="data/01_raw/UCI/wine-quality-red/index_target.txt",
    filepath_index_rows="data/01_raw/UCI/wine-quality-red/index_test_0.txt"
).load()

model_hyperparams = dict(
    num_trees=[100,300],
    depth=[1,2],
    flow_hidden_dims=[[80, 40], [80, 80, 40]],
    flow_num_blocks=[3]
)
def generate_params_for_grid_search(param_grid):
    return [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
for hyperparams in generate_params_for_grid_search(model_hyperparams):
    nodeflow = NodeFlow(
        input_dim=x_tr.shape[1],
        output_dim=y_tr.shape[1],
        device="cuda",
        **hyperparams
    )
    # x_val.values, y_val.values,
    nodeflow.fit(x_tr.values, y_tr.values, None, None, n_epochs=100, batch_size=1024, max_patience=20, verbose=True)

    nll_train = nodeflow.nll(x_tr.values, y_tr.values)
    nll_val = nodeflow.nll(x_val.values, y_val.values)
    nll_test = nodeflow.nll(x_test.values, y_test.values)
    print(nll_train, nll_test)
    rmse_1, rmse_2, crps = calculate_metrics_nodeflow(nodeflow, x_val.values, y_val.values, num_samples=1000, batch_size=32)
    print(rmse_1, rmse_2, crps)

    # rmse_2 = calculate_rmse_at_2(nodeflow, x_val, y_val, num_samples=1000, batch_size=128)
    # with open("results.csv", "a") as results_f:
    #     results_f.write(f"{hyperparams},{nll_train},{nll_val},{nll_test}")
    # print(hyperparams, nll_train, nll_val, nll_test, rmse_2)
