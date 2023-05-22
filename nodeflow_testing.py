from sklearn.model_selection import train_test_split

from src.probabilistic_flow_boosting.extras.datasets.uci_dataset import UCIDataSet
from src.probabilistic_flow_boosting.nodeflow import NodeFlow

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

x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
nodeflow = NodeFlow(
    input_dim=x_tr.shape[1],
    output_dim=y_tr.shape[1],
    num_trees=20,
    depth=6,
    num_layers=6,
    tree_output_dim=2
)

nodeflow.fit(x_tr.values, y_tr.values, x_val.values, y_val.values, n_epochs=500, batch_size=1024, max_patience=20)

nll_train = nodeflow.nll(x_tr.values, y_tr.values)
nll_val = nodeflow.nll(x_val.values, y_val.values)
nll_test = nodeflow.nll(x_test.values, y_test.values)
print(nll_train, nll_val, nll_test)
