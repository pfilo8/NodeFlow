import torch

from sklearn.model_selection import train_test_split

from src.probabilistic_flow_boosting.extras.datasets.uci_dataset import UCIDataSet
from src.probabilistic_flow_boosting.pipelines.modeling.utils import setup_random_seed
from src.probabilistic_flow_boosting.tfboost.softtreeflow import SoftTreeFlow

RANDOM_SEED = 42

setup_random_seed(RANDOM_SEED)

x_train = UCIDataSet(
    filepath_data="data/01_raw/UCI/wine-quality-red/data.txt",
    filepath_index_columns="data/01_raw/UCI/wine-quality-red/index_features.txt",
    filepath_index_rows="data/01_raw/UCI/wine-quality-red/index_train_1.txt"
).load()
y_train = UCIDataSet(
    filepath_data="data/01_raw/UCI/wine-quality-red/data.txt",
    filepath_index_columns="data/01_raw/UCI/wine-quality-red/index_target.txt",
    filepath_index_rows="data/01_raw/UCI/wine-quality-red/index_train_1.txt"
).load()

x_test = UCIDataSet(
    filepath_data="data/01_raw/UCI/wine-quality-red/data.txt",
    filepath_index_columns="data/01_raw/UCI/wine-quality-red/index_features.txt",
    filepath_index_rows="data/01_raw/UCI/wine-quality-red/index_test_1.txt"
).load()
y_test = UCIDataSet(
    filepath_data="data/01_raw/UCI/wine-quality-red/data.txt",
    filepath_index_columns="data/01_raw/UCI/wine-quality-red/index_target.txt",
    filepath_index_rows="data/01_raw/UCI/wine-quality-red/index_test_1.txt"
).load()

x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=RANDOM_SEED)
x_tr = torch.Tensor(x_tr.values)
x_val = torch.Tensor(x_val.values)
x_test = torch.Tensor(x_test.values)

y_tr = torch.Tensor(y_tr.values)
y_val = torch.Tensor(y_val.values)
y_test = torch.Tensor(y_test.values)

model = SoftTreeFlow(
    input_dim=x_tr.shape[1],
    output_dim=y_tr.shape[1],
    tree_depth=2,
    flow_num_blocks=1
)

model.eval()

# model.fit(x_tr, y_tr, x_val, y_val, n_epochs=1, batch_size=2000)

with torch.no_grad():
    print('Test 1')
    logprob_test = - model.log_prob(x_test, y_test, batch_size=1000).mean()
    print(logprob_test)

    print('Test 2')
    logprob_test = - model.log_prob(x_test, y_test, batch_size=1000).mean()
    print(logprob_test)

    print('Test 3')
    logprob_test = - model.log_prob(x_test, y_test, batch_size=1000).mean()
    print(logprob_test)


print('Test 1')
logprob_test = - model.log_prob(x_test, y_test, batch_size=1000).mean()
print(logprob_test)

print('Test 2')
logprob_test = - model.log_prob(x_test, y_test, batch_size=1000).mean()
print(logprob_test)

print('Test 3')
logprob_test = - model.log_prob(x_test, y_test, batch_size=1000).mean()
print(logprob_test)
