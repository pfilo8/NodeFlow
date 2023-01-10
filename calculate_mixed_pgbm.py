import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split

from src.probabilistic_flow_boosting.pipelines.modeling.utils import setup_random_seed
from src.probabilistic_flow_boosting.pgbm import PGBM


def get_dataset(dataset):
    if dataset == 'avocado':
        df = pd.read_csv('data/01_raw/CatData/avocado/avocado.csv', index_col=0)
        x = df.drop(columns=['Date', 'AveragePrice'])
        x = pd.get_dummies(x)
        y = df[['AveragePrice']]
    elif dataset == 'bigmart':
        df = pd.read_csv('data/01_raw/CatData/bigmart/bigmart.csv')
        df['Outlet_Size'] = df['Outlet_Size'].fillna('')
        x = df.drop(columns=['Item_Identifier', 'Item_Outlet_Sales'])
        x = pd.get_dummies(x)
        y = np.log10(df[['Item_Outlet_Sales']])
    elif dataset == 'diamonds':
        df = pd.read_csv('data/01_raw/CatData/diamonds/diamonds.csv', index_col=0)
        x = df.drop(columns=['price'])
        x = pd.get_dummies(x)
        y = np.log10(df[['price']])
    elif dataset == 'diamonds2':
        df = pd.read_csv('data/01_raw/CatData/diamonds2/diamonds_dataset.csv')
        x = df.drop(columns=['id', 'url', 'price', 'date_fetched'])
        x = pd.get_dummies(x)
        y = np.log10(df[['price']])
    elif dataset == 'laptop':
        df = pd.read_csv('data/01_raw/CatData/laptop/laptop_price.csv', index_col=0, engine='pyarrow')
        df['Weight'] = pd.to_numeric(df['Weight'].str.replace('kg', ''))
        df['Ram'] = pd.to_numeric(df['Ram'].str.replace('GB', ''))
        x = df.drop(columns=['Product', 'Price_euros'])
        x = pd.get_dummies(x)
        y = np.log10(df[['Price_euros']])
    elif dataset == 'pakwheels':
        df = pd.read_csv('data/01_raw/CatData/pak-wheels/PakWheelsDataSet.csv', index_col=0)
        x = df.drop(columns=['Name', 'Price'])
        x = pd.get_dummies(x)
        y = np.log10(df[['Price']])
    elif dataset == 'sydney_house':
        df = pd.read_csv('data/01_raw/CatData/sydney_house/SydneyHousePrices.csv')
        x = df.drop(columns=['Date', 'Id', 'sellPrice'])
        x = pd.get_dummies(x)
        y = np.log10(df[['sellPrice']])
    elif dataset == 'wine_reviews':
        df = pd.read_csv('data/01_raw/CatData/wine_reviews/winemag-data_first150k.csv', index_col=0)
        df['country'] = df['country'].fillna('')
        df['province'] = df['province'].fillna('')
        df = df.dropna(subset=['price'])
        x = df.drop(columns=['description', 'price', 'designation', 'region_1', 'region_2', 'winery'])
        x = pd.get_dummies(x)
        y = df[['price']]
    else:
        raise ValueError(f'Invalid dataset {dataset}')
    return x, y


DATASETS_LIST = [
    'avocado',
    'bigmart',
    'diamonds',
    'diamonds2',
    'laptop',
    'pakwheels',
    'sydney_house',
    'wine_reviews'
]

results = []

for dataset in DATASETS_LIST:
    for RANDOM_SEED in range(1, 6):
        setup_random_seed(RANDOM_SEED)
        x, y = get_dataset(dataset)
        ## Data Engineering
        df = pd.read_csv('data/01_raw/CatData/avocado/avocado.csv', index_col=0)

        x = df.drop(columns=['Date', 'AveragePrice'])
        x = pd.get_dummies(x)
        y = df[['AveragePrice']]

        ## Modeling
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_SEED)
        x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=RANDOM_SEED)

        x_tr = torch.Tensor(x_tr.values)
        x_val = torch.Tensor(x_val.values)
        x_test = torch.Tensor(x_test.values)

        y_tr = torch.Tensor(y_tr.values)
        y_val = torch.Tensor(y_val.values)
        y_test = torch.Tensor(y_test.values)

        print(x_train.shape, x_test.shape)


        def mseloss_objective(yhat, y, sample_weight=None):
            gradient = (yhat - y)
            hessian = torch.ones_like(yhat)
            return gradient, hessian


        def rmseloss_metric(yhat, y, sample_weight=None):
            loss = torch.sqrt(torch.mean(torch.square(yhat - y)))
            return loss


        model = PGBM()

        params = {
            'min_split_gain': 0,
            'min_data_in_leaf': 2,
            'max_leaves': 8,
            'max_bin': 64,
            'learning_rate': 0.1,
            'verbose': 0,
            'early_stopping_rounds': 200,
            'feature_fraction': 1,
            'bagging_fraction': 1,
            'seed': RANDOM_SEED,
            'reg_lambda': 1,
            'device': 'gpu',
            'gpu_device_id': 0,
            'derivatives': 'exact',
            'distribution': 'normal',
            'n_estimators': 2000
        }

        model.train(
            train_set=(x_tr, y_tr),
            objective=mseloss_objective,
            metric=rmseloss_metric,
            valid_set=(x_val, y_val),
            params=params
        )

        model.optimize_distribution(x_val, y_val.reshape(-1))

        y_test_dist = model.predict_dist(x_test, n_forecasts=1000)
        crps = model.crps_ensemble(y_test_dist, y_test.reshape(-1)).mean().item()
        nll = model.nll(x_test, y_test.reshape(-1)).mean().item()
        print(dataset, RANDOM_SEED, crps, nll)

        results.append([dataset, RANDOM_SEED, crps, nll])

r = pd.DataFrame(results, columns=['dataset', 'index', 'crps', 'nll'])
r.to_csv('results_raw_mixed_pgbm.csv', index=False)
g = r.groupby(['dataset'])[['crps', 'nll']].agg([np.mean, np.std])
g.to_csv('results_mixed_pgbm.csv')
