import pandas as pd
import numpy as np

class Portfolio:

    def __init__(self, weights="weights.txt"):
        """
        A constructor can be called with no parameters.
        Otherwise, it may load a pre-saved weights vector.
        Note: If you use a pre-saved weights, than your submission must include this file.

        """
        weights = np.loadtxt(weights, dtype=float,delimiter=',')
        self.portfolio = weights

    def train(self, train_data):
        """
        Input: train_data: a dataframe as downloaded from yahoo finance,
         containing about 5 years of history, with all the training data. The following day (the first that does not appear in the history) is the test day.
        Output: numpy vector of stocks allocations.
        Note: Entries must match original order in the input dataframe!
        """
        pass

    def get_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function generates model's portfolio for the next day.
        Input: train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
            with all the training data. The following day (the first that does not appear in the history) is the test day.
        Output: numpy vector of stocks allocations.
        Note: Entries must match original order in the input dataframe!

        """
        # size = train_data['Adj Close'].columns.size
        # if size>503:
        #     remainder= size-503
        #     np.pad(self.portfolio, (0, remainder), 'constant')

        return self.portfolio

############################################ Our code #########################################

from deepdow.benchmarks import Benchmark, OneOverN, Random
from deepdow.callbacks import EarlyStoppingCallback
from deepdow.data import InRAMDataset, RigidDataLoader, prepare_standard_scaler, Scale
from deepdow.data.synthetic import sin_single
from deepdow.experiments import Run
from deepdow.layers import SoftmaxAllocator
from deepdow.losses import MeanReturns, SharpeRatio, MaximumDrawdown,StandardDeviation
from deepdow.visualize import generate_metrics_table, generate_weights_table, plot_metrics, plot_weight_heatmap
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torch.utils.data import Dataset, DataLoader,random_split
from torch.nn import functional as F
from tqdm import tqdm
import pandas as pd
import random
import deepdow as dd
from deepdow.utils import returns_to_Xy

seed = 48 # The seed we used to get reproducibility results
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Fetch the list of S&P 500 companies from Wikipedia (or another reliable source)
# For example, you can use pandas to scrape the data from Wikipedia
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
tables = pd.read_html(url)
sp500_table = tables[0]
sp500_tickers = sp500_table['Symbol'].tolist()

# Calculate the date range (5 years back from today)
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=5*365)).strftime('%Y-%m-%d')
data = yf.download(sp500_tickers, start=start_date, end=end_date)

# Select the 'Adj Close' prices from the downloaded data
adj_close_data = data['Adj Close']

# Create a Pandas DataFrame to store the data
df = pd.DataFrame(adj_close_data).pct_change()


X, timestamps, y = returns_to_Xy(df,
                                lookback=30,
                                gap=0,

                                horizon=30)

n_samples =  len(df) - 30 - 30 + 1  # 10

assert X.shape == (n_samples, 1, 30, 501)
print(timestamps[0],df.index[30])
assert timestamps[0] == df.index[29]

from deepdow.data import InRAMDataset

dataset = InRAMDataset(X, y, timestamps=timestamps, asset_names=df.columns)

X_sample, y_sample, timestamp_sample, asset_names = dataset[0]


assert isinstance(dataset, torch.utils.data.Dataset)


assert torch.is_tensor(X_sample)

assert torch.is_tensor(y_sample)

assert timestamp_sample == timestamps[0]

from deepdow.data import RigidDataLoader


batch_size = 32

train_indecies = random.sample(range(n_samples),int(0.8*n_samples))
test = set(list(range(n_samples))).difference(set(train_indecies))

train_loader = RigidDataLoader(dataset, batch_size=batch_size,indices=list(range(int(0.8*len(dataset)))))
test_loader = RigidDataLoader(dataset, batch_size=batch_size,indices=list(range(int(0.8*len(dataset)),len(dataset))))

class PortfolioLSTM(nn.Module,Benchmark):
    def __init__(self, input_size, hidden_size, output_size):
        super(PortfolioLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.reshape(x.shape[0],x.shape[2],x.shape[3])
        batch_size = x.size(0)

        out, _ = self.lstm(x)

        out = self.fc(out[:, -1, :])  # Take the last time step's output
        return F.softmax(out,dim=1)


device = "cuda:0" if torch.cuda.is_available() else "cpu"

input_size = 501
hidden_size = 256
output_size = 501

# Instantiate the LSTM Predictor model
network = PortfolioLSTM(input_size, hidden_size, output_size).to(device)

loss =StandardDeviation()  + 2*SharpeRatio()
run = Run(network,
          loss,
          train_loader,
          val_dataloaders={'test': test_loader},
          optimizer=torch.optim.Adam(network.parameters(), amsgrad=True),
          callbacks=[EarlyStoppingCallback(metric_name='loss',
                                           dataloader_name='test',
                                           patience=15)])

history = run.launch(30)

