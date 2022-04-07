import torch.nn as nn

from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data, DataSet, DataLoader
from torch_geometric.nn import GATConv, Sequential
from torch_geometric.nn.glob import global_mean_pool


class SiameseGraphDataset(Dataset):
    def __init__(self, pairs: list[tuple[Data, Data, int]]):
        self.pairs = pairs
        
    def __getitem__(self, index: int) -> tuple[Data, Data, int]:
        left, right, y = self.pairs[index]
        
        return left, right, y
    
    def __len__(self) -> int:
        return len(self.pairs)


class Net(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size:int,
        output_size: int,
        heads: int
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(NUMBER_EMBEDDINGS, input_size)
        
        self.conv1 = GATConv(input_size, hidden_size, heads=heads)
        self.lin1 = nn.Linear(input_size, heads * hidden_size)
        self.conv2 = GATConv(heads * hidden_size, hidden_size, heads=heads)
        self.lin2 = nn.Linear(heads * hidden_size, heads * hidden_size)
        self.conv3 = GATConv(heads * hidden_size, output_size, heads=heads, concat=False)
        self.lin3 = nn.Linear(heads * hidden_size, output_size)

    def encode(self, data: Data) -> torch.Tensor:
        emb = self.embedding(data.x)

        batch = (
            torch.tensor([0] * len(data.x), dtype=torch.long)
            if data.batch is None
            else data.batch
        )

        x = (self.conv1(emb, data.edge_index) + self.lin1(emb)).relu()
        x = (self.conv2(x, data.edge_index) + self.lin2(x)).relu()
        x = self.conv3(x, data.edge_index) + self.lin3(x)

        return global_mean_pool(x, batch)


    def forward(self, pair: tuple[Data, Data]):
        out_left = self.encode(pair[0])
        out_right = self.encode(pair[1])
        return out_left, out_right


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    train_sample_size: int
) -> float:
    model.train()

    total_loss = num_graphs = 0
    for batch in train_loader:
        y = batch[2] * 2 - 1
        
        optimizer.zero_grad()
        out = model(batch)
        batch_loss = criterion(out[0], out[1], y)
        batch_loss.backward()
        optimizer.step()
        
        batch_size = len(y)
        total_loss += float(batch_loss) * batch_size
        num_graphs += batch_size

        if num_graphs >= train_sample_size:
            break

    total_loss = total_loss / num_graphs
    return total_loss


def test(model: nn.Module, sim_fx: nn.Module, test_loader: DataLoader) -> float:
    model.eval()

    target, prediction = [], []
    for batch in test_loader:
        out = model(batch)
        y_prob = sim_fx(out[0], out[1])
        y_prob = y_prob.cpu().detach()
        y_prob = torch.clamp(y_prob, min=0, max=1)

        target.append(batch[2].numpy())
        prediction.append(y_prob.numpy())
    
    return roc_auc_score(np.concatenate(target), np.concatenate(prediction))
