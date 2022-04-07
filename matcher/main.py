
from .data import get_dataset_as_dataframe
from .model import Net, train, test
from .service import (
    get_negatives,
    graph_from_string,
    parse_entities,
    sample_positives,
    train_test_split
)
from .tfidf import get_tfidf_score

from torch_geometric.loader import DataLoader

EPOCHS = 100


def main() -> None:
    # Prepare data
    birkbeck_df = get_dataset_as_dataframe('birkbeck')
    corpora_df = parse_entities(birkbeck_df)
    positiv_df = sample_positives(corpora_df)
    negativ_df = get_negatives(corpora_df)
    train_df, test_df = train_test_split(positiv_df, negativ_df)
    
    # Get benchmark score
    print(f"Benchmark: {get_tfidf_score(train_df, test_df)}")

    # Train GCN
    train_graphs = [
    (
        graph_from_string(word),
        graph_from_string(misspell),
        y
    )
    for _, (word, misspell, y) in train_df.iterrows()
    ]

    test_graphs = [
        (
            graph_from_string(word),
            graph_from_string(misspell),
            y
        )
        for _, (word, misspell, y) in test_df.iterrows()
    ]

    train_loader = DataLoader(SiameseGraphDataset(train_graphs), batch_size=32, shuffle=True)
    test_loader = DataLoader(SiameseGraphDataset(test_graphs), batch_size=32, shuffle=False)

    model = Net(input_size=32, hidden_size=256, output_size=32, heads=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-4)
    criterion = nn.CosineEmbeddingLoss(reduction="mean")
    sim_fx = nn.CosineSimilarity(dim=1)

    for i in range(1, EPOCHS + 1):
        loss = train(model, optimizer, criterion, train_loader, 1000)
        error = test(model, sim_fx, test_loader)

        print(f"Epoch: {i:2d}, Loss: {loss:.4f}, Error: {error:.4f}")




if __name__  == "__main__":
    main()
