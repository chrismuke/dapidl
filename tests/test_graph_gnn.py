# tests/test_graph_gnn.py
import torch
from dapidl.graph.gnn import NucleusNodeCNN, scatter_mean, SageCellTyper


def test_scatter_mean_matches_hand_average():
    src = torch.tensor([0, 0, 1])         # node 0 has 2 neighbours, node 1 has 1
    dst = torch.tensor([1, 2, 2])
    x = torch.tensor([[1.0], [3.0], [5.0]])
    out = scatter_mean(x[dst], src, num_nodes=3)
    assert torch.allclose(out[0], torch.tensor([4.0]))   # mean(x[1], x[2]) = mean(3,5)
    assert torch.allclose(out[1], torch.tensor([5.0]))   # mean(x[2]) = 5
    assert torch.allclose(out[2], torch.tensor([0.0]))   # node 2 has no out-edges


def test_node_cnn_output_shape():
    cnn = NucleusNodeCNN(out_dim=128)
    y = cnn(torch.randn(4, 1, 40, 40))
    assert y.shape == (4, 128)


def test_sage_celltyper_forward_shape():
    model = SageCellTyper(node_dim=128, hidden=64, num_classes=4, layers=2)
    crops = torch.randn(6, 1, 40, 40)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    logits = model(crops, edge_index)
    assert logits.shape == (6, 4)
