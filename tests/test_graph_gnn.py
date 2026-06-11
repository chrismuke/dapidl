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


def test_nograph_aggregator_returns_zeros():
    import torch
    from dapidl.graph.gnn import NoGraphAggregator
    agg = NoGraphAggregator()
    se = torch.randn(3, 5)
    ne = torch.randn(3, 4, 5)
    valid = torch.ones(3, 4)
    out = agg(se, ne, valid)
    assert out.shape == se.shape
    assert torch.all(out == 0)
    assert agg.needs_neighbours is False


def test_mean_aggregator_masked_mean():
    import torch
    from dapidl.graph.gnn import MeanAggregator
    agg = MeanAggregator()
    se = torch.zeros(1, 2)
    ne = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [9.0, 9.0]]])   # 3 neighbours
    valid = torch.tensor([[1.0, 1.0, 0.0]])                      # 3rd is padding -> ignored
    out = agg(se, ne, valid)
    assert torch.allclose(out, torch.tensor([[2.0, 2.0]]))       # mean of first two
    assert agg.needs_neighbours is True


def test_edge_gatv2_forward_shape_and_attention():
    import torch
    from dapidl.graph.gnn import EdgeGATv2Aggregator
    agg = EdgeGATv2Aggregator(node_dim=16, edge_dim=8, heads=4)
    se = torch.randn(5, 16)
    ne = torch.randn(5, 3, 16)
    valid = torch.tensor([[1., 1., 1.]] * 4 + [[0., 0., 0.]])     # last node: no valid neighbours
    edge_attr = torch.randn(5, 3, 8)
    out = agg(se, ne, valid, edge_attr)
    assert out.shape == (5, 16)                                   # heads*head_dim == node_dim
    assert torch.isfinite(out).all()                             # all-invalid row -> finite (zero)
    assert torch.allclose(out[4], torch.zeros(16))               # no valid neighbours -> zero agg
    assert agg.needs_neighbours and agg.needs_edge_attr


def test_edge_gatv2_edge_attr_changes_output():
    import torch
    torch.manual_seed(0)
    from dapidl.graph.gnn import EdgeGATv2Aggregator
    agg = EdgeGATv2Aggregator(node_dim=16, edge_dim=8, heads=2)
    se = torch.randn(2, 16); ne = torch.randn(2, 3, 16); valid = torch.ones(2, 3)
    o1 = agg(se, ne, valid, torch.zeros(2, 3, 8))
    o2 = agg(se, ne, valid, torch.ones(2, 3, 8))
    assert not torch.allclose(o1, o2)                            # edge features influence attention
