import numpy as np
from dapidl.graph.edge_geometry import build_edge_attr


def _toy():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 50, (6, 2))
    node_geom = np.column_stack([rng.uniform(-np.pi, np.pi, 6),    # angle
                                 rng.uniform(0, 1, 6),             # ecc
                                 rng.uniform(0, 4, 6)])            # log_area
    nbr = np.array([[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]])
    return coords, node_geom, nbr


def test_shape_and_padding():
    coords, ng, nbr = _toy()
    nbr2 = nbr.copy(); nbr2[0, 1] = -1
    ea = build_edge_attr(coords, ng, nbr2)
    assert ea.shape == (6, 2, 8)
    assert np.all(ea[0, 1] == 0.0)                        # -1 slot -> zero row


def test_rotation_invariant():
    coords, ng, nbr = _toy()
    ea0 = build_edge_attr(coords, ng, nbr)
    phi = 0.9
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    coords_r = coords @ R.T
    ng_r = ng.copy(); ng_r[:, 0] = ng[:, 0] + phi          # rotate axes with the slide
    ea1 = build_edge_attr(coords_r, ng_r, nbr)             # same nbr (rotation preserves kNN)
    assert np.allclose(ea0, ea1, atol=1e-6)


def test_nan_axis_zeros_directional_terms():
    coords, ng, nbr = _toy()
    ng[0, 0] = np.nan                                      # node 0 has no orientation
    ea = build_edge_attr(coords, ng, nbr)
    # for any edge touching node 0, the 3 directional dims (indices 3,4,5) are zero
    assert np.all(ea[0, :, 3:6] == 0.0)                    # node 0 as source
    assert np.all(ea[1, 0, 3:6] == 0.0)                    # node 0 as neighbour of node 1
    assert np.isfinite(ea).all()
