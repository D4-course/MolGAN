from activation_fn import soft_gumbel, hard_gumbel
import numpy as np
import torch
torch.manual_seed(0)

def test_soft_gumbel():
    torch.manual_seed(0)

    test_input = torch.tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
    out = soft_gumbel(test_input)

    expected = np.array([[0.089941, 0.28343024, 0.62662876],
                         [0.17598645, 0.47606989, 0.34794365]])

    val = out.cpu().detach().numpy()
    val = np.round_(val, decimals=4)
    expected = np.round_(expected, decimals=4)


    np.testing.assert_array_equal(val, expected)

def test_hard_gumbel():
    torch.manual_seed(0)

    test_input = torch.tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
    out = hard_gumbel(test_input)

    expected = np.array([[0., 0., 1.],
                         [0., 1., 0.]])

    val = out.cpu().detach().numpy()
    val = np.round_(val, decimals=4)
    expected = np.round_(expected, decimals=4)


    np.testing.assert_array_equal(val, expected)
