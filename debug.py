# import torch

# # refined
# a = [
#     [[-9.6051e+01, -5.7985e+00,  1.6425e+00,  2.5098e+00,  5.4158e+00,
#        2.0036e+00,  1.0783e+01,  1.8010e+00,  8.5754e-02],
#      [-4.5106e+01, -6.9942e-01,  1.3083e+00,  1.3477e+00,  4.8121e+00,
#        1.6665e+00,  5.1986e+00,  3.5421e-01,  1.0773e+00],
#      [-1.9341e+01, -2.3320e+00,  1.1595e+00,  1.9861e+00,  3.9933e+00,
#        1.1141e+00, -2.7313e+01, -9.0219e-01,  3.8476e-01],
#      [-8.6813e-01, -1.4406e+00,  5.8911e-01,  1.1994e+00,  1.7449e+00,
#        1.4683e+00, -2.6428e-01, -1.7289e+00, -9.1123e-02],
#      [-5.6549e+00,  6.2761e-01,  1.0495e+00,  1.7673e+00,  4.8811e+00,
#        1.0709e+00, -2.3016e+01, -1.1544e+00, -1.9495e-01],
#      [-3.8156e+00,  1.5862e+00,  1.6591e+00,  9.7604e-01,  3.4416e+00,
#        1.3147e+00, -1.3508e+01,  1.3782e+00,  9.7196e-01]]
# ]

# b= [
#     [[-9.5560e+01, -5.6850e+00,  1.6150e+00,  2.5140e+00,  5.4150e+00,
#        2.0000e+00, -1.1662e+00,  1.1861e+00,  8.1340e-02],
#      [-1.0270e+02, -5.4854e+00,  1.6486e+00,  2.5140e+00,  5.4150e+00,
#        2.0000e+00,  5.8263e+00,  1.6428e+00,  1.2381e-01],
#      [-9.0022e+01, -4.1252e+00,  1.4815e+00,  2.5140e+00,  5.4150e+00,
#        2.0000e+00,  1.8317e+01,  2.6437e+00,  1.4759e-01],
#      [-8.5295e+01, -2.9753e+00,  1.4266e+00,  2.5140e+00,  5.4150e+00,
#        2.0000e+00,  9.4586e+00,  2.2991e+00,  1.5115e-01],
#      [-8.0566e+01, -1.8263e+00,  1.3728e+00,  2.5140e+00,  5.4150e+00,
#        2.0000e+00,  9.4314e+00,  2.1272e+00,  1.5471e-01],
#      [-7.5866e+01, -8.4827e-01,  1.3231e+00,  2.5140e+00,  5.4150e+00,
#        2.0000e+00,  4.4996e+00,  1.4373e+00,  1.5826e-01]]
# ]



# # none
# # a = [[[ 4.9000,  4.2726,  1.6300,  3.2400,  6.1736,  2.7674, -3.2219,
# #            2.6101,  1.0434],
# #          [ 0.6225,  1.3182,  0.1690,  1.7964,  4.0805,  1.9954,  1.6375,
# #            5.3024,  3.6140],
# #          [ 7.9745,  5.8231,  2.0060,  3.5412,  6.3394,  3.2900, -4.0935,
# #            3.3110,  1.3115],
# #          [ 2.1556,  3.5352,  1.3493,  2.4970,  4.2852,  2.3735, -2.8198,
# #            2.3543,  1.0816],
# #          [ 1.9180,  3.4806,  1.3304,  2.4493,  4.2033,  2.3411, -2.7743,
# #            2.3385,  1.0370],
# #          [ 1.0930,  2.7360,  1.2033,  2.1014,  3.5086,  2.0573, -2.1884,
# #            1.8353,  0.8754]]]

# # b = [[[-5.5438e+01,  1.7333e+00,  1.2322e+00,  1.8750e+00,  4.6300e+00,
# #            1.6240e+00,  2.2367e+01, -1.4847e+00, -4.5982e-02],
# #          [-4.7360e+01,  1.2082e+00,  1.2743e+00,  1.8750e+00,  4.6300e+00,
# #            1.6240e+00,  1.4216e+01, -1.2836e+00, -4.5982e-02],
# #          [-4.1933e+01,  5.1492e-01,  1.2774e+00,  1.8750e+00,  4.6300e+00,
# #            1.6240e+00,  1.0579e+01, -9.2823e-01, -1.1080e-02],
# #          [-3.6790e+01,  2.8099e-01,  1.2744e+00,  1.8750e+00,  4.6300e+00,
# #            1.6240e+00,  9.9211e+00, -2.4131e-01, -1.1080e-02],
# #          [-3.2020e+01,  2.7404e-01,  1.2699e+00,  1.8750e+00,  4.6300e+00,
# #            1.6240e+00,  1.2091e+01, -1.2554e-01,  6.3716e-03],
# #          [-2.4695e+01,  1.5573e-01,  1.2589e+00,  1.8750e+00,  4.6300e+00,
# #            1.6240e+00,  1.0469e+01, -2.5387e-01, -1.1080e-02]]]



# pred = torch.tensor(a, device='cuda')

# target = torch.tensor(b, device='cuda')

# ade = torch.linalg.norm((pred[:,:,:3]- target[:,:,:3]), dim=-1)
# print(ade)




import torch.linalg
import numpy as np

def inv_unscented_transform(sigma_new, weight_c, weight_m, residual_fn=None):
    print(weight_m.dtype)
    state_new = sigma_new @ weight_m[..., None]
    if residual_fn is not None:
        sigma_deltas = residual_fn(sigma_new, state_new)
    else:
        sigma_deltas = sigma_new - state_new
    P_new = sigma_deltas @ (torch.diagflat(weight_c) @ sigma_deltas.mT)
    # P_new = torch.sum(outer_products * weight_c[..., None, None], dim=-3)
    return state_new, P_new


def unscented_transform(state, P, alpha, kappa, beta):
    l = state.shape[-2]
    lmbd = alpha**2 * (l + kappa) - l
    matrix_sqrt, info = torch.linalg.cholesky_ex(
        (l + lmbd) * P,
    )  # TODO: cholesky is super unstable in training. replace!!!
    matrix_sqrt = matrix_sqrt.to(state.dtype).to(state.device)
    print(matrix_sqrt.dtype)
    # print(matrix_sqrt)
    sigma_i = state + matrix_sqrt.mT
    sigma_iL = state - matrix_sqrt.mT
    sigma = torch.cat((state, sigma_iL, sigma_i), dim=-1)
    weight_m_0 = lmbd / (l + lmbd)
    weight_c_0 = lmbd / (l + lmbd) + (1 - alpha**2 + beta)
    weight_m_i = 1 / (2 * (l + lmbd))
    weight_c_i = 1 / (2 * (l + lmbd))
    weight_m = torch.tensor(
        np.asarray((weight_m_0,) + (weight_m_i,) * 2 * l),
        dtype=state.dtype,
        device=state.device,
    )
    weight_c = torch.tensor(
        np.asarray((weight_c_0,) + (weight_c_i,) * 2 * l),
        dtype=state.dtype,
        device=state.device,
    )
    return sigma, weight_c, weight_m


a = [
    [[-9.6051e+01, -5.7985e+00,  1.6425e+00,  2.5098e+00,  5.4158e+00,
       2.0036e+00,  1.0783e+01,  1.8010e+00,  8.5754e-02],
     [-4.5106e+01, -6.9942e-01,  1.3083e+00,  1.3477e+00,  4.8121e+00,
       1.6665e+00,  5.1986e+00,  3.5421e-01,  1.0773e+00],
     [-1.9341e+01, -2.3320e+00,  1.1595e+00,  1.9861e+00,  3.9933e+00,
       1.1141e+00, -2.7313e+01, -9.0219e-01,  3.8476e-01],
     [-8.6813e-01, -1.4406e+00,  5.8911e-01,  1.1994e+00,  1.7449e+00,
       1.4683e+00, -2.6428e-01, -1.7289e+00, -9.1123e-02],
     [-5.6549e+00,  6.2761e-01,  1.0495e+00,  1.7673e+00,  4.8811e+00,
       1.0709e+00, -2.3016e+01, -1.1544e+00, -1.9495e-01],
     [-3.8156e+00,  1.5862e+00,  1.6591e+00,  9.7604e-01,  3.4416e+00,
       1.3147e+00, -1.3508e+01,  1.3782e+00,  9.7196e-01]]
]
a = torch.tensor(a, device='cuda').reshape(6,9,1)

P = torch.tensor(np.mat(np.eye(9)) * 0.01)

sigma, wc, wm = unscented_transform(a, P, alpha=0.1, kappa=0, beta=2)

print(sigma.shape)
print(wm.shape)
nstate, ncov = inv_unscented_transform(sigma, wc, wm)
# ncov = np.mat(ncov[-1,:].cpu().numpy())
# print(nstate.squeeze(-1))
print(nstate.shape)
print(ncov.shape)
