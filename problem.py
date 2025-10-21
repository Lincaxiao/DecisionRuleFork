import torch


class PrimalLP:
    def __init__(self, c, nonnegative_mask, DC3LHS=False, A_eq=None, b_eq=None, h_ineq=None):
        super().__init__()
        self.c = c
        self.nonnegative_mask = nonnegative_mask.bool()
        self.DC3LHS = DC3LHS
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.h_ineq = h_ineq

    @staticmethod
    def optimality_gap(obj, true_obj):
        return (obj - true_obj) / true_obj

    def obj_fn(self, x):
        if self.DC3LHS:
            return x @ self.c[~self.nonnegative_mask]
        return x @ self.c

    def ineq_residual(self, x, G_ineq):
        if self.DC3LHS:
            return (G_ineq @ x.unsqueeze(-1)).squeeze(-1) - self.h_ineq
        return torch.relu(-x[:, self.nonnegative_mask])

    def eq_residual(self, x, A, b):
        if self.DC3LHS:
            return x @ self.A_eq.T - self.b_eq
        return (A @ x.flatten() - b.flatten()).view(-1, b.shape[-1])



