"""
2D Ising model Hamiltonian.

The Ising model is a nearest-neighbor spin model on a 2D square lattice
with uniform coupling J_{ij} = 1 (no random disorder).
"""

import torch


class IsingHamiltonian:
    """
    2D Ising Hamiltonian.

    The Ising model is defined by the Hamiltonian:
        H = -sum_{<i,j>} s_i * s_j

    where the sum is over nearest-neighbor pairs on a square lattice
    All couplings J_{ij} = 1, so no coupling matrix is needed.

    Args:
        L: Linear dimension of the square lattice.
        device: Device to place tensors on.
        boundary: Boundary condition ('OBC' or 'PBC').
    """

    def __init__(
        self,
        L: int,
        device: torch.device,
        boundary: str = "OBC",
    ):
        self.L = L
        self.n = L * L
        self.boundary = boundary
        self.device = device
        
    
    
    def energy(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Compute energy of spin configurations.

        Directly sums over nearest-neighbor pairs on the 2D grid without
        constructing an N x N coupling matrix.

        Args:
            samples: Spin configurations of shape [B, n] with values in {-1, +1}.
            n must equal L * L.

        Returns:
            Energy values of shape [B].
        """
        samples = samples.view(samples.shape[0], -1)
        assert samples.shape[1] == self.n

        cfgs = samples.view(-1, self.L, self.L)
        # Horizontal interactions (right neighbor)
        energy = -torch.sum(cfgs[:, :, : self.L - 1] * cfgs[:, :, 1:], dim=(1, 2))
        # Vertical interactions (down neighbor)
        energy += -torch.sum(cfgs[:, : self.L - 1, :] * cfgs[:, 1:, :], dim=(1, 2))

        if self.boundary == "PBC":
            # Periodic wrap in y (horizontal)
            energy += -torch.sum(cfgs[:, :, 0] * cfgs[:, :, self.L - 1], dim=1)
            # Periodic wrap in x (vertical)
            energy += -torch.sum(cfgs[:, 0, :] * cfgs[:, self.L - 1, :], dim=1)

        return energy
    
    def __repr__(self) -> str:
        return (
            f"IsingHamiltonian(L={self.L}, n={self.n}, "
            f"boundary={self.boundary})"
        )
