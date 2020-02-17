import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np


class BatchNormNode(nn.Module):
    """Batch normalization for node features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormNode, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, x):
        """
        Args:
            x: Node features (batch_size, hidden_dim, num_nodes)

        Returns:
            x_bn: Node features after batch normalization (batch_size, hidden_dim, num_nodes)
        """
        x_bn = self.batch_norm(x) # B x H x N
        return x_bn


class BatchNormEdge(nn.Module):
    """Batch normalization for edge features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormEdge, self).__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)

    def forward(self, e):
        """
        Args:
            e: Edge features (batch_size, hidden_dim, num_nodes, num_nodes)

        Returns:
            e_bn: Edge features after batch normalization (batch_size, hidden_dim, num_nodes, num_nodes)
        """
        e_bn = self.batch_norm(e) # B x H x N x N
        return e_bn


class NodeFeatures(nn.Module):
    """Convnet features for nodes.
    
    Using `sum` aggregation:
        x_i = U*x_i +  sum_j [ gate_ij * (V*x_j) ]
    
    Using `mean` aggregation:
        x_i = U*x_i + ( sum_j [ gate_ij * (V*x_j) ] / sum_j [ gate_ij] )
    """
    
    def __init__(self, hidden_dim, aggregation="mean"):
        super(NodeFeatures, self).__init__()
        self.aggregation = aggregation
        self.U = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.V = nn.Conv1d(hidden_dim, hidden_dim, 1)

    def forward(self, x, edge_gate):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            edge_gate: Edge gate values (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
        """
        Ux = self.U(x)  # B x H x V
        Vx = self.V(x)  # B x H x V
        Vx = Vx.unsqueeze(2)  # extend Vx from "B x H x V" to "B x H x 1 x V"
        gateVx = edge_gate * Vx  # B x H x V x V
        if self.aggregation=="mean":
            x_new = Ux + torch.sum(gateVx, dim=3) / (1e-20 + torch.sum(edge_gate, dim=3))  # B x H x V
        elif self.aggregation=="sum":
            x_new = Ux + torch.sum(gateVx, dim=3)  # B x H x V
        return x_new


class EdgeFeatures(nn.Module):
    """Convnet features for edges.

    e_ij = U*e_ij + V*(x_i + x_j)
    """

    def __init__(self, hidden_dim):
        super(EdgeFeatures, self).__init__()
        self.U = nn.Conv1d(hidden_dim, hidden_dim, (1,1))
        self.V = nn.Conv1d(hidden_dim, hidden_dim, 1)
        
    def forward(self, x, e):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        Ue = self.U(e) # B x H x V x V
        Vx = self.V(x) # B x H x V
        Wx = Vx.unsqueeze(2)  # Extend Vx from "B x H x V" to "B x H x 1 x V"
        Vx = Vx.unsqueeze(3)  # extend Vx from "B x H x V" to "B x H x V x 1"
        e_new = Ue + Vx + Wx
        return e_new


class ResidualGatedGCNLayer(nn.Module):
    """Convnet layer with gating and residual connection.
    """

    def __init__(self, hidden_dim, aggregation="sum"):
        super(ResidualGatedGCNLayer, self).__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormEdge(hidden_dim)

    def forward(self, x, e):
        """
        Args:
            x: Node features (batch_size, hidden_dim, num_nodes)
            e: Edge features (batch_size, hidden_dim, num_nodes, num_nodes)

        Returns:
            x_new: Convolved node features (batch_size, hidden_dim, num_nodes)
            e_new: Convolved edge features (batch_size, hidden_dim, num_nodes, num_nodes)
        """
        e_in = e # B x H x V x V
        x_in = x # B x H x V
        # Edge convolution
        e_tmp = self.edge_feat(x_in, e_in)  # B x H x V x V
        # Compute edge gates
        edge_gate = F.sigmoid(e_tmp)
        # Node convolution
        x_tmp = self.node_feat(x_in, edge_gate) # B x H x V
        # Batch normalization
        e_tmp = self.bn_edge(e_tmp)
        x_tmp = self.bn_node(x_tmp)
        # ReLU Activation
        e = F.relu(e_tmp)
        x = F.relu(x_tmp)
        # Residual connection
        x_new = x_in + x
        e_new = e_in + e
        return x_new, e_new


class MLP(nn.Module):
    """Multi-layer Perceptron for output prediction.
    """

    def __init__(self, hidden_dim, output_dim, L=2):
        super(MLP, self).__init__()
        self.L = L
        U = []
        for layer in range(self.L - 1):
            U.append(nn.Conv1d(hidden_dim, hidden_dim, (1, 1))) # B x H x V x V
        self.U = nn.ModuleList(U)
        self.V = nn.Conv1d(hidden_dim, output_dim, (1, 1)) # B x O x V x V

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, hidden_dim, num_nodes, num_nodes)

        Returns:
            y: Output predictions (batch_size, output_dim, num_nodes, num_nodes)
        """
        Ux = x
        for U_i in self.U:
            Ux = U_i(Ux)  # B x H x V x V
            Ux = F.relu(Ux)  # B x H x V x V
        y = self.V(Ux)  # B x O x V x V
        y = y.permute(0, 2, 3, 1)
        return y
