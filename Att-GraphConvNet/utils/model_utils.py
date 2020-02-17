import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import math

from utils.beamsearch import *
from utils.graph_utils import *


def loss_nodes(y_pred_nodes, y_nodes, node_cw):
    """
    Loss function for node predictions.

    Args:
        y_pred_nodes: Predictions for nodes (batch_size, num_nodes)
        y_nodes: Targets for nodes (batch_size, num_nodes)
        node_cw: Class weights for nodes loss

    Returns:
        loss_nodes: Value of loss function
    
    """
    # Node loss
    y = F.log_softmax(y_pred_nodes, dim=2)  # B x V x voc_nodes_out
    y = y.permute(0, 2, 1)  # B x voc_nodes x V
    loss_nodes = nn.NLLLoss(node_cw)(y, y_nodes) # 交叉熵损失
    return loss_nodes


def loss_edges_hnm(y_pred_edges, y_edges, edge_cw, num_neg = 4):
    """
    Loss function for edge predictions using hard negative mining.

    Args:
        y_pred_edges : Predictions for edges (batch_size, num_nodes, num_nodes, voc_edges)
        y_edges : Targets for edges (batch_size, num_nodes, num_nodes)
        edge_cw : Class weights for edges loss
        num_neg : the ratio between the negative examples and positive examples
        permute_shape : reshape for mask (num_classes, batch_size, num_nodes, num_neg + 2)
    Returns:
        loss_edges: Value of loss function
    
    """
    # Edge loss
    y = F.log_softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
    y = y.permute(0, 3, 1, 2)  # B x voc_edges x V x V
    #print('pre-shape is {}'.format(y.shape))
    #print('num_neg is {}'.format(num_neg))
    # qiukaibin --- 2019-12-23
    mask = hard_negative_mining(y_pre_edges=y[:,1,:,:].clone(), labels=y_edges, num_neg=num_neg)
    
    y_mask, y_edges_mask = loss_edges_max_negative(y_pred_edges=y, y_edges=y_edges, mask = mask, num_neg=num_neg)
    #loss_edges = nn.NLLLoss(edge_cw)(y_mask, y_edges_mask)
    
    
    
    return loss_edges

def hard_negative_mining(y_pre_edges, labels, num_neg = 4):
    """
    Args:
        y_pre_edges : the prediction for each edge (batch_size, num_nodes, num_nodes)  
        labels : Targets for edges (batch_size, num_nodes, num_nodes)
        num_neg : the ratio between the negative examples and positive examples
    
    Returns:
        mask : the mask for loss (batch_size, num_nodes, num_nodes)
    """
    pos_mask = labels > 0 # B x N x N
    
    y_pre_edges[pos_mask] = -math.inf # 
    #print('pos_mask shape is {}'.format(y_pre_edges[pos_mask].shape))
    _, indexes = y_pre_edges.sort(dim = 2, descending = True)
    _, orders = indexes.sort(dim = 2)
    neg_mask = orders < num_neg
    mask = pos_mask | neg_mask # B x N x N
    mask = Variable(mask)
    
    return mask

def loss_edges_max_negative(y_pred_edges, y_edges, mask, num_neg = 4):
    """
    Args:
        y_pred_edges : Prediction for edges (batch_size, num_classes, num_nodes, num_nodes)
        y_edges : Target for edges (batch_size, num_nodes, num_nodes)
        mask : the mask for loss (batch_size, num_nodes, num_nodes)
        permute_shape : reshape for mask (num_classes, batch_size, num_nodes, num_neg + 2)
    
    """
    batch_size, num_classes, num_nodes, _ = y_pred_edges.shape # B, voc, N
    #print(batch_size, num_classes, num_nodes)
    y_pred_edges = y_pred_edges.permute(1, 0, 2, 3) # voc x B x N x N
    y_pred_edges_mask = y_pred_edges[:, mask].view(num_classes, batch_size, num_nodes, num_neg + 2) # voc x B x N x num_neg + 2
    y_pred_edges_mask = y_pred_edges_mask.permute(1, 0, 2, 3) # B x voc x N x N
    
    y_edges_mask = y_edges[mask].view(batch_size, num_nodes, num_neg + 2) # B x N x num_neg + 2
    
    return y_pred_edges_mask, y_edges_mask

def loss_edges(y_pred_edges, y_edges, edge_cw, loss_type = 'CE',
               reduction = 'mean', gamma = 2):
    """
    Loss function for edge predictions.

    Args:
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
        y_edges: Targets for edges (batch_size, num_nodes, num_nodes)
        edge_cw: Class weights for edges loss

    Returns:
        loss_edges: Value of loss function
    
    """
    # Edge loss
    if loss_type == 'CE':
        y = F.log_softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
        y = y.permute(0, 3, 1, 2)  # B x voc_edges x V x V
        loss_edges = nn.NLLLoss(edge_cw)(y, y_edges)
    elif loss_type == 'FL':
        #print(gamma)
        y = y_pred_edges.permute(0, 3, 1, 2)  # B x voc_edges x V x V
        loss_edges = FocalLoss(weight = edge_cw, gamma = gamma, reduction = reduction)(y, y_edges)
    return loss_edges

class FocalLoss(nn.Module):
    """
    Focal Loss for edge predictions.
    
    
    """
    def __init__(self, weight=None, 
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight=self.weight,
            reduction = self.reduction
        )

def beamsearch_tour_nodes(y_pred_edges, beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type='raw', random_start=False):
    """
    Performs beamsearch procedure on edge prediction matrices and returns possible TSP tours.

    Args:
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
        beam_size: Beam size
        batch_size: Batch size
        num_nodes: Number of nodes in TSP tours
        dtypeFloat: Float data type (for GPU/CPU compatibility)
        dtypeLong: Long data type (for GPU/CPU compatibility)
        random_start: Flag for using fixed (at node 0) vs. random starting points for beamsearch

    Returns: TSP tours in terms of node ordering (batch_size, num_nodes)

    """
    if probs_type == 'raw':
        # Compute softmax over edge prediction matrix
        y = F.softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
        # Consider the second dimension only
        y = y[:, :, :, 1]  # B x V x V
    elif probs_type == 'logits':
        # Compute logits over edge prediction matrix
        y = F.log_softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
        # Consider the second dimension only
        y = y[:, :, :, 1]  # B x V x V
        y[y == 0] = -1e-20  # Set 0s (i.e. log(1)s) to very small negative number
    # Perform beamsearch
    beamsearch = Beamsearch(beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type, random_start)
    trans_probs = y.gather(1, beamsearch.get_current_state())
    for step in range(num_nodes - 1):
        beamsearch.advance(trans_probs)
        trans_probs = y.gather(1, beamsearch.get_current_state())
    # Find TSP tour with highest probability among beam_size candidates
    ends = torch.zeros(batch_size, 1).type(dtypeLong)
    return beamsearch.get_hypothesis(ends)


def beamsearch_tour_nodes_shortest(y_pred_edges, x_edges_values, beam_size, batch_size, num_nodes,
                                   dtypeFloat, dtypeLong, probs_type='raw', random_start=False):
    """
    Performs beamsearch procedure on edge prediction matrices and returns possible TSP tours.

    Final predicted tour is the one with the shortest tour length.
    (Standard beamsearch returns the one with the highest probability and does not take length into account.)

    Args:
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
        x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
        beam_size: Beam size
        batch_size: Batch size
        num_nodes: Number of nodes in TSP tours
        dtypeFloat: Float data type (for GPU/CPU compatibility)
        dtypeLong: Long data type (for GPU/CPU compatibility)
        probs_type: Type of probability values being handled by beamsearch (either 'raw'/'logits'/'argmax'(TODO))
        random_start: Flag for using fixed (at node 0) vs. random starting points for beamsearch

    Returns:
        shortest_tours: TSP tours in terms of node ordering (batch_size, num_nodes)

    """
    if probs_type == 'raw':
        # Compute softmax over edge prediction matrix
        y = F.softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
        # Consider the second dimension only
        y = y[:, :, :, 1]  # B x V x V
    elif probs_type == 'logits':
        # Compute logits over edge prediction matrix
        y = F.log_softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
        # Consider the second dimension only
        y = y[:, :, :, 1]  # B x V x V
        y[y == 0] = -1e-20  # Set 0s (i.e. log(1)s) to very small negative number
    # Perform beamsearch
    beamsearch = Beamsearch(beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type, random_start)
    trans_probs = y.gather(1, beamsearch.get_current_state())
    for step in range(num_nodes - 1):
        beamsearch.advance(trans_probs)
        trans_probs = y.gather(1, beamsearch.get_current_state())
    # Initially assign shortest_tours as most probable tours i.e. standard beamsearch
    ends = torch.zeros(batch_size, 1).type(dtypeLong)
    shortest_tours = beamsearch.get_hypothesis(ends)
    # Compute current tour lengths
    shortest_lens = [1e6] * len(shortest_tours)
    for idx in range(len(shortest_tours)):
        shortest_lens[idx] = tour_nodes_to_tour_len(shortest_tours[idx].cpu().numpy(),
                                                    x_edges_values[idx].cpu().numpy())
    # Iterate over all positions in beam (except position 0 --> highest probability)
    for pos in range(1, beam_size):
        ends = pos * torch.ones(batch_size, 1).type(dtypeLong)  # New positions
        hyp_tours = beamsearch.get_hypothesis(ends)
        for idx in range(len(hyp_tours)):
            hyp_nodes = hyp_tours[idx].cpu().numpy()
            hyp_len = tour_nodes_to_tour_len(hyp_nodes, x_edges_values[idx].cpu().numpy())
            # Replace tour in shortest_tours if new length is shorter than current best
            if hyp_len < shortest_lens[idx] and is_valid_tour(hyp_nodes, num_nodes):
                shortest_tours[idx] = hyp_tours[idx]
                shortest_lens[idx] = hyp_len
    return shortest_tours


def update_learning_rate(optimizer, lr):
    """
    Updates learning rate for given optimizer.

    Args:
        optimizer: Optimizer object
        lr: New learning rate

    Returns:
        optimizer: Updated optimizer object
        s
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def edge_error(y_pred, y_target, x_edges):
    """
    Computes edge error metrics for given batch prediction and targets.

    Args:
        y_pred: Edge predictions (batch_size, num_nodes, num_nodes, voc_edges)
        y_target: Edge targets (batch_size, num_nodes, num_nodes)
        x_edges: Adjacency matrix (batch_size, num_nodes, num_nodes)

    Returns:
        err_edges, err_tour, err_tsp, edge_err_idx, err_idx_tour, err_idx_tsp
    
    """
    y = F.softmax(y_pred, dim=3)  # B x V x V x voc_edges
    y = y.argmax(dim=3)  # B x V x V

    # Edge error: Mask out edges which are not connected
    mask_no_edges = x_edges.long()
    err_edges, _ = _edge_error(y, y_target, mask_no_edges)

    # TSP tour edges error: Mask out edges which are not on true TSP tours
    mask_no_tour = y_target
    err_tour, err_idx_tour = _edge_error(y, y_target, mask_no_tour)

    # TSP tour edges + positively predicted edges error:
    # Mask out edges which are not on true TSP tours or are not predicted positively by model
    mask_no_tsp = ((y_target + y) > 0).long()
    err_tsp, err_idx_tsp = _edge_error(y, y_target, mask_no_tsp)

    return 100 * err_edges, 100 * err_tour, 100 * err_tsp, err_idx_tour, err_idx_tsp


def _edge_error(y, y_target, mask):
    """
    Helper method to compute edge errors.

    Args:
        y: Edge predictions (batch_size, num_nodes, num_nodes)
        y_target: Edge targets (batch_size, num_nodes, num_nodes)
        mask: Edges which are not counted in error computation (batch_size, num_nodes, num_nodes)

    Returns:
        err: Mean error over batch
        err_idx: One-hot array of shape (batch_size)- 1s correspond to indices which are not perfectly predicted
    
    """
    # Compute equalities between pred and target
    acc = (y == y_target).long()
    # Multipy by mask => set equality to 0 on disconnected edges
    acc = (acc * mask)
    #  Get accuracy of each y in the batch (sum of 1s in acc_edges divided by sum of 1s in edges mask)
    acc = acc.sum(dim=1).sum(dim=1).to(dtype=torch.float) / mask.sum(dim=1).sum(dim=1).to(dtype=torch.float)
    # Compute indices which are not perfect
    err_idx = (acc < 1.0)
    # Take mean over batch
    acc = acc.sum().to(dtype=torch.float).item() / acc.numel()
    # Compute error
    err = 1.0 - acc
    return err, err_idx
