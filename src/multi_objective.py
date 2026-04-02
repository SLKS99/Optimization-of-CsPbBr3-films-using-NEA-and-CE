"""
Multi-Objective Optimization using Pareto Front
==============================================
Implements Pareto-optimal selection for multi-objective optimization.
Useful when optimizing multiple conflicting objectives (e.g., PL intensity vs FWHM).
"""

import numpy as np
from typing import List, Tuple
from src.models import ExperimentParams


def calculate_pareto_front(candidates: List[ExperimentParams], 
                          objectives: List[str] = None) -> List[ExperimentParams]:
    """
    Find Pareto-optimal candidates (non-dominated solutions).
    
    A solution is Pareto-optimal if no other solution is better in ALL objectives.
    This is useful when you have conflicting objectives (e.g., maximize PL but minimize FWHM).
    
    Args:
        candidates: List of scored candidates
        objectives: List of objective names to optimize. Options:
            - 'quality': Maximize predicted quality
            - 'uncertainty': Maximize uncertainty (for exploration)
            - 'pl_intensity': Maximize PL intensity (if available)
            - 'fwhm': Minimize FWHM (if available)
            Default: ['quality', 'uncertainty']
    
    Returns:
        List of Pareto-optimal candidates
    """
    if not candidates:
        return []
    
    if objectives is None:
        objectives = ['quality', 'uncertainty']
    
    # Extract objective values for each candidate
    objective_values = []
    valid_candidates = []
    
    for cand in candidates:
        values = []
        valid = True
        
        for obj in objectives:
            if obj == 'quality':
                val = cand.metrics.predicted_performance if cand.metrics.predicted_performance is not None else 0.0
                values.append(val)
            elif obj == 'uncertainty':
                val = cand.metrics.uncertainty_score if cand.metrics.uncertainty_score is not None else 0.0
                values.append(val)
            elif obj == 'pl_intensity':
                # Would need to extract from history if available
                # For now, use predicted quality as proxy
                val = cand.metrics.predicted_performance if cand.metrics.predicted_performance is not None else 0.0
                values.append(val)
            elif obj == 'fwhm':
                # Would need to extract from history if available
                # For now, use negative uncertainty as proxy (lower uncertainty = better)
                val = -cand.metrics.uncertainty_score if cand.metrics.uncertainty_score is not None else 0.0
                values.append(val)
            else:
                valid = False
                break
        
        if valid:
            objective_values.append(np.array(values))
            valid_candidates.append(cand)
    
    if not objective_values:
        return []
    
    objective_matrix = np.array(objective_values)
    
    # Find Pareto-optimal solutions
    # A solution is Pareto-optimal if no other solution dominates it
    # (i.e., no other solution is better in ALL objectives)
    pareto_indices = []
    
    for i in range(len(valid_candidates)):
        is_pareto = True
        for j in range(len(valid_candidates)):
            if i == j:
                continue
            
            # Check if j dominates i (better in all objectives)
            # For maximization: j dominates i if j >= i in all objectives and j > i in at least one
            if np.all(objective_matrix[j] >= objective_matrix[i]) and np.any(objective_matrix[j] > objective_matrix[i]):
                is_pareto = False
                break
        
        if is_pareto:
            pareto_indices.append(i)
    
    return [valid_candidates[i] for i in pareto_indices]


def select_diverse_pareto_front(candidates: List[ExperimentParams],
                               n_select: int = 10,
                               objectives: List[str] = None) -> List[ExperimentParams]:
    """
    Select diverse candidates from Pareto front.
    
    First finds Pareto-optimal candidates, then selects diverse subset
    to cover different trade-offs in objective space.
    If there are fewer Pareto candidates than requested, supplements with
    top-ranked non-Pareto candidates.
    
    Args:
        candidates: List of scored candidates
        n_select: Number of candidates to select
        objectives: List of objective names (see calculate_pareto_front)
    
    Returns:
        List of diverse candidates (Pareto-optimal + top-ranked supplements)
    """
    pareto_candidates = calculate_pareto_front(candidates, objectives)
    
    # If we have enough Pareto candidates, select diverse subset
    if len(pareto_candidates) >= n_select:
        # Continue with diverse selection from Pareto front
        pass
    else:
        # Not enough Pareto candidates - supplement with top-ranked candidates
        selected = pareto_candidates.copy()
        # Use id() for comparison since ExperimentParams objects are not hashable
        pareto_ids = {id(c) for c in pareto_candidates}
        
        # Get top-ranked candidates that are NOT in Pareto front
        remaining_candidates = [c for c in candidates if id(c) not in pareto_ids]
        remaining_candidates.sort(key=lambda x: x.rank_score, reverse=True)
        
        # Add top-ranked candidates until we reach n_select
        needed = n_select - len(selected)
        for cand in remaining_candidates[:needed]:
            selected.append(cand)
        
        return selected
    
    # Select diverse subset using k-means-like clustering in objective space
    # For simplicity, use greedy selection to maximize diversity
    selected = []
    remaining = pareto_candidates.copy()
    
    # Start with the candidate that has highest combined score
    if pareto_candidates:
        first = max(pareto_candidates, 
                   key=lambda x: (x.metrics.predicted_performance or 0) + (x.metrics.uncertainty_score or 0))
        selected.append(first)
        remaining.remove(first)
    
    # Greedily add candidates that are most different from already selected
    while len(selected) < n_select and remaining:
        max_distance = -1
        best_candidate = None
        
        for cand in remaining:
            # Calculate minimum distance to any selected candidate
            min_dist = float('inf')
            for sel in selected:
                # Distance in objective space
                dist = 0.0
                if cand.metrics.predicted_performance is not None and sel.metrics.predicted_performance is not None:
                    dist += abs(cand.metrics.predicted_performance - sel.metrics.predicted_performance)
                if cand.metrics.uncertainty_score is not None and sel.metrics.uncertainty_score is not None:
                    dist += abs(cand.metrics.uncertainty_score - sel.metrics.uncertainty_score)
                
                min_dist = min(min_dist, dist)
            
            if min_dist > max_distance:
                max_distance = min_dist
                best_candidate = cand
        
        if best_candidate:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            break
    
    return selected
