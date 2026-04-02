"""
Convergence Criteria and Optimization History Tracking
======================================================
Tracks optimization progress and determines when to stop.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime


class ConvergenceTracker:
    """Tracks optimization progress and checks convergence criteria."""
    
    def __init__(self, history_file: str = 'data/optimization_history.csv'):
        self.history_file = history_file
        self.history = self._load_history()
    
    def _load_history(self) -> pd.DataFrame:
        """Load optimization history from CSV."""
        if os.path.exists(self.history_file):
            try:
                return pd.read_csv(self.history_file)
            except:
                return pd.DataFrame()
        return pd.DataFrame()
    
    def _save_history(self):
        """Save optimization history to CSV."""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        self.history.to_csv(self.history_file, index=False)
    
    def record_cycle(self, 
                    cycle: int,
                    best_quality: float,
                    avg_uncertainty_top10: float,
                    n_candidates: int,
                    n_unique_conditions: int = None) -> Dict:
        """
        Record optimization cycle results.
        
        Returns:
            Dictionary with convergence status and recommendations
        """
        record = {
            'cycle': cycle,
            'timestamp': datetime.now().isoformat(),
            'best_quality': best_quality,
            'avg_uncertainty_top10': avg_uncertainty_top10,
            'n_candidates': n_candidates,
            'n_unique_conditions': n_unique_conditions if n_unique_conditions else n_candidates
        }
        
        # Add to history
        if self.history.empty:
            self.history = pd.DataFrame([record])
        else:
            self.history = pd.concat([self.history, pd.DataFrame([record])], ignore_index=True)
        
        self._save_history()
        
        # Check convergence
        status = self.check_convergence()
        record.update(status)
        
        return record
    
    def check_convergence(self,
                         improvement_threshold: float = 0.05,
                         uncertainty_threshold: float = 10.0,
                         quality_threshold: Optional[float] = None,
                         min_cycles: int = 3,
                         plateau_cycles: int = 3) -> Dict:
        """
        Check if optimization has converged.
        
        Args:
            improvement_threshold: Minimum improvement (%) to not be considered plateau
            uncertainty_threshold: Maximum uncertainty for convergence
            quality_threshold: Target quality score (if None, not checked)
            min_cycles: Minimum cycles before checking convergence
            plateau_cycles: Number of cycles to check for plateau
        
        Returns:
            Dictionary with convergence status and recommendations
        """
        if len(self.history) < min_cycles:
            return {
                'converged': False,
                'reason': f'Need at least {min_cycles} cycles (currently {len(self.history)})',
                'recommendation': 'Continue optimization'
            }
        
        latest = self.history.iloc[-1]
        
        # Check quality threshold
        if quality_threshold and latest['best_quality'] >= quality_threshold:
            return {
                'converged': True,
                'reason': f'Quality threshold reached: {latest["best_quality"]:.1f} >= {quality_threshold}',
                'recommendation': 'Stop optimization - target achieved'
            }
        
        # Check uncertainty
        if latest['avg_uncertainty_top10'] < uncertainty_threshold:
            return {
                'converged': True,
                'reason': f'Low uncertainty: {latest["avg_uncertainty_top10"]:.1f} < {uncertainty_threshold}',
                'recommendation': 'Stop optimization - GP is confident'
            }
        
        # Check improvement plateau
        if len(self.history) >= plateau_cycles:
            recent = self.history.tail(plateau_cycles)
            best_qualities = recent['best_quality'].values
            
            # Calculate improvement over plateau period
            if len(best_qualities) >= plateau_cycles:
                improvement = (best_qualities[-1] - best_qualities[0]) / (best_qualities[0] + 1e-10)
                
                if improvement < improvement_threshold:
                    return {
                        'converged': True,
                        'reason': f'Improvement plateau: {improvement*100:.1f}% < {improvement_threshold*100}% over last {plateau_cycles} cycles',
                        'recommendation': 'Stop optimization - no significant improvement'
                    }
        
        # Not converged
        return {
            'converged': False,
            'reason': 'Still improving',
            'recommendation': 'Continue optimization'
        }
    
    def get_improvement_stats(self) -> Dict:
        """Get statistics about optimization progress."""
        if len(self.history) < 2:
            return {
                'total_improvement': 0.0,
                'avg_improvement_per_cycle': 0.0,
                'best_quality': 0.0,
                'cycles': len(self.history)
            }
        
        first_quality = self.history.iloc[0]['best_quality']
        latest_quality = self.history.iloc[-1]['best_quality']
        total_improvement = (latest_quality - first_quality) / (first_quality + 1e-10) * 100
        
        improvements = []
        for i in range(1, len(self.history)):
            prev = self.history.iloc[i-1]['best_quality']
            curr = self.history.iloc[i]['best_quality']
            if prev > 0:
                improvements.append((curr - prev) / prev * 100)
        
        avg_improvement = np.mean(improvements) if improvements else 0.0
        
        return {
            'total_improvement': total_improvement,
            'avg_improvement_per_cycle': avg_improvement,
            'best_quality': latest_quality,
            'cycles': len(self.history)
        }
    
    def get_recommended_exploration_rate(self, base_rate: float = 0.15) -> float:
        """
        Adaptively adjust exploration rate based on progress.
        
        If improving quickly: reduce exploration (more exploitation)
        If plateauing: increase exploration (find new regions)
        """
        if len(self.history) < 2:
            return base_rate
        
        # Calculate recent improvement rate
        recent = self.history.tail(3)
        if len(recent) >= 2:
            improvements = []
            for i in range(1, len(recent)):
                prev = recent.iloc[i-1]['best_quality']
                curr = recent.iloc[i]['best_quality']
                if prev > 0:
                    improvements.append((curr - prev) / prev)
            
            if improvements:
                avg_improvement = np.mean(improvements)
                
                # If improving quickly (>2% per cycle), reduce exploration
                if avg_improvement > 0.02:
                    return max(0.05, base_rate * 0.7)
                # If plateauing (<0.5% per cycle), increase exploration
                elif avg_improvement < 0.005:
                    return min(0.30, base_rate * 1.5)
        
        return base_rate
