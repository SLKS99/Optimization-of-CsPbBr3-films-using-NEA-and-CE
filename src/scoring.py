import math
from src.models import ExperimentParams

def calculate_target_score(pl_intensity: float, peak_pos: float, fwhm: float) -> float:
    """
    Combines PL Intensity, Peak Position, and FWHM into a single 'Quality Score'.
    
    Args:
        pl_intensity: Higher is better (Arbitrary units, e.g., 0-10000)
        peak_pos: Bandgap target. E.g., for CsPbI3, ~700nm is ideal (1.77eV).
                  Closer to target is better.
        fwhm: Full Width Half Max. Lower is better (narrower emission = better crystal).
    
    Returns:
        score: A float value representing overall quality.
    """
    if math.isnan(pl_intensity) or math.isnan(peak_pos) or math.isnan(fwhm):
        return 0.0
        
    # 1. PL Component (Linear scale or Log scale if varying wildly)
    # Normalized roughly to 0-1 range then scaled up. 
    # Assuming typical PL max is ~1000 for now.
    score_pl = pl_intensity / 1000.0
    
    # 2. Peak Position Component (Gaussian penalty around target)
    target_peak = 800.0 # nm (Adjust for specific material if needed)
    sigma_peak = 20.0   # Tolerance width
    score_peak = math.exp(-0.5 * ((peak_pos - target_peak) / sigma_peak)**2)
    
    # 3. FWHM Component (Inverse or bounded)
    # Typical good FWHM < 40 nm
    # Score = 1 if FWHM < 20, drops as FWHM increases
    score_fwhm = max(0, 1.0 - (fwhm - 20.0) / 50.0) # Drops to 0 at 70nm
    
    # Combine
    # Weights: PL (40%), Peak (30%), FWHM (30%)
    # Base score roughly 0-1
    total_score = (0.4 * score_pl) + (0.3 * score_peak) + (0.3 * score_fwhm)
    
    return total_score * 100.0 # Scale to 0-100 range

