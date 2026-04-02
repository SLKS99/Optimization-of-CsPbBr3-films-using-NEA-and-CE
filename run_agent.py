#!/usr/bin/env python3
"""
LLM Agent Runner for Perovskite Optimization
=============================================
Standalone script to run the LLM agent for:
- Pattern analysis from experiment history
- Film image quality assessment
- Config optimization suggestions

Usage:
    python run_agent.py --analyze                    # Analyze patterns only
    python run_agent.py --images path/to/images/    # Analyze film images
    python run_agent.py --optimize                  # Full optimization cycle
    python run_agent.py --auto-apply                # Auto-apply suggestions
    python run_agent.py --summary                   # Get recommendation summary
"""

import argparse
import os
import sys
import json
import glob
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm_agent import create_agent, PerovskiteOptimizationAgent


def load_experiment_data(config_path: str = "config.yaml") -> tuple:
    """Load experiment history and candidates from data files."""
    import yaml
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load history
    history_path = cfg['data'].get('history', 'data/templates/experiments_log.csv')
    history_df = pd.DataFrame()
    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)
        print(f"Loaded {len(history_df)} experiments from history")
    else:
        print(f"No history file found at {history_path}")
    
    # Load candidates if available
    candidates_path = 'data/candidates_analysis.csv'
    candidates_df = pd.DataFrame()
    if os.path.exists(candidates_path):
        candidates_df = pd.read_csv(candidates_path)
        print(f"Loaded {len(candidates_df)} GP candidates")
    else:
        print("No candidates file found")
    
    return history_df, candidates_df


def find_film_images(image_dir: str) -> list:
    """Find all image files in a directory."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.webp', '*.bmp']
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(image_dir, ext)))
        images.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    return sorted(images)


def main():
    parser = argparse.ArgumentParser(
        description="LLM Agent for Perovskite Film Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_agent.py --analyze
  python run_agent.py --images ./film_photos/
  python run_agent.py --optimize --auto-apply
  python run_agent.py --analyze-image ./film.jpg
        """
    )
    
    parser.add_argument('--api-key', type=str, 
                        help='Gemini API key (or set GEMINI_API_KEY env var)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config.yaml')
    
    # Action flags
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze patterns from experiment data')
    parser.add_argument('--images', type=str, metavar='DIR',
                        help='Directory containing film images to analyze')
    parser.add_argument('--analyze-image', type=str, metavar='FILE',
                        help='Analyze a single film image')
    parser.add_argument('--optimize', action='store_true',
                        help='Run full optimization cycle')
    parser.add_argument('--auto-apply', action='store_true',
                        help='Auto-apply high-confidence suggestions')
    parser.add_argument('--min-confidence', type=float, default=0.7,
                        help='Minimum confidence for auto-apply (default: 0.7)')
    parser.add_argument('--summary', action='store_true',
                        help='Get recommendation summary')
    parser.add_argument('--output', type=str, metavar='FILE',
                        help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Check for API key (from arg, env var, or config file)
    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    
    # Try reading from config if not provided
    if not api_key and os.path.exists(args.config):
        try:
            import yaml
            with open(args.config, 'r') as f:
                cfg = yaml.safe_load(f)
                api_key = cfg.get('llm_agent', {}).get('api_key')
        except Exception:
            pass
    
    if not api_key:
        print("Error: No Gemini API key provided.")
        print("Options:")
        print("  1. Set in config.yaml: llm_agent.api_key")
        print("  2. Set GEMINI_API_KEY environment variable")
        print("  3. Use --api-key flag")
        sys.exit(1)
    
    # Create agent
    print("Initializing LLM Agent...")
    agent = create_agent(api_key, args.config)
    
    # Load data
    history_df, candidates_df = load_experiment_data(args.config)
    
    results = {}
    
    # Handle single image analysis
    if args.analyze_image:
        print(f"\nAnalyzing image: {args.analyze_image}")
        analysis = agent.analyze_film_image(args.analyze_image)
        print(f"\n{'='*50}")
        print("FILM ANALYSIS RESULTS")
        print(f"{'='*50}")
        print(f"Uniformity Score: {analysis.uniformity_score}/100")
        print(f"Coverage Estimate: {analysis.coverage_estimate}%")
        print(f"Color Consistency: {analysis.color_consistency}")
        print(f"\nDefects Detected:")
        for defect in analysis.defects_detected:
            print(f"  - {defect}")
        print(f"\nRecommendations:")
        for rec in analysis.recommendations:
            print(f"  - {rec}")
        print(f"\nDetailed Analysis:\n{analysis.raw_analysis}")
        results['film_analysis'] = {
            'uniformity_score': analysis.uniformity_score,
            'coverage_estimate': analysis.coverage_estimate,
            'defects': analysis.defects_detected,
            'recommendations': analysis.recommendations
        }
    
    # Handle pattern analysis
    elif args.analyze:
        print("\nAnalyzing patterns from experiment data...")
        patterns = agent.analyze_patterns(history_df, candidates_df)
        print(f"\n{'='*50}")
        print("PATTERN ANALYSIS RESULTS")
        print(f"{'='*50}")
        for i, p in enumerate(patterns, 1):
            print(f"\n[Pattern {i}] {p.parameter}")
            print(f"  Observation: {p.observation}")
            print(f"  Confidence: {p.confidence:.2f}")
            print(f"  Suggested Action: {p.suggested_action}")
            print(f"  Evidence: {p.evidence}")
        results['patterns'] = [
            {'parameter': p.parameter, 'observation': p.observation, 
             'confidence': p.confidence, 'action': p.suggested_action}
            for p in patterns
        ]
    
    # Handle full optimization cycle
    elif args.optimize:
        film_paths = []
        if args.images:
            film_paths = find_film_images(args.images)
            print(f"Found {len(film_paths)} film images")
        
        results = agent.run_optimization_cycle(
            history_df=history_df,
            candidates_df=candidates_df,
            film_image_paths=film_paths if film_paths else None,
            auto_apply=args.auto_apply,
            min_confidence=args.min_confidence
        )
    
    # Handle image directory analysis
    elif args.images:
        film_paths = find_film_images(args.images)
        print(f"\nFound {len(film_paths)} images to analyze")
        results['film_analyses'] = []
        for img_path in film_paths:
            print(f"\nAnalyzing: {img_path}")
            try:
                analysis = agent.analyze_film_image(img_path)
                print(f"  Uniformity: {analysis.uniformity_score}/100, Coverage: {analysis.coverage_estimate}%")
                results['film_analyses'].append({
                    'image': img_path,
                    'uniformity': analysis.uniformity_score,
                    'coverage': analysis.coverage_estimate,
                    'defects': analysis.defects_detected
                })
            except Exception as e:
                print(f"  Error: {e}")
    
    # Handle summary request
    if args.summary or (not args.analyze and not args.images and not args.optimize and not args.analyze_image):
        print("\nGenerating recommendation summary...")
        summary = agent.get_recommendation_summary()
        print(f"\n{'='*50}")
        print("RECOMMENDATION SUMMARY")
        print(f"{'='*50}")
        print(summary)
        results['summary'] = summary
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
