"""
LLM Agent for Perovskite Film Optimization
==========================================
Uses Google Gemini to:
1. Learn patterns from GP results and experiment history
2. Analyze film images for uniformity
3. Suggest parameter restrictions and optimizations
4. Automatically propose config changes
"""

import os
import json
import yaml
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
from PIL import Image

# Try the new google-genai package first, fall back to deprecated one
USE_NEW_API = False
genai = None
types = None

try:
    import google.genai
    from google.genai import types
    genai = google.genai
    USE_NEW_API = True
except (ImportError, AttributeError):
    try:
        import google.generativeai as genai
        USE_NEW_API = False
    except ImportError:
        raise ImportError(
            "Neither google-genai nor google-generativeai package found. "
            "Install with: pip install google-genai"
        )

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class PatternInsight:
    """Represents a learned pattern from experiment data."""
    parameter: str
    observation: str
    confidence: float  # 0-1
    suggested_action: str
    evidence: str

@dataclass
class FilmAnalysis:
    """Results from analyzing a film image."""
    uniformity_score: float  # 0-100
    defects_detected: List[str]
    coverage_estimate: float  # 0-100 percent
    color_consistency: str
    recommendations: List[str]
    raw_analysis: str

@dataclass
class ConfigSuggestion:
    """A suggested modification to config.yaml."""
    parameter_path: str  # e.g., "spin_coating.second_step.speed_rpm"
    current_value: Any
    suggested_value: Any
    reasoning: str
    expected_improvement: str
    confidence: float

# ---------------------------------------------------------------------------
# LLM Agent Class
# ---------------------------------------------------------------------------
class PerovskiteOptimizationAgent:
    """
    Autonomous agent that learns from GP results and suggests optimizations.
    """
    
    SYSTEM_PROMPT = """You are an expert materials scientist specializing in perovskite thin-film fabrication. 
You analyze experimental data from a Gaussian Process Bayesian Optimization system and provide actionable insights.

Your expertise includes:
- Spin coating parameters (speed, time, acceleration, two-step processes)
- Annealing/drying conditions (temperature, time)
- Solvent systems and antisolvent quenching
- Film quality metrics (uniformity, coverage, crystallinity)
- Structure-property relationships in perovskites

When analyzing data:
1. Look for correlations between parameters and outcomes
2. Identify parameter ranges that consistently produce good/bad results
3. Consider physical mechanisms (e.g., why certain spin speeds work better)
4. Suggest specific, actionable parameter restrictions

Always provide evidence-based recommendations with confidence levels."""

    def __init__(self, api_key: str, config_path: str = "config.yaml"):
        """Initialize the agent with Gemini API."""
        self.api_key = api_key
        self.config_path = config_path
        self.insights_history: List[PatternInsight] = []
        self.analysis_history: List[FilmAnalysis] = []
        
        if USE_NEW_API:
            # New google-genai API
            self.client = genai.Client(api_key=api_key)
            self.model_name = "gemini-2.0-flash"
        else:
            # Legacy google-generativeai API
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.vision_model = genai.GenerativeModel('gemini-pro-vision')
        
    def _load_config(self) -> dict:
        """Load current config.yaml."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _save_config(self, config: dict):
        """Save modified config.yaml."""
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False, default_flow_style=False)
    
    def _format_experiment_data(self, history_df: pd.DataFrame, candidates_df: pd.DataFrame = None) -> str:
        """Format experiment data for LLM context."""
        context = "## Experiment History\n"
        
        if history_df is not None and not history_df.empty:
            context += f"Total past experiments: {len(history_df)}\n\n"
            context += "### Recent Experiments (last 20):\n"
            context += history_df.tail(20).to_string(index=False)
            context += "\n\n"
            
            # Add summary statistics
            context += "### Summary Statistics:\n"
            numeric_cols = history_df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                context += f"- {col}: mean={history_df[col].mean():.2f}, std={history_df[col].std():.2f}, "
                context += f"min={history_df[col].min():.2f}, max={history_df[col].max():.2f}\n"
        else:
            context += "No experiment history available yet.\n"
        
        if candidates_df is not None and not candidates_df.empty:
            context += "\n## GP-Generated Candidates (Top 20 by Rank Score):\n"
            top_candidates = candidates_df.nlargest(20, 'Rank_Score') if 'Rank_Score' in candidates_df.columns else candidates_df.head(20)
            context += top_candidates.to_string(index=False)
            context += "\n"
        
        return context
    
    def _format_config_context(self) -> str:
        """Format current config for LLM context."""
        config = self._load_config()
        context = "## Current Configuration:\n```yaml\n"
        context += yaml.dump(config, default_flow_style=False)
        context += "```\n"
        return context

    def analyze_patterns(
        self, 
        history_df: pd.DataFrame, 
        candidates_df: pd.DataFrame = None,
        quality_column: str = "Quality_Score"
    ) -> List[PatternInsight]:
        """
        Analyze experiment history and GP candidates to find patterns.
        
        Args:
            history_df: DataFrame with past experiment results
            candidates_df: DataFrame with GP-generated candidates
            quality_column: Column name for quality metric
            
        Returns:
            List of PatternInsight objects
        """
        data_context = self._format_experiment_data(history_df, candidates_df)
        config_context = self._format_config_context()
        
        prompt = f"""{data_context}

{config_context}

Based on this experimental data, identify patterns and correlations:

1. Which parameter ranges consistently produce HIGH quality films?
2. Which parameter ranges consistently produce LOW quality films?
3. Are there any parameter combinations that seem particularly effective or problematic?
4. What physical mechanisms might explain these patterns?

For each pattern you identify, provide:
- The parameter(s) involved
- The observation (what you noticed)
- Confidence level (0-1, based on data support)
- Suggested action (specific parameter restriction or range)
- Evidence (specific data points or statistics supporting this)

Format your response as a JSON array of objects with keys: 
parameter, observation, confidence, suggested_action, evidence

Only include patterns with confidence >= 0.5. Be specific and actionable."""

        if USE_NEW_API:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            response_text = response.text
        else:
            response = self.model.generate_content(prompt)
            response_text = response.text
        
        try:
            # Extract JSON from response
            # Find JSON array in response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                patterns_data = json.loads(json_str)
                
                insights = []
                for p in patterns_data:
                    insight = PatternInsight(
                        parameter=p.get('parameter', ''),
                        observation=p.get('observation', ''),
                        confidence=float(p.get('confidence', 0.5)),
                        suggested_action=p.get('suggested_action', ''),
                        evidence=p.get('evidence', '')
                    )
                    insights.append(insight)
                    self.insights_history.append(insight)
                
                return insights
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Could not parse pattern analysis response: {e}")
            # Return raw analysis as single insight
            return [PatternInsight(
                parameter="general",
                observation=response.text[:500],
                confidence=0.5,
                suggested_action="Review raw analysis",
                evidence="See raw response"
            )]
        
        return []

    def analyze_film_image(self, image_path: str, experiment_id: str = None) -> FilmAnalysis:
        """
        Analyze a film image for quality assessment.
        
        Args:
            image_path: Path to the film image
            experiment_id: Optional ID to link to experiment
            
        Returns:
            FilmAnalysis object with uniformity and defect information
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        img = Image.open(image_path)
        
        prompt = """Analyze this perovskite thin film image for quality assessment.

Evaluate the following aspects:

1. **Uniformity Score (0-100)**: How uniform is the film? 
   - 90-100: Excellent, mirror-like uniformity
   - 70-89: Good, minor variations
   - 50-69: Fair, visible non-uniformity
   - Below 50: Poor, significant defects

2. **Defects Detected**: List any visible defects:
   - Pinholes
   - Cracks
   - Dewetting
   - Coffee-ring effects
   - Crystallization defects
   - Color variations
   - Edge effects

3. **Coverage Estimate (0-100%)**: What percentage of the substrate is covered?

4. **Color Consistency**: Describe the color uniformity (uniform/gradient/patchy)

5. **Recommendations**: What processing changes might improve this film?

Format your response as JSON with keys:
uniformity_score, defects_detected (array), coverage_estimate, color_consistency, recommendations (array), raw_analysis (your detailed observations)"""

        if USE_NEW_API:
            # Upload image for new API
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=f"image/{img.format.lower() if img.format else 'jpeg'}"),
                    prompt
                ]
            )
        else:
            response = self.vision_model.generate_content([prompt, img])
        
        try:
            response_text = response.text
            # Find JSON in response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                analysis_data = json.loads(json_str)
                
                analysis = FilmAnalysis(
                    uniformity_score=float(analysis_data.get('uniformity_score', 50)),
                    defects_detected=analysis_data.get('defects_detected', []),
                    coverage_estimate=float(analysis_data.get('coverage_estimate', 0)),
                    color_consistency=analysis_data.get('color_consistency', 'unknown'),
                    recommendations=analysis_data.get('recommendations', []),
                    raw_analysis=analysis_data.get('raw_analysis', response_text)
                )
                self.analysis_history.append(analysis)
                return analysis
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Could not parse image analysis response: {e}")
            return FilmAnalysis(
                uniformity_score=0,
                defects_detected=["Analysis failed"],
                coverage_estimate=0,
                color_consistency="unknown",
                recommendations=["Retry analysis"],
                raw_analysis=response.text
            )

    def suggest_config_changes(
        self, 
        history_df: pd.DataFrame,
        candidates_df: pd.DataFrame = None,
        recent_film_analyses: List[FilmAnalysis] = None
    ) -> List[ConfigSuggestion]:
        """
        Suggest specific changes to config.yaml based on all available data.
        
        Args:
            history_df: Experiment history
            candidates_df: GP candidates
            recent_film_analyses: Recent film image analyses
            
        Returns:
            List of ConfigSuggestion objects
        """
        data_context = self._format_experiment_data(history_df, candidates_df)
        config_context = self._format_config_context()
        
        # Add film analysis context if available
        film_context = ""
        if recent_film_analyses:
            film_context = "\n## Recent Film Analyses:\n"
            for i, analysis in enumerate(recent_film_analyses[-5:], 1):
                film_context += f"\n### Film {i}:\n"
                film_context += f"- Uniformity: {analysis.uniformity_score}/100\n"
                film_context += f"- Coverage: {analysis.coverage_estimate}%\n"
                film_context += f"- Defects: {', '.join(analysis.defects_detected) if analysis.defects_detected else 'None'}\n"
                film_context += f"- Recommendations: {', '.join(analysis.recommendations)}\n"
        
        # Add insights history
        insights_context = ""
        if self.insights_history:
            insights_context = "\n## Previously Identified Patterns:\n"
            for insight in self.insights_history[-10:]:
                insights_context += f"- {insight.parameter}: {insight.observation} (confidence: {insight.confidence:.2f})\n"
        
        prompt = f"""{data_context}

{config_context}

{film_context}

{insights_context}

Based on all available data, suggest specific modifications to config.yaml that would improve film quality.

For each suggestion, consider:
1. What parameter should change?
2. What is the current value?
3. What should the new value be?
4. Why will this improve results? (physical reasoning)
5. What improvement do you expect?
6. How confident are you? (0-1)

Focus on:
- Spin coating parameters (speed, time, acceleration, two-step settings)
- Annealing conditions (temperature, time)
- Concentration ranges
- Antisolvent parameters

Format your response as a JSON array with objects containing:
parameter_path, current_value, suggested_value, reasoning, expected_improvement, confidence

Only suggest changes with confidence >= 0.6. Be specific with values (use actual numbers, not ranges)."""

        if USE_NEW_API:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            response_text = response.text
        else:
            response = self.model.generate_content(prompt)
            response_text = response.text
        
        try:
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                suggestions_data = json.loads(json_str)
                
                suggestions = []
                for s in suggestions_data:
                    suggestion = ConfigSuggestion(
                        parameter_path=s.get('parameter_path', ''),
                        current_value=s.get('current_value'),
                        suggested_value=s.get('suggested_value'),
                        reasoning=s.get('reasoning', ''),
                        expected_improvement=s.get('expected_improvement', ''),
                        confidence=float(s.get('confidence', 0.5))
                    )
                    suggestions.append(suggestion)
                
                return suggestions
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Could not parse config suggestions: {e}")
            return []
        
        return []

    def apply_suggestion(self, suggestion: ConfigSuggestion, auto_backup: bool = True) -> bool:
        """
        Apply a config suggestion to config.yaml.
        
        Args:
            suggestion: The ConfigSuggestion to apply
            auto_backup: Whether to backup config before modifying
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config = self._load_config()
            
            # Backup if requested
            if auto_backup:
                backup_path = self.config_path + '.backup'
                with open(backup_path, 'w') as f:
                    yaml.dump(config, f, sort_keys=False, default_flow_style=False)
            
            # Navigate to the parameter using dot notation
            path_parts = suggestion.parameter_path.split('.')
            current = config
            for part in path_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the new value
            final_key = path_parts[-1]
            current[final_key] = suggestion.suggested_value
            
            # Save modified config
            self._save_config(config)
            print(f"Applied: {suggestion.parameter_path} = {suggestion.suggested_value}")
            return True
            
        except Exception as e:
            print(f"Error applying suggestion: {e}")
            return False

    def run_optimization_cycle(
        self,
        history_df: pd.DataFrame,
        candidates_df: pd.DataFrame = None,
        film_image_paths: List[str] = None,
        auto_apply: bool = False,
        min_confidence: float = 0.7
    ) -> Dict[str, Any]:
        """
        Run a full optimization cycle: analyze patterns, evaluate films, suggest changes.
        
        Args:
            history_df: Experiment history
            candidates_df: GP candidates
            film_image_paths: Paths to film images to analyze
            auto_apply: Whether to automatically apply high-confidence suggestions
            min_confidence: Minimum confidence to auto-apply
            
        Returns:
            Dictionary with analysis results and suggestions
        """
        results = {
            'patterns': [],
            'film_analyses': [],
            'suggestions': [],
            'applied_changes': []
        }
        
        print("=" * 60)
        print("LLM Agent Optimization Cycle")
        print("=" * 60)
        
        # 1. Analyze patterns from data
        print("\n[1/4] Analyzing patterns from experiment data...")
        patterns = self.analyze_patterns(history_df, candidates_df)
        results['patterns'] = [asdict(p) for p in patterns]
        print(f"Found {len(patterns)} patterns")
        for p in patterns:
            print(f"  - {p.parameter}: {p.observation[:60]}... (conf: {p.confidence:.2f})")
        
        # 2. Analyze film images if provided
        if film_image_paths:
            print(f"\n[2/4] Analyzing {len(film_image_paths)} film images...")
            for img_path in film_image_paths:
                try:
                    analysis = self.analyze_film_image(img_path)
                    results['film_analyses'].append(asdict(analysis))
                    print(f"  - {img_path}: Uniformity={analysis.uniformity_score}/100, Coverage={analysis.coverage_estimate}%")
                except Exception as e:
                    print(f"  - Error analyzing {img_path}: {e}")
        else:
            print("\n[2/4] No film images provided, skipping image analysis...")
        
        # 3. Generate config suggestions
        print("\n[3/4] Generating config suggestions...")
        film_analyses = self.analysis_history[-len(film_image_paths):] if film_image_paths else None
        suggestions = self.suggest_config_changes(history_df, candidates_df, film_analyses)
        results['suggestions'] = [asdict(s) for s in suggestions]
        print(f"Generated {len(suggestions)} suggestions")
        for s in suggestions:
            print(f"  - {s.parameter_path}: {s.current_value} -> {s.suggested_value} (conf: {s.confidence:.2f})")
        
        # 4. Apply high-confidence suggestions if auto_apply is enabled
        if auto_apply:
            print(f"\n[4/4] Auto-applying suggestions with confidence >= {min_confidence}...")
            for s in suggestions:
                if s.confidence >= min_confidence:
                    if self.apply_suggestion(s):
                        results['applied_changes'].append(asdict(s))
                        print(f"  ✓ Applied: {s.parameter_path}")
                    else:
                        print(f"  ✗ Failed: {s.parameter_path}")
        else:
            print("\n[4/4] Auto-apply disabled. Review suggestions and apply manually.")
        
        # 5. Generate recommendation summary
        print("\n[5/5] Generating recommendation summary...")
        try:
            summary = self.get_recommendation_summary()
            results['summary'] = summary
            print("\n" + "-" * 60)
            print("RECOMMENDATION SUMMARY")
            print("-" * 60)
            print(summary)
        except Exception as e:
            print(f"  Warning: Could not generate summary: {e}")
            results['summary'] = None
        
        print("\n" + "=" * 60)
        print("Optimization cycle complete!")
        print("=" * 60)
        
        return results

    def get_recommendation_summary(self) -> str:
        """Generate a human-readable summary of all recommendations."""
        config = self._load_config()
        
        prompt = f"""Based on the following insights and analyses, provide a concise summary of recommendations for the next experimental run.

## Accumulated Insights:
{json.dumps([asdict(i) for i in self.insights_history[-20:]], indent=2) if self.insights_history else "No insights yet."}

## Film Analyses:
{json.dumps([asdict(a) for a in self.analysis_history[-10:]], indent=2) if self.analysis_history else "No film analyses yet."}

## Current Config:
```yaml
{yaml.dump(config, default_flow_style=False)}
```

Provide a clear, actionable summary in plain English:
1. What's working well?
2. What needs improvement?
3. Top 3 specific changes to try next
4. Any warnings or concerns?

Keep it concise but informative."""

        if USE_NEW_API:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
        else:
            response = self.model.generate_content(prompt)
        return response.text


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------
def create_agent(api_key: str = None, config_path: str = "config.yaml") -> PerovskiteOptimizationAgent:
    """
    Create an optimization agent.
    
    Args:
        api_key: Gemini API key. If None, reads from config or GEMINI_API_KEY env var.
        config_path: Path to config.yaml
        
    Returns:
        PerovskiteOptimizationAgent instance
    """
    if api_key is None:
        # Try to read from config first
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
                api_key = cfg.get('llm_agent', {}).get('api_key')
        
        # Fall back to environment variable
        if not api_key:
            api_key = os.environ.get('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError("No API key provided. Set in config.yaml, GEMINI_API_KEY env var, or pass api_key parameter.")
    
    return PerovskiteOptimizationAgent(api_key, config_path)


def quick_film_analysis(image_path: str, api_key: str = None) -> FilmAnalysis:
    """
    Quick one-off film analysis without full agent setup.
    
    Args:
        image_path: Path to film image
        api_key: Gemini API key
        
    Returns:
        FilmAnalysis object
    """
    agent = create_agent(api_key)
    return agent.analyze_film_image(image_path)
