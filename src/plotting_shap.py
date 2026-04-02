"""
SHAP plotting functions for model interpretability.
Shows which features (FA ratio, spin parameters, etc.) have the most impact on quality predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PathCollection # Import PathCollection
import pandas as pd
from typing import Optional
from src.tree_model import ProcessTreeLearner
from src.learner import ActiveLearner
from matplotlib.collections import PathCollection


def plot_shap_beeswarm(ax, tree_learner: ProcessTreeLearner, history_df: pd.DataFrame, quality_learner: ActiveLearner):
    """
    Plot SHAP beeswarm plot showing feature importance for the decision tree model.
    
    Shows which features (FA ratio, annealing temp, spin speed, concentration, antisolvent)
    have the most impact on quality predictions.
    """
    if tree_learner is None or not tree_learner.is_trained or tree_learner.model is None:
        ax.text(0.5, 0.5, 'Decision Tree Not Trained\n\nSHAP requires a trained model', 
               ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title('SHAP Feature Impact Analysis', fontsize=12)
        return
    
    if history_df is None or history_df.empty:
        ax.text(0.5, 0.5, 'No History Data\n\nSHAP requires training data', 
               ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title('SHAP Feature Impact Analysis', fontsize=12)
        return
    
    try:
        import shap
        
        # Prepare feature data
        X_rows = []
        for _, row in history_df.iterrows():
            pl_val = row.get("PL_Intensity", row.get("PL_Intensity ", None))
            if pd.isna(pl_val) and pd.isna(row.get("Stability_Hours")):
                continue
            
            x = tree_learner._featurize_log_row(row)
            if x is not None:
                X_rows.append(x)
        
        if len(X_rows) < 10:
            ax.text(0.5, 0.5, 'Not Enough Data\n\nNeed 10+ experiments for SHAP', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.set_title('SHAP Feature Impact Analysis', fontsize=12)
            return
        
        X = np.vstack(X_rows)
        
        # Feature names
        feature_names = ['FA Ratio', 'Annealing Temp (°C)', 'Spin Speed (rpm)',
                        'Is Chloroform', 'Is Chlorobenzene', 'Is Toluene', 'Is None Antisolvent']
        
        # Create SHAP explainer
        # Use TreeExplainer for RandomForest (much faster than KernelExplainer)
        explainer = shap.TreeExplainer(tree_learner.model)
        
        # Calculate SHAP values (use subset if too many samples for performance)
        max_samples = 100  # Limit samples for performance
        if len(X) > max_samples:
            # Sample randomly
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        shap_values = explainer.shap_values(X_sample)
        
        # Handle SHAP values - TreeExplainer returns array directly for regression
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # For binary classification, take first class
        
        # Create SHAP Explanation object with feature names
        # This is the proper way to pass feature names to SHAP plots
        shap_explanation = shap.Explanation(
            values=shap_values,
            base_values=np.zeros(len(X_sample)),  # For regression, base is typically 0
            data=X_sample,
            feature_names=feature_names
        )
        
        # Use SHAP's built-in beeswarm plot directly
        # SHAP creates its own figure, so we need to capture and copy it
        # Save current figure state
        current_fig = ax.figure
        current_fig_num = current_fig.number
        
        # Create a temporary figure for SHAP to plot to
        temp_fig = plt.figure(figsize=(10, 6))
        temp_fig_num = temp_fig.number
        
        try:
            # Set this as the current figure
            plt.figure(temp_fig_num)
            
            # Plot beeswarm using Explanation object (which includes feature names)
            shap.plots.beeswarm(shap_explanation, show=False)
            
            # Get the figure SHAP created/used
            shap_fig = plt.gcf()
            shap_fig_num = shap_fig.number
            
            # Check if SHAP used our temp figure or created a new one
            if shap_fig_num == temp_fig_num:
                # SHAP plotted to our temp figure - use it directly
                shap_axes = shap_fig.get_axes()
                if shap_axes and len(shap_axes) > 0:
                    shap_ax = shap_axes[0]
                    
                    # Clear our target axis
                    ax.clear()
                    
                    # Copy all plot elements from SHAP's axis
                    copied_any = False
                    for element in shap_ax.get_children():
                        # Skip text elements
                        if isinstance(element, plt.Text):
                            continue
                        
                        # Copy patches
                        if isinstance(element, plt.Polygon):
                            try:
                                ax.add_patch(plt.Polygon(
                                    element.get_xy(),
                                    facecolor=element.get_facecolor(),
                                    edgecolor=element.get_edgecolor(),
                                    alpha=element.get_alpha() if hasattr(element, 'get_alpha') else 0.6,
                                    linewidth=element.get_linewidth() if hasattr(element, 'get_linewidth') else 0.5
                                ))
                                copied_any = True
                            except:
                                pass
                        # Copy lines
                        elif isinstance(element, plt.Line2D):
                            try:
                                ax.add_line(element)
                                copied_any = True
                            except:
                                pass
                        # Copy collections (scatter points, etc.)
                        elif hasattr(element, 'get_offsets'):
                            offsets = element.get_offsets()
                            if len(offsets) > 0:
                                try:
                                    # Get colors
                                    colors = element.get_array()
                                    if colors is None:
                                        fc = element.get_facecolors()
                                        if fc.size > 0:
                                            colors = fc[:, :3] if fc.shape[1] >= 3 else fc
                                        else:
                                            colors = 'darkred'
                                    
                                    # Get sizes
                                    sizes = element.get_sizes()
                                    if isinstance(sizes, np.ndarray):
                                        sizes = sizes[0] if sizes.size > 0 else 20
                                    else:
                                        sizes = 20
                                    
                                    # Plot points
                                    ax.scatter(offsets[:, 0], offsets[:, 1],
                                             c=colors, s=sizes,
                                             alpha=element.get_alpha() if hasattr(element, 'get_alpha') else 0.6,
                                             cmap=element.get_cmap() if hasattr(element, 'get_cmap') else None,
                                             edgecolors=element.get_edgecolors() if hasattr(element, 'get_edgecolors') else 'black',
                                             linewidths=element.get_linewidths() if hasattr(element, 'get_linewidths') else 0.5)
                                    copied_any = True
                                except Exception as copy_err:
                                    # If copying fails, skip this element
                                    pass
                    
                    if copied_any:
                        # Copy axis properties
                        ax.set_xlabel(shap_ax.get_xlabel() if shap_ax.get_xlabel() else 'SHAP Value', fontsize=11)
                        ax.set_ylabel(shap_ax.get_ylabel() if shap_ax.get_ylabel() else 'Features', fontsize=11)
                        ax.set_xlim(shap_ax.get_xlim())
                        ax.set_ylim(shap_ax.get_ylim())
                        ax.set_yticks(shap_ax.get_yticks())
                        ax.set_yticklabels(shap_ax.get_yticklabels(), fontsize=10)

                        # Try to get colorbar info from SHAP's scatter plot if available
                        mappable = None
                        for collection in shap_ax.collections:
                            if isinstance(collection, PathCollection) and collection.get_array() is not None:
                                # This is likely the main beeswarm points
                                cmap = collection.get_cmap()
                                norm = collection.get_norm()
                                if cmap and norm:
                                    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                                    mappable.set_array(collection.get_array()) # Use the data that was colored
                                    break

                        if mappable:
                            cbar = plt.colorbar(mappable, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
                            cbar.set_label('Feature Value (Low \u2192 High)', fontsize=10) # Unicode arrow
                        else:
                            # Fallback if we can't extract mappable directly from SHAP's plot
                            print("Warning: Could not extract mappable for SHAP colorbar. Using default.")
                            # Create a dummy mappable for a general explanation
                            cmap_fallback = plt.cm.get_cmap('RdBu')
                            norm_fallback = plt.Normalize(vmin=-1, vmax=1) # Common range for normalized features
                            mappable_fallback = plt.cm.ScalarMappable(norm=norm_fallback, cmap=cmap_fallback)
                            mappable_fallback.set_array([]) # Empty array
                            cbar = plt.colorbar(mappable_fallback, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
                            cbar.set_label('Feature Value (Low \u2192 High)', fontsize=10)

                    else:
                        ax.text(0.5, 0.5, 'SHAP Plot Copy Failed\n\nCould not copy plot elements', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=11)
                    
                    # Close temp figure
                    plt.close(temp_fig)
                else:
                    ax.text(0.5, 0.5, 'SHAP Plot Error\n\nNo axes found in SHAP figure', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=11)
                    plt.close(temp_fig)
            elif shap_fig_num != current_fig_num:
                # SHAP created a new figure, copy its content to our axis
                shap_axes = shap_fig.get_axes()
                if shap_axes and len(shap_axes) > 0:
                    shap_ax = shap_axes[0]
                    
                    # Clear our axis only if we successfully got SHAP's axis
                    ax.clear()
                    
                    # Copy all plot elements from SHAP's axis
                    copied_any = False
                    for element in shap_ax.get_children():
                        # Skip text elements
                        if isinstance(element, plt.Text):
                            continue
                        
                        # Copy patches
                        if isinstance(element, plt.Polygon):
                            try:
                                ax.add_patch(plt.Polygon(
                                    element.get_xy(),
                                    facecolor=element.get_facecolor(),
                                    edgecolor=element.get_edgecolor(),
                                    alpha=element.get_alpha() if hasattr(element, 'get_alpha') else 0.6,
                                    linewidth=element.get_linewidth() if hasattr(element, 'get_linewidth') else 0.5
                                ))
                                copied_any = True
                            except:
                                pass
                        # Copy lines
                        elif isinstance(element, plt.Line2D):
                            try:
                                ax.add_line(element)
                                copied_any = True
                            except:
                                pass
                        # Copy collections (scatter points, etc.)
                        elif hasattr(element, 'get_offsets'):
                            offsets = element.get_offsets()
                            if len(offsets) > 0:
                                try:
                                    # Get colors
                                    colors = element.get_array()
                                    if colors is None:
                                        fc = element.get_facecolors()
                                        if fc.size > 0:
                                            colors = fc[:, :3] if fc.shape[1] >= 3 else fc
                                        else:
                                            colors = 'darkred'
                                    
                                    # Get sizes
                                    sizes = element.get_sizes()
                                    if isinstance(sizes, np.ndarray):
                                        sizes = sizes[0] if sizes.size > 0 else 20
                                    else:
                                        sizes = 20
                                    
                                    # Plot points
                                    ax.scatter(offsets[:, 0], offsets[:, 1],
                                             c=colors, s=sizes,
                                             alpha=element.get_alpha() if hasattr(element, 'get_alpha') else 0.6,
                                             cmap=element.get_cmap() if hasattr(element, 'get_cmap') else None,
                                             edgecolors=element.get_edgecolors() if hasattr(element, 'get_edgecolors') else 'black',
                                             linewidths=element.get_linewidths() if hasattr(element, 'get_linewidths') else 0.5)
                                    copied_any = True
                                except Exception as copy_err:
                                    # If copying fails, skip this element
                                    pass
                    
                    if copied_any:
                        # Copy axis properties
                        ax.set_xlabel(shap_ax.get_xlabel() if shap_ax.get_xlabel() else 'SHAP Value', fontsize=11)
                        ax.set_ylabel(shap_ax.get_ylabel() if shap_ax.get_ylabel() else 'Features', fontsize=11)
                        ax.set_xlim(shap_ax.get_xlim())
                        ax.set_ylim(shap_ax.get_ylim())
                        ax.set_yticks(shap_ax.get_yticks())
                        ax.set_yticklabels(shap_ax.get_yticklabels(), fontsize=10)
                    else:
                        # If nothing was copied, show error
                        ax.text(0.5, 0.5, 'SHAP Plot Copy Failed\n\nCould not copy plot elements', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=11)
                    
                    # Close SHAP figure
                    plt.close(shap_fig)
                else:
                    ax.text(0.5, 0.5, 'SHAP Plot Error\n\nNo axes found in SHAP figure', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=11)
                
                # Close temp figure
                plt.close(temp_fig)
            else:
                # SHAP plotted to temp figure - copy from there
                shap_axes = temp_fig.get_axes()
                if shap_axes and len(shap_axes) > 0:
                    shap_ax = shap_axes[0]
                    ax.clear()
                    
                    # Copy elements (same as above)
                    copied_any = False
                    for element in shap_ax.get_children():
                        if isinstance(element, plt.Text):
                            continue
                        if isinstance(element, plt.Polygon):
                            try:
                                ax.add_patch(plt.Polygon(
                                    element.get_xy(),
                                    facecolor=element.get_facecolor(),
                                    edgecolor=element.get_edgecolor(),
                                    alpha=element.get_alpha() if hasattr(element, 'get_alpha') else 0.6,
                                    linewidth=element.get_linewidth() if hasattr(element, 'get_linewidth') else 0.5
                                ))
                                copied_any = True
                            except:
                                pass
                        elif isinstance(element, plt.Line2D):
                            try:
                                ax.add_line(element)
                                copied_any = True
                            except:
                                pass
                        elif hasattr(element, 'get_offsets'):
                            offsets = element.get_offsets()
                            if len(offsets) > 0:
                                try:
                                    colors = element.get_array()
                                    if colors is None:
                                        fc = element.get_facecolors()
                                        if fc.size > 0:
                                            colors = fc[:, :3] if fc.shape[1] >= 3 else fc
                                        else:
                                            colors = 'darkred'
                                    sizes = element.get_sizes()
                                    if isinstance(sizes, np.ndarray):
                                        sizes = sizes[0] if sizes.size > 0 else 20
                                    else:
                                        sizes = 20
                                    ax.scatter(offsets[:, 0], offsets[:, 1],
                                             c=colors, s=sizes,
                                             alpha=element.get_alpha() if hasattr(element, 'get_alpha') else 0.6,
                                             cmap=element.get_cmap() if hasattr(element, 'get_cmap') else None,
                                             edgecolors=element.get_edgecolors() if hasattr(element, 'get_edgecolors') else 'black',
                                             linewidths=element.get_linewidths() if hasattr(element, 'get_linewidths') else 0.5)
                                    copied_any = True
                                except:
                                    pass
                    
                    if copied_any:
                        ax.set_xlabel(shap_ax.get_xlabel() if shap_ax.get_xlabel() else 'SHAP Value', fontsize=11)
                        ax.set_ylabel(shap_ax.get_ylabel() if shap_ax.get_ylabel() else 'Features', fontsize=11)
                        ax.set_xlim(shap_ax.get_xlim())
                        ax.set_ylim(shap_ax.get_ylim())
                        ax.set_yticks(shap_ax.get_yticks())
                        ax.set_yticklabels(shap_ax.get_yticklabels(), fontsize=10)
                    else:
                        ax.text(0.5, 0.5, 'SHAP Plot Copy Failed\n\nCould not copy plot elements', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=11)
                
                plt.close(temp_fig) 
        except Exception as shap_err:
            plt.close(temp_fig)
            ax.text(0.5, 0.5, f'SHAP Plot Error:\n{str(shap_err)[:80]}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            print(f"SHAP plotting error: {shap_err}")
            import traceback
            traceback.print_exc()
        
        ax.set_title('SHAP Beeswarm: Feature Impact on Quality', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
    except ImportError:
        ax.text(0.5, 0.5, 'SHAP Library Not Installed\n\nRun: pip install shap', 
               ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title('SHAP Feature Impact Analysis', fontsize=12)
    except Exception as e:
        # Fallback: Create a simple bar plot of mean absolute SHAP values
        try:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            sorted_idx = np.argsort(mean_abs_shap)[::-1]
            sorted_feature_names = [feature_names[i] for i in sorted_idx]
            sorted_importances = mean_abs_shap[sorted_idx]
            
            ax.clear()
            colors = [plt.cm.PuRd(0.3 + 0.5 * (imp / sorted_importances.max())) 
                     for imp in sorted_importances]
            bars = ax.barh(range(len(sorted_feature_names)), sorted_importances, 
                          color=colors, edgecolor='black', linewidth=0.5)
            ax.set_yticks(range(len(sorted_feature_names)))
            ax.set_yticklabels(sorted_feature_names, fontsize=10)
            ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
            ax.set_title('SHAP Feature Impact (Fallback)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, sorted_importances)):
                ax.text(val + 0.01 * sorted_importances.max(), i, f'{val:.3f}', 
                       va='center', fontsize=9)
        except Exception as e2:
            ax.text(0.5, 0.5, f'SHAP Plot Error:\n{str(e)[:80]}\n\nFallback also failed:\n{str(e2)[:50]}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_title('SHAP Feature Impact Analysis', fontsize=12)
