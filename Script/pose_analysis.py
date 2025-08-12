#!/usr/bin/env python3
"""
Pose Analysis Module

This module provides functionality to compare pose estimation results from DeepLabCut (DLC)
and SLEAP against manually labeled ground truth data.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple, Set
from scipy import stats
import warnings
import h5py

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('ggplot')
sns.set_style("whitegrid")

# Define body parts for analysis
BODY_PARTS = [
    'Nose', 'Left_ear', 'Right_ear', 'Spine_1', 'Center', 'Spine_2',
    'Left_fhip', 'Right_fhip', 'Left_bhip', 'Right_bhip', 'Tail_base',
    'Tail_1', 'Tail_2', 'Tail_tip'
]

class PoseAnalyzer:
    """Class for comparing pose estimation results against manual labels."""
    
    def __init__(self, dlc_project_dir: str, sleap_dir: str, output_dir: str, use_timestamp: bool = True):
        """
        Initialize the PoseAnalyzer.
        
        Args:
            dlc_project_dir: Path to the DLC project directory
            sleap_dir: Path to the SLEAP export directory
            output_dir: Directory to save analysis results
            use_timestamp: Whether to create a timestamped subfolder for each run
        """
        self.dlc_project_dir = dlc_project_dir
        self.sleap_dir = sleap_dir
        self.output_dir = output_dir
        
        # Store paths to labeled data and videos
        self.labeled_data_dir = os.path.join(dlc_project_dir, 'labeled-data')
        self.dlc_videos_dir = os.path.join(dlc_project_dir, 'videos')
        
        # Create output directories with optional timestamp
        if use_timestamp:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(output_dir, f'analysis_{timestamp}')
            
        # Create output directories
        self.results_dir = os.path.join(self.output_dir, 'results')
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Create detailed differences directory
        self.detailed_dir = os.path.join(self.results_dir, 'detailed_differences')
        os.makedirs(self.detailed_dir, exist_ok=True)
        
        # Statistics and results storage
        self.results = {}
        self.detailed_stats = {}
        
        # Validate directories
        self._validate_directories()
        
        # Get list of videos with labeled data
        self.labeled_videos = self._find_labeled_videos()
        
        print(f"Initialized PoseAnalyzer with:")
        print(f"  DLC Project: {dlc_project_dir}")
        print(f"  SLEAP Dir: {sleap_dir}")
        print(f"  Output Dir: {output_dir}")
        print(f"  Found {len(self.labeled_videos)} videos with labeled data")
    
    def _validate_directories(self):
        """Validate that all required directories exist."""
        if not os.path.exists(self.dlc_videos_dir):
            raise FileNotFoundError(f"DLC videos directory not found: {self.dlc_videos_dir}")
            
        if not os.path.exists(self.labeled_data_dir):
            raise FileNotFoundError(f"Labeled data directory not found: {self.labeled_data_dir}")
            
        if not os.path.exists(self.sleap_dir):
            raise FileNotFoundError(f"SLEAP directory not found: {self.sleap_dir}")
            
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def _find_labeled_videos(self) -> Set[str]:
        """Find all videos that have labeled data."""
        labeled_videos = set()
        
        # Look for labeled data in the labeled-data directory
        if not os.path.exists(self.labeled_data_dir):
            print(f"Warning: Labeled data directory not found: {self.labeled_data_dir}")
            return labeled_videos
            
        for item in os.listdir(self.labeled_data_dir):
            item_path = os.path.join(self.labeled_data_dir, item)
            if os.path.isdir(item_path):
                # Check if this directory contains a CSV file with the same name
                csv_file = os.path.join(item_path, f"CollectedData_{item}.csv")
                if os.path.exists(csv_file):
                    labeled_videos.add(item)
                else:
                    # Also check for other possible CSV file patterns
                    for file in os.listdir(item_path):
                        if file.endswith('.csv') and 'CollectedData' in file:
                            labeled_videos.add(item)
                            break
        
        print(f"Found {len(labeled_videos)} videos with labeled data")
        return labeled_videos
    
    def _find_sleap_file(self, video_name: str) -> Optional[str]:
        """Find the SLEAP analysis file for a given video."""
        # Try different patterns to match the SLEAP file
        patterns = [
            f"*{video_name}*.analysis.csv",  # Exact match first
            f"*{video_name.replace('_', '.')}*.analysis.csv",  # Try with dots instead of underscores
            f"*{video_name.split('_clip')[0]}*.analysis.csv",  # Try without the clip suffix
            "*.analysis.csv"  # Fallback to any analysis file
        ]
        
        for pattern in patterns:
            files = glob.glob(os.path.join(self.sleap_dir, pattern))
            if files:
                return files[0]  # Return the first match
                
        return None
        
    def _find_dlc_file(self, video_name: str) -> Optional[str]:
        """Find the DLC analysis file for a given video."""
        # Try different patterns to match the DLC file
        patterns = [
            f"*{video_name}DLC_*.h5",  # Match files like 'video_nameDLC_...'
            f"*{video_name}*.h5",       # Match any h5 file with video name
            "*.h5"                      # Fallback to any h5 file
        ]
        
        for pattern in patterns:
            files = glob.glob(os.path.join(self.dlc_videos_dir, pattern))
            if files:
                return files[0]  # Return the first match
                
        print(f"Files available: {os.listdir(self.sleap_dir)}")
        return None
    
    def load_manual_labels(self, video_name: str) -> pd.DataFrame:
        """
        Load manual labels for a video in the format used by the CIN notebook.
        
        Args:
            video_name: Name of the video to load labels for
            
        Returns:
            DataFrame with columns for each body part's x, y coordinates,
            indexed by frame number extracted from image filenames (img0001.png -> 1)
        """
        print(f"  Loading manual labels for {video_name}...")
        
        # Look for CSV files in the labeled data directory
        video_dir = os.path.join(self.labeled_data_dir, video_name)
        if not os.path.exists(video_dir):
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
            
        # Look for the CSV file (CollectedData_*.csv or similar)
        csv_file = os.path.join(video_dir, "CollectedData_MAS.csv")
        if not os.path.exists(csv_file):
            csv_files = glob.glob(os.path.join(video_dir, "*CollectedData*.csv"))
            if not csv_files:
                # Try alternative naming convention
                csv_files = glob.glob(os.path.join(video_dir, "*.csv"))
                if not csv_files:
                    raise FileNotFoundError(f"No CSV file found in {video_dir}")
            csv_file = csv_files[0]
            
        print(f"  Found label file: {os.path.basename(csv_file)}")
        
        try:
            # Read the CSV file, handling the format from the notebook
            df = pd.read_csv(csv_file, header=None)
            
            # Remove the first 3 rows (header info in DLC format)
            df = df.drop([0, 1, 2])
            
            # Remove the first two columns (scorer and bodyparts)
            df = df.drop(columns=[0, 1])
            
            # Set column names based on the standard body parts
            body_parts = [
                'Nose', 'Left_ear', 'Right_ear', 'Spine_1', 'Center', 'Spine_2',
                'Left_fhip', 'Right_fhip', 'Left_bhip', 'Right_bhip', 'Tail_base',
                'Tail_1', 'Tail_2', 'Tail_tip'
            ]
            
            # Create multi-index columns for x and y coordinates
            columns = []
            for part in body_parts:
                columns.append(f"{part}.x")
                columns.append(f"{part}.y")
            
            # If we have more columns than expected, add them as extra parts
            extra_cols = len(df.columns) - len(columns)
            if extra_cols > 0:
                for i in range(extra_cols // 2):
                    part_name = f"extra_{i+1}"
                    columns.append(f"{part_name}.x")
                    columns.append(f"{part_name}.y")
            
            # Set the column names, handling mismatches
            if len(df.columns) == len(columns):
                df.columns = columns
            else:
                print(f"  Warning: Expected {len(columns)} columns but found {len(df.columns)}. Adjusting column names...")
                # Use the standard columns we have, and number any extra columns
                all_columns = columns.copy()
                extra_cols = len(df.columns) - len(columns)
                if extra_cols > 0:
                    for i in range(extra_cols):
                        all_columns.append(f"extra_{i+1}")
                df.columns = all_columns[:len(df.columns)]
            
            # Add frame numbers from the image filenames (stored in the index)
            frame_numbers = []
            for img_name in df.index:
                frame_num = self._extract_frame_number(str(img_name))
                frame_numbers.append(frame_num)
            
            df['Frame'] = frame_numbers
            df = df.set_index('Frame')
            
            # Convert all columns to numeric, coercing errors to NaN
            df = df.apply(pd.to_numeric, errors='coerce')
            
            print(f"  Loaded {len(df)} labeled frames")
            return df
            
        except Exception as e:
            print(f"  Error loading manual labels: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _extract_frame_number(self, img_name: str) -> int:
        """
        Extract frame number from image filename.
        
        Args:
            img_name: Image filename (e.g., 'img0001.png' or 'frame_0042.jpg')
            
        Returns:
            Extracted frame number as integer, or 0 if parsing fails
        """
        if not isinstance(img_name, str):
            return 0
            
        # Try different patterns to extract frame number
        import re
        
        # Pattern for 'img0001.png' or 'frame_0042.jpg'
        match = re.search(r'(?:img|frame[_-]?)(\d+)\.', img_name)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                pass
                
        # If no pattern matched, try to extract any sequence of digits
        digits = re.findall(r'\d+', img_name)
        if digits:
            try:
                return int(digits[0])
            except (ValueError, IndexError):
                pass
                
        return 0  # Default to 0 if we can't parse the frame number
        
    def load_dlc_predictions(self, video_name: str) -> pd.DataFrame:
        """
        Load and process DLC predictions from H5 file.
        
        Args:
            video_name: Name of the video to load predictions for
            
        Returns:
            DataFrame with columns for each body part's x, y coordinates and likelihood,
            or None if no predictions found or an error occurred
        """
        # Find the DLC analysis file
        dlc_file = self._find_dlc_file(video_name)
        if not dlc_file:
            print(f"  Warning: No DLC predictions found for video: {video_name}")
            return None
            
        print(f"  Loading DLC file: {os.path.basename(dlc_file)}")
        
        try:
            with h5py.File(dlc_file, 'r') as f:
                # The data is stored in a pandas-compatible format
                if 'df_with_missing' not in f or 'table' not in f['df_with_missing']:
                    print(f"  Error: Invalid DLC H5 file format - missing 'df_with_missing/table'")
                    return None
                
                # Get the data as a structured numpy array
                data = f['df_with_missing/table'][:]
                
                # Get the body part names from the column metadata
                body_parts = []
                if 'values_block_0' in data.dtype.names:
                    # The number of body parts is the number of columns divided by 3 (x, y, likelihood)
                    num_parts = data['values_block_0'].shape[1] // 3
                    
                    # Generate generic body part names if we can't get them from the file
                    body_parts = [f"bodypart_{i+1}" for i in range(num_parts)]
                    
                    # Try to get actual body part names from the attributes if available
                    if 'values_block_0' in f['df_with_missing/table'].attrs:
                        attrs = f['df_with_missing/table'].attrs['values_block_0']
                        if 'items' in attrs:
                            # The items attribute contains the column names
                            cols = attrs['items']
                            # Extract unique body part names (every 3rd column starting from 0 is x, 1 is y, 2 is likelihood)
                            body_parts = [cols[i].decode('utf-8').split('.')[0] for i in range(0, len(cols), 3)]
                
                print(f"  Found {len(body_parts)} body parts in {len(data)} frames")
                
                # Map from DLC body part names to standard body part names
                # This is a simple mapping - you may need to adjust this based on your specific DLC project
                body_part_mapping = {
                    'bodypart_1': 'Nose',
                    'bodypart_2': 'Left_ear',
                    'bodypart_3': 'Right_ear',
                    'bodypart_4': 'Spine_1',
                    'bodypart_5': 'Center',
                    'bodypart_6': 'Spine_2',
                    'bodypart_7': 'Left_fhip',
                    'bodypart_8': 'Right_fhip',
                    'bodypart_9': 'Left_bhip',
                    'bodypart_10': 'Right_bhip',
                    'bodypart_11': 'Tail_base',
                    'bodypart_12': 'Tail_1',
                    'bodypart_13': 'Tail_2',
                    'bodypart_14': 'Tail_tip'
                }
                
                # Update body part names to standard names if they match our mapping
                for i, part in enumerate(body_parts):
                    if part in body_part_mapping:
                        body_parts[i] = body_part_mapping[part]
                
                # Create a DataFrame with the predictions
                rows = []
                for i, row in enumerate(data):
                    frame_data = {'Frame': i}
                    
                    # Get the coordinates for each body part
                    if 'values_block_0' in row.dtype.names:
                        coords = row['values_block_0']
                        for part_idx in range(len(body_parts)):
                            x = coords[part_idx * 3]     # x coordinate
                            y = coords[part_idx * 3 + 1]  # y coordinate
                            likelihood = coords[part_idx * 3 + 2]  # likelihood
                            
                            # Only include points with reasonable confidence (likelihood > 0.5)
                            if likelihood > 0.5:
                                frame_data[f"{body_parts[part_idx]}.x"] = x
                                frame_data[f"{body_parts[part_idx]}.y"] = y
                    
                    rows.append(frame_data)
                
                # Create DataFrame and set index
                df = pd.DataFrame(rows).set_index('Frame')
                print(f"  Successfully loaded {len(df)} frames with {len(body_parts)} body parts")
                
                # Print the first few rows for debugging
                print("  First few rows of DLC data:")
                print(df.head())
                
                return df
                
        except Exception as e:
            print(f"  Error loading DLC predictions: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_sleap_predictions(self, video_name: str) -> pd.DataFrame:
        """Load and process SLEAP predictions."""
        # Find the SLEAP analysis file
        sleap_file = self._find_sleap_file(video_name)
        if not sleap_file:
            raise FileNotFoundError(f"No SLEAP analysis file found for video: {video_name}")
        
        print(f"  Loading SLEAP file: {os.path.basename(sleap_file)}")
        
        # Read the SLEAP CSV file
        df = pd.read_csv(sleap_file)
        
        # Initialize a DataFrame with frame numbers as index
        result = pd.DataFrame(index=df['frame_idx'].unique())
        result.index.name = 'Frame'
        
        # The SLEAP CSV already has columns in the format we need (BodyPart.x, BodyPart.y)
        # We just need to select the right columns and group by frame
        for part in BODY_PARTS:
            x_col = f"{part}.x"
            y_col = f"{part}.y"
            
            if x_col in df.columns and y_col in df.columns:
                # Group by frame and take the first instance (assuming single animal tracking)
                x_coords = df.groupby('frame_idx')[x_col].first()
                y_coords = df.groupby('frame_idx')[y_col].first()
                
                # Add to result DataFrame
                result[x_col] = x_coords
                result[y_col] = y_coords
        
        print(f"  Loaded {len(result)} frames with {len(BODY_PARTS)} body parts each")
        return result
    
    def calculate_distances(self, ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Calculate Euclidean distances between ground truth and predicted points.
        Only compares frames that exist in the ground truth data.
        
        Args:
            ground_truth: DataFrame with ground truth coordinates (columns: 'part.x' and 'part.y')
            predictions: DataFrame with predicted coordinates (columns: 'part.x' and 'part.y')
            
        Returns:
            Dictionary with detailed distance statistics for each body part
        """
        results = {}
        
        # Get list of frames with ground truth data
        ground_truth_frames = set(ground_truth.index)
        
        # Filter predictions to only include frames present in ground truth
        common_frames = predictions.index.intersection(ground_truth_frames)
        
        if len(common_frames) == 0:
            print("  Warning: No common frames found between ground truth and predictions")
            return results
        
        print(f"  Calculating distances for {len(common_frames)} labeled frames...")
        
        # Get list of body parts from ground truth columns (format: 'part.x' and 'part.y')
        body_parts = set()
        for col in ground_truth.columns:
            if '.' in col and col.endswith('.x'):
                body_parts.add(col[:-2])  # Remove '.x' suffix
        
        # Calculate ear distance for normalization (as done in the notebook)
        ear_distances = []
        for frame in common_frames:
            try:
                # Get left and right ear coordinates from ground truth
                left_ear_x = ground_truth.at[frame, 'Left_ear.x']
                left_ear_y = ground_truth.at[frame, 'Left_ear.y']
                right_ear_x = ground_truth.at[frame, 'Right_ear.x']
                right_ear_y = ground_truth.at[frame, 'Right_ear.y']
                
                # Calculate Euclidean distance between ears
                if (not np.isnan(left_ear_x) and not np.isnan(left_ear_y) and 
                    not np.isnan(right_ear_x) and not np.isnan(right_ear_y)):
                    dist = np.sqrt((left_ear_x - right_ear_x)**2 + (left_ear_y - right_ear_y)**2)
                    ear_distances.append(dist)
            except KeyError:
                # Ear coordinates not available, skip this frame
                continue
        
        # Calculate the average of the top 5 ear distances (as done in the notebook)
        if ear_distances:
            ear_distances_sorted = sorted(ear_distances, reverse=True)
            avg_ear_distance = np.mean(ear_distances_sorted[:min(5, len(ear_distances_sorted))])
            print(f"  Average ear distance (top 5 frames): {avg_ear_distance:.2f} px")
        else:
            avg_ear_distance = 100.0  # Default value if ear distances can't be calculated
            print("  Warning: Could not calculate ear distance, using default value")
        
        # For each body part, calculate distances for common frames
        for part in body_parts:
            x_col = f"{part}.x"
            y_col = f"{part}.y"
            
            # Only proceed if both x and y columns exist in both DataFrames
            if (x_col in ground_truth.columns and y_col in ground_truth.columns and 
                x_col in predictions.columns and y_col in predictions.columns):
                
                # Initialize lists to store valid points
                distances = []
                normalized_distances = []
                
                # Calculate distance for each frame that exists in both ground truth and predictions
                for frame in common_frames:
                    try:
                        # Get ground truth coordinates
                        gt_x = ground_truth.at[frame, x_col]
                        gt_y = ground_truth.at[frame, y_col]
                        
                        # Get predicted coordinates
                        pred_x = predictions.at[frame, x_col]
                        pred_y = predictions.at[frame, y_col]
                        
                        # Only calculate distance if all coordinates are valid numbers
                        if (not np.isnan(gt_x) and not np.isnan(gt_y) and 
                            not np.isnan(pred_x) and not np.isnan(pred_y)):
                            
                            # Calculate Euclidean distance
                            dist = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
                            distances.append(dist)
                            
                            # Calculate normalized distance (divided by ear distance)
                            if avg_ear_distance > 0:
                                normalized_dist = dist / avg_ear_distance
                                normalized_distances.append(normalized_dist)
                    except Exception as e:
                        print(f"    Error processing frame {frame} for {part}: {str(e)}")
                        continue
                
                # Calculate statistics if we have valid distances
                if distances:
                    dist_array = np.array(distances)
                    results[part] = {
                        'mean': float(np.mean(dist_array)),
                        'median': float(np.median(dist_array)),
                        'std': float(np.std(dist_array, ddof=1)),
                        'min': float(np.min(dist_array)),
                        'max': float(np.max(dist_array)),
                        'count': int(len(dist_array)),
                        'distances': dist_array.tolist(),
                        'normalized_distances': normalized_distances if normalized_distances else None
                    }
                    
                    # Add normalized error metrics if available
                    if normalized_distances:
                        norm_array = np.array(normalized_distances)
                        results[part].update({
                            'mean_normalized': float(np.mean(norm_array)),
                            'median_normalized': float(np.median(norm_array)),
                            'std_normalized': float(np.std(norm_array, ddof=1)),
                        })
                    
                    print(f"    {part}: {results[part]['mean']:.2f} ± {results[part]['std']:.2f} px "
                          f"(n={results[part]['count']}, norm={results[part].get('mean_normalized', 'N/A'):.3f})")
                else:
                    results[part] = {
                        'mean': float('nan'),
                        'median': float('nan'),
                        'std': float('nan'),
                        'min': float('nan'),
                        'max': float('nan'),
                        'count': 0,
                        'distances': [],
                        'normalized_distances': []
                    }
                    print(f"    {part}: No valid data points")
            else:
                print(f"    {part}: Missing coordinate columns")
        
        return results
    
    def plot_error_distribution(self, distances: Dict[str, Any], video_name: str, model_name: str):
        """Plot the distribution of errors for each body part."""
        plt.figure(figsize=(12, 8))
        
        # Prepare data for plotting
        data = []
        for part, stats in distances.items():
            if 'distances' in stats and len(stats['distances']) > 0:
                for dist in stats['distances']:
                    data.append({'Body Part': part, 'Distance (pixels)': dist})
        
        if not data:
            print("  No valid distance data available for plotting")
            return
            
        df = pd.DataFrame(data)
        
        # Create violin plot
        plt.figure(figsize=(14, 8))
        sns.violinplot(x='Body Part', y='Distance (pixels)', data=df, inner='quartile')
        plt.title(f'Error Distribution by Body Part\n{video_name} - {model_name}', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(self.plots_dir, f"{video_name}_{model_name}_error_distribution.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_file
    
    def plot_frame_by_frame_errors(self, ground_truth: pd.DataFrame, predictions: pd.DataFrame, 
                                 video_name: str, model_name: str):
        """Plot frame-by-frame errors for each body part."""
        # Make sure we only compare frames that exist in both DataFrames
        common_frames = ground_truth.index.intersection(predictions.index)
        if len(common_frames) == 0:
            return None
            
        gt_common = ground_truth.loc[common_frames]
        pred_common = predictions.loc[common_frames]
        
        # Create a figure with subplots for each body part
        n_parts = len(BODY_PARTS)
        n_cols = 3
        n_rows = (n_parts + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        if n_parts > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for i, part in enumerate(BODY_PARTS):
            x_col, y_col = f"{part}.x", f"{part}.y"
            if x_col in gt_common.columns and x_col in pred_common.columns:
                # Calculate distances for each frame
                gt_x = gt_common[x_col].astype(float)
                gt_y = gt_common[y_col].astype(float)
                pred_x = pred_common[x_col].astype(float)
                pred_y = pred_common[y_col].astype(float)
                
                dists = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
                
                # Plot
                ax = axes[i]
                ax.plot(common_frames, dists, label=part, linewidth=1)
                ax.set_title(part)
                ax.set_xlabel('Frame')
                ax.set_ylabel('Error (pixels)')
                ax.grid(True, alpha=0.3)
                
                # Add mean error to the plot
                mean_error = np.nanmean(dists)
                if not np.isnan(mean_error):
                    ax.axhline(mean_error, color='r', linestyle='--', 
                              label=f'Mean: {mean_error:.2f}px')
                    ax.legend()
        
        # Remove any extra subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.suptitle(f'Frame-by-Frame Errors\n{video_name} - {model_name}', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(self.plots_dir, f"{video_name}_{model_name}_frame_errors.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_file
        
    def plot_model_comparison(self, dlc_distances: Dict[str, Any], 
                            sleap_distances: Dict[str, Any], 
                            video_name: str) -> str:
        """
        Create comparison plots between DLC and SLEAP results.
        
        Args:
            dlc_distances: Dictionary of DLC distance metrics per body part
            sleap_distances: Dictionary of SLEAP distance metrics per body part
            video_name: Name of the video being analyzed
            
        Returns:
            Path to the saved comparison plot
        """
        # Prepare data for comparison
        comparison_data = []
        
        # Get common body parts
        common_parts = set(dlc_distances.keys()).intersection(set(sleap_distances.keys()))
        
        if not common_parts:
            print("  No common body parts found for model comparison")
            return ""
        
        # Collect mean errors for each model and body part
        for part in common_parts:
            if 'mean' in dlc_distances[part] and 'mean' in sleap_distances[part]:
                comparison_data.append({
                    'Body Part': part,
                    'DLC': dlc_distances[part]['mean'],
                    'SLEAP': sleap_distances[part]['mean']
                })
        
        if not comparison_data:
            print("  No valid data for model comparison")
            return ""
        
        df = pd.DataFrame(comparison_data)
        
        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. Bar plot comparing mean errors
        df_melted = df.melt(id_vars=['Body Part'], 
                           var_name='Model', 
                           value_name='Mean Error (pixels)')
        
        sns.barplot(data=df_melted, x='Body Part', y='Mean Error (pixels)', 
                   hue='Model', ax=ax1, palette='viridis')
        ax1.set_title('Mean Error by Body Part and Model')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # 2. Scatter plot of DLC vs SLEAP errors
        ax2.scatter(df['DLC'], df['SLEAP'], alpha=0.6, edgecolors='w')
        
        # Add identity line for reference
        max_val = max(df[['DLC', 'SLEAP']].max().max() * 1.1, 1.0)
        ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        
        # Add labels and title
        ax2.set_xlabel('DLC Error (pixels)')
        ax2.set_ylabel('SLEAP Error (pixels)')
        ax2.set_title('DLC vs SLEAP Error Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        r = np.corrcoef(df['DLC'], df['SLEAP'])[0, 1]
        ax2.text(0.05, 0.95, f'r = {r:.2f}', 
                transform=ax2.transAxes, 
                bbox=dict(facecolor='white', alpha=0.7))
        
        # Add body part labels to points
        for i, row in df.iterrows():
            ax2.text(row['DLC'], row['SLEAP'], row['Body Part'], 
                   fontsize=8, alpha=0.7)
        
        # Adjust layout and save
        plt.suptitle(f'Model Comparison\n{video_name}', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(self.plots_dir, f"{video_name}_model_comparison.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved model comparison plot: {os.path.basename(plot_file)}")
        return plot_file
    
    def analyze_video(self, video_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Run analysis for a single video, following the CIN notebook's approach.
        
        Returns:
            Dictionary containing detailed results for each model (DLC and SLEAP)
        """
        print(f"\nAnalyzing video: {video_name}")
        results = {}
        
        try:
            # Load ground truth - this now uses the notebook's format
            print("  Loading manual labels...")
            ground_truth = self.load_manual_labels(video_name)
            
            # Dictionary to store predictions from different models
            predictions = {}
            
            # Try to load and process DLC predictions
            try:
                print("  Loading DLC predictions...")
                dlc_pred = self.load_dlc_predictions(video_name)
                if dlc_pred is not None:
                    predictions['DLC'] = dlc_pred
                    print(f"    Loaded DLC predictions for {len(dlc_pred)} frames")
                else:
                    print("    No DLC predictions found")
            except Exception as e:
                print(f"  Warning: Could not load DLC predictions: {str(e)}")
            
            # Try to load and process SLEAP predictions
            try:
                print("  Loading SLEAP predictions...")
                sleap_pred = self.load_sleap_predictions(video_name)
                if sleap_pred is not None:
                    predictions['SLEAP'] = sleap_pred
                    print(f"    Loaded SLEAP predictions for {len(sleap_pred)} frames")
                else:
                    print("    No SLEAP predictions found")
            except Exception as e:
                print(f"  Warning: Could not load SLEAP predictions: {str(e)}")
            
            if not predictions:
                print("  Error: No prediction data available for any model")
                return {}
            
            # Calculate distances and statistics for each model
            print("  Calculating distances and statistics...")
            for model_name, pred in predictions.items():
                try:
                    print(f"  Processing {model_name}...")
                    distances = self.calculate_distances(ground_truth, pred)
                    if distances:  # Only add if we got valid distances
                        results[model_name] = distances
                        
                        # Calculate overall statistics for this model
                        all_distances = []
                        all_normalized = []
                        
                        for part, stats in distances.items():
                            if 'distances' in stats and stats['distances']:
                                all_distances.extend(stats['distances'])
                            if 'normalized_distances' in stats and stats['normalized_distances']:
                                all_normalized.extend(stats['normalized_distances'])
                        
                        if all_distances:
                            all_distances = np.array(all_distances)
                            results[model_name]['_overall'] = {
                                'mean': float(np.mean(all_distances)),
                                'median': float(np.median(all_distances)),
                                'std': float(np.std(all_distances, ddof=1)),
                                'min': float(np.min(all_distances)),
                                'max': float(np.max(all_distances)),
                                'count': int(len(all_distances))
                            }
                            
                            if all_normalized:
                                all_normalized = np.array(all_normalized)
                                results[model_name]['_overall'].update({
                                    'mean_normalized': float(np.mean(all_normalized)),
                                    'median_normalized': float(np.median(all_normalized)),
                                    'std_normalized': float(np.std(all_normalized, ddof=1))
                                })
                            
                            print(f"  {model_name} overall error: "
                                  f"{results[model_name]['_overall']['mean']:.2f} ± "
                                  f"{results[model_name]['_overall']['std']:.2f} px")
                except Exception as e:
                    print(f"  Error calculating distances for {model_name}: {str(e)}")
            
            if not results:
                print("  Error: Could not calculate distances for any model")
                return {}
            
            # Save detailed differences for each model
            print("  Saving detailed differences...")
            self._save_detailed_differences(video_name, ground_truth, predictions)
            
            # Generate visualizations
            print("  Generating visualizations...")
            for model_name, distances in results.items():
                if distances:  # Only if we have valid distances
                    # Plot error distribution for each model
                    self.plot_error_distribution(distances, video_name, model_name)
                    
                    # Plot frame-by-frame errors for each model
                    self.plot_frame_by_frame_errors(
                        ground_truth, predictions[model_name], video_name, model_name
                    )
            
            # If we have both DLC and SLEAP results, create comparison plots
            if 'DLC' in results and 'SLEAP' in results:
                self.plot_model_comparison(
                    results['DLC'], 
                    results['SLEAP'], 
                    video_name
                )
            
            # Print summary statistics
            for model_name, distances in results.items():
                if distances:
                    mean_errors = [stats['mean'] for part, stats in distances.items() 
                                 if 'mean' in stats and not np.isnan(stats['mean'])]
                    if mean_errors:
                        print(f"  {model_name} - Average error: {np.mean(mean_errors):.2f} ± {np.std(mean_errors, ddof=1):.2f} pixels")
            
            return results
            
        except Exception as e:
            print(f"Error analyzing {video_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_analysis(self, video_list: List[str] = None):
        """
        Run analysis on a list of videos.
        
        Args:
            video_list: List of video names to analyze. If None, processes all labeled videos.
        """
        if video_list is None:
            # Use all labeled videos if none specified
            video_list = sorted(self.labeled_videos)
        else:
            # Filter to only include videos that have labeled data
            video_list = [v for v in video_list if v in self.labeled_videos]
        
        if not video_list:
            print("No videos with labeled data found to analyze.")
            return {}
        
        print(f"\nFound {len(video_list)} videos with labeled data:")
        for video in video_list:
            print(f"  - {video}")
        
        all_results = {}
        for video in video_list:
            results = self.analyze_video(video)
            if results:
                all_results[video] = results
        
        # Save results
        if all_results:
            self.save_results(all_results)
        else:
            print("No results to save.")
            
        return all_results
    
    def save_results(self, results: Dict, filename: str = 'pose_analysis_results.json') -> None:
        """
        Save analysis results to JSON and CSV files, following the CIN notebook's format.
        
        Args:
            results: Dictionary containing analysis results
            filename: Base filename for output files (without extension)
        """
        if not results:
            print("  No results to save")
            return
            
        # Ensure the results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Save as JSON
        json_path = os.path.join(self.results_dir, filename)
        
        # Prepare results in the format expected by the notebook
        notebook_results = {}
        for video_name, video_data in results.items():
            notebook_results[video_name] = {}
            
            for model_name, model_results in video_data.items():
                notebook_results[video_name][model_name] = {}
                
                # Add overall statistics
                if '_overall' in model_results:
                    notebook_results[video_name][model_name]['overall'] = model_results['_overall']
                
                # Add per-body-part statistics
                for part, stats in model_results.items():
                    if part.startswith('_'):
                        continue  # Skip internal keys like '_overall'
                        
                    notebook_results[video_name][model_name][part] = {
                        'mean': stats.get('mean', float('nan')),
                        'std': stats.get('std', float('nan')),
                        'median': stats.get('median', float('nan')),
                        'min': stats.get('min', float('nan')),
                        'max': stats.get('max', float('nan')),
                        'count': stats.get('count', 0),
                        'mean_normalized': stats.get('mean_normalized', float('nan')),
                        'std_normalized': stats.get('std_normalized', float('nan'))
                    }
        
        # Save the results
        with open(json_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(x) for x in obj]
                else:
                    return obj
            
            json.dump(convert_to_serializable(notebook_results), f, indent=2)
        print(f"  Saved results to {json_path}")
        
        # Also save a CSV summary in the format expected by the notebook
        self._save_results_csv(notebook_results, filename.replace('.json', '.csv'))
    
    def _save_results_csv(self, results: Dict, filename: str) -> None:
        """
        Save a summary of results to a CSV file with comprehensive statistics.
        
        Args:
            results: Dictionary containing analysis results
            filename: Output CSV filename (without path)
        """
        if not results:
            print("  No results to save to CSV")
            return
            
        csv_path = os.path.join(self.results_dir, filename)
        
        # Initialize DataFrames for statistics
        stats_df = pd.DataFrame(columns=['video', 'body_part', 'mean_error', 'std_error', 'n_frames', 'model'])
        model_stats = []
        
        # Process each video and model
        for video_name, video_data in results.items():
            if not isinstance(video_data, dict):
                continue
                
            for model_name, model_data in video_data.items():
                if not isinstance(model_data, dict):
                    continue
                
                # Process body part data
                for part_name, part_data in model_data.items():
                    # Skip non-body part entries
                    if part_name in ['overall', 'mean', 'std', 'median', 'min', 'max', 'mean_normalized', 'std_normalized']:
                        continue
                        
                    if isinstance(part_data, dict):
                        # Handle dictionary format
                        mean_val = part_data.get('mean')
                        std_val = part_data.get('std')
                        n_frames = part_data.get('n', 1)
                        
                        if mean_val is not None and not np.isnan(mean_val):
                            stats_df = pd.concat([
                                stats_df,
                                pd.DataFrame([{
                                    'video': video_name,
                                    'body_part': part_name,
                                    'mean_error': float(mean_val),
                                    'std_error': float(std_val) if std_val is not None else np.nan,
                                    'n_frames': int(n_frames) if n_frames is not None else 1,
                                    'model': model_name
                                }])
                            ], ignore_index=True)
                    
                    # Also check for direct numeric values in the model data
                    elif isinstance(part_data, (int, float)) and not np.isnan(part_data):
                        stats_df = pd.concat([
                            stats_df,
                            pd.DataFrame([{
                                'video': video_name,
                                'body_part': part_name,
                                'mean_error': float(part_data),
                                'std_error': np.nan,
                                'n_frames': 1,
                                'model': model_name
                            }])
                        ], ignore_index=True)
                
                # Extract overall model statistics if available
                if 'overall' in model_data and isinstance(model_data['overall'], dict):
                    overall = model_data['overall']
                    model_stats.append({
                        'video': video_name,
                        'model': model_name,
                        'mean_error': overall.get('mean', np.nan),
                        'std_error': overall.get('std', np.nan),
                        'n_frames': overall.get('n', 0)
                    })
        
        # If no valid data, exit early
        if stats_df.empty and not model_stats:
            print("  No valid results to save to CSV")
            return
            
        # Generate summary statistics
        summary_rows = []
        
        # Add overall model statistics
        for stat in model_stats:
            # Count unique body parts for this video and model
            if not stats_df.empty:
                body_part_count = len(stats_df[
                    (stats_df['video'] == stat['video']) & 
                    (stats_df['model'] == stat['model'])
                ].body_part.unique())
            else:
                body_part_count = 0
                
            summary_rows.append({
                'Video': stat['video'],
                'Model': stat['model'],
                'Type': 'Overall',
                'Body Part': 'All',
                'Mean Error (px)': f"{stat['mean_error']:.2f} ± {stat['std_error']:.2f}",
                'Frames': int(stat['n_frames']),
                'Body Parts': body_part_count
            })
        
        # Add per-body-part statistics
        if not stats_df.empty:
            # Group by video, model, and body part to get mean stats
            part_stats = stats_df.groupby(['video', 'model', 'body_part']).agg({
                'mean_error': 'mean',
                'std_error': 'mean',
                'n_frames': 'sum'
            }).reset_index()
            
            for _, row in part_stats.iterrows():
                summary_rows.append({
                    'Video': row['video'],
                    'Model': row['model'],
                    'Type': 'By Body Part',
                    'Body Part': row['body_part'],
                    'Mean Error (px)': f"{row['mean_error']:.2f} ± {row['std_error']:.2f}",
                    'Frames': int(row['n_frames']),
                    'Body Parts': 1
                })
        
        # Calculate overall statistics for the summary
        if not stats_df.empty:
            overall_stats = stats_df.groupby('model').agg({
                'mean_error': ['mean', 'std', 'count'],
                'n_frames': 'sum'
            }).reset_index()
        else:
            # Create empty stats if no body part data
            overall_stats = pd.DataFrame(columns=['model', 'mean_error', 'n_frames'])
        
        # Create a summary row for each model
        for _, row in overall_stats.iterrows():
            model = row[('model', '')]
            summary_rows.append({
                'Model': model,
                'Type': 'Overall',
                'Body Part': 'All',
                'Mean Error (px)': f"{row[('mean_error', 'mean')]:.2f} ± {row[('mean_error', 'std')]:.2f}",
                'Frames': int(row[('n_frames', 'sum')]),
                'Body Parts': int(row[('mean_error', 'count')])
            })
        
        # Add body part specific statistics if we have data
        if not stats_df.empty:
            # Get unique body parts from the stats DataFrame
            unique_body_parts = stats_df['body_part'].unique()
            
            for part in unique_body_parts:
                part_stats = stats_df[stats_df['body_part'] == part].groupby('model').agg({
                    'mean_error': ['mean', 'std'],
                    'n_frames': 'sum'
                }).reset_index()
                
                for _, row in part_stats.iterrows():
                    model = row[('model', '')]
                    summary_rows.append({
                        'Model': model,
                        'Type': 'By Body Part',
                        'Body Part': part,
                        'Mean Error (px)': f"{row[('mean_error', 'mean')]:.2f} ± {row[('mean_error', 'std')]:.2f}",
                        'Frames': int(row[('n_frames', 'sum')]),
                        'Body Parts': 1
                    })
        
        # Create and save the summary DataFrame
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(csv_path, index=False)
        
        # Print comprehensive summary to console
        print("\n" + "="*70)
        print("POSE ESTIMATION ANALYSIS SUMMARY")
        print("="*70)
        
        # Print overall statistics
        print("\nOVERALL PERFORMANCE")
        print("-" * 50)
        print(overall_stats[['model', 'mean_error']].rename(columns={
            'model': 'Model',
            ('mean_error', 'mean'): 'Mean Error (px)',
            ('mean_error', 'std'): 'Std Dev (px)',
            ('mean_error', 'count'): 'Body Parts',
            ('n_frames', 'sum'): 'Total Frames'
        }).to_string(index=False, float_format='%.2f'))
        
        # Print best and worst performing body parts
        if not stats_df.empty:
            # Best performing body parts (lowest mean error)
            best_parts = stats_df.loc[stats_df.groupby('body_part')['mean_error'].idxmin()]\
                               .sort_values('mean_error')\
                               .head(5)
            
            print("\nBEST PERFORMING BODY PARTS (LOWEST ERROR)")
            print("-" * 50)
            print(best_parts[['body_part', 'mean_error', 'std_error', 'model']]\
                  .rename(columns={
                      'body_part': 'Body Part',
                      'mean_error': 'Mean Error (px)',
                      'std_error': 'Std Dev (px)',
                      'model': 'Model'
                  })\
                  .to_string(index=False, float_format='%.2f'))
            
            # Worst performing body parts (highest mean error)
            worst_parts = stats_df.loc[stats_df.groupby('body_part')['mean_error'].idxmax()]\
                                .sort_values('mean_error', ascending=False)\
                                .head(5)
            
            print("\nWORST PERFORMING BODY PARTS (HIGHEST ERROR)")
            print("-" * 50)
            print(worst_parts[['body_part', 'mean_error', 'std_error', 'model']]\
                  .rename(columns={
                      'body_part': 'Body Part',
                      'mean_error': 'Mean Error (px)',
                      'std_error': 'Std Dev (px)',
                      'model': 'Model'
                  })\
                  .to_string(index=False, float_format='%.2f'))
            
            # Print model comparison
            print("\nMODEL COMPARISON")
            print("-" * 50)
            model_comparison = stats_df.groupby('model').agg({
                'mean_error': ['mean', 'std', 'count'],
                'n_frames': 'sum'
            })
            
            print(model_comparison.rename(columns={
                'mean': 'Mean Error (px)',
                'std': 'Std Dev (px)',
                'count': 'Body Parts',
                'sum': 'Total Frames'
            }).to_string(float_format='%.2f'))
        
        print("\n" + "="*70)
        print(f"Detailed results saved to: {csv_path}")
        print("="*70)
    
    def _save_detailed_differences(self, video_name: str, ground_truth: pd.DataFrame, 
                                 predictions: Dict[str, pd.DataFrame]) -> None:
        """
        Save detailed frame-by-frame differences for each body part and model.
        
        Args:
            video_name: Name of the video being analyzed
            ground_truth: DataFrame with ground truth coordinates
            predictions: Dictionary of model predictions with model names as keys
        """
        if not ground_truth.empty and predictions:
            # Ensure the detailed directory exists (should be created in __init__)
            os.makedirs(self.detailed_dir, exist_ok=True)
            
            for model_name, preds in predictions.items():
                if preds is None or preds.empty:
                    continue
                    
                # Get common frames between ground truth and predictions
                common_frames = ground_truth.index.intersection(preds.index)
                if len(common_frames) == 0:
                    continue
                    
                # Create a list to store all differences
                all_differences = []
                
                # Calculate differences for each body part
                for part in [col.split('.')[0] for col in ground_truth.columns if '.x' in col]:
                    x_col = f"{part}.x"
                    y_col = f"{part}.y"
                    
                    if x_col in ground_truth.columns and y_col in ground_truth.columns and \
                       x_col in preds.columns and y_col in preds.columns:
                        
                        for frame in common_frames:
                            gt_x = ground_truth.at[frame, x_col]
                            gt_y = ground_truth.at[frame, y_col]
                            pred_x = preds.at[frame, x_col]
                            pred_y = preds.at[frame, y_col]
                            
                            if not (np.isnan(gt_x) or np.isnan(gt_y) or 
                                  np.isnan(pred_x) or np.isnan(pred_y)):
                                error = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
                                all_differences.append({
                                    'frame': frame,
                                    'body_part': part,
                                    'x_error': abs(gt_x - pred_x),
                                    'y_error': abs(gt_y - pred_y),
                                    'euclidean_error': error,
                                    'gt_x': gt_x,
                                    'gt_y': gt_y,
                                    'pred_x': pred_x,
                                    'pred_y': pred_y
                                })
                
                # Save to CSV if we have differences
                if all_differences:
                    df = pd.DataFrame(all_differences)
                    output_file = os.path.join(self.detailed_dir, f"{video_name}_{model_name}_differences.csv")
                    df.to_csv(output_file, index=False)
                    print(f"  Saved detailed differences to {output_file}")
    
    def _save_results_csv(self, results: Dict, filename: str) -> None:
        """
        Save a summary of results to a CSV file with comprehensive statistics.
        
        Args:
            results: Dictionary containing analysis results
            filename: Output CSV filename (without path)
        """
        if not results:
            print("  No results to save to CSV")
            return
            
        csv_path = os.path.join(self.results_dir, filename)
        
        # Initialize DataFrames for statistics
        stats_df = pd.DataFrame(columns=['video', 'body_part', 'mean_error', 'std_error', 'n_frames', 'model'])
        model_stats = []
        
        # Process each video and model
        for video_name, video_data in results.items():
            if not isinstance(video_data, dict):
                continue
                
            for model_name, model_data in video_data.items():
                if not isinstance(model_data, dict):
                    continue
                
                # Process body part data
                for part_name, part_data in model_data.items():
                    # Skip non-body part entries
                    if part_name in ['overall', 'mean', 'std', 'median', 'min', 'max', 'mean_normalized', 'std_normalized']:
                        continue
                        
                    if isinstance(part_data, dict):
                        # Handle dictionary format
                        mean_val = part_data.get('mean')
                        std_val = part_data.get('std')
                        n_frames = part_data.get('n', 1)
                        
                        if mean_val is not None and not np.isnan(mean_val):
                            stats_df = pd.concat([
                                stats_df,
                                pd.DataFrame([{
                                    'video': video_name,
                                    'body_part': part_name,
                                    'mean_error': float(mean_val),
                                    'std_error': float(std_val) if std_val is not None else np.nan,
                                    'n_frames': int(n_frames) if n_frames is not None else 1,
                                    'model': model_name
                                }])
                            ], ignore_index=True)
                    
                    # Also check for direct numeric values in the model data
                    elif isinstance(part_data, (int, float)) and not np.isnan(part_data):
                        stats_df = pd.concat([
                            stats_df,
                            pd.DataFrame([{
                                'video': video_name,
                                'body_part': part_name,
                                'mean_error': float(part_data),
                                'std_error': np.nan,
                                'n_frames': 1,
                                'model': model_name
                            }])
                        ], ignore_index=True)
                
                # Extract overall model statistics if available
                if 'overall' in model_data and isinstance(model_data['overall'], dict):
                    overall = model_data['overall']
                    model_stats.append({
                        'video': video_name,
                        'model': model_name,
                        'mean_error': overall.get('mean', np.nan),
                        'std_error': overall.get('std', np.nan),
                        'n_frames': overall.get('n', 0)
                    })
        
        # If no valid data, exit early
        if stats_df.empty and not model_stats:
            print("  No valid results to save to CSV")
            return
            
        # Generate summary statistics
        summary_rows = []
        
        # Add overall model statistics
        for stat in model_stats:
            # Count unique body parts for this video and model
            if not stats_df.empty:
                body_part_count = len(stats_df[
                    (stats_df['video'] == stat['video']) & 
                    (stats_df['model'] == stat['model'])
                ].body_part.unique())
            else:
                body_part_count = 0
                
            summary_rows.append({
                'Video': stat['video'],
                'Model': stat['model'],
                'Type': 'Overall',
                'Body Part': 'All',
                'Mean Error (px)': f"{stat['mean_error']:.2f} ± {stat['std_error']:.2f}",
                'Frames': int(stat['n_frames']),
                'Body Parts': body_part_count
            })
        
        # Add per-body-part statistics
        if not stats_df.empty:
            # Group by video, model, and body part to get mean stats
            part_stats = stats_df.groupby(['video', 'model', 'body_part']).agg({
                'mean_error': 'mean',
                'std_error': 'mean',
                'n_frames': 'sum'
            }).reset_index()
            
            for _, row in part_stats.iterrows():
                summary_rows.append({
                    'Video': row['video'],
                    'Model': row['model'],
                    'Type': 'By Body Part',
                    'Body Part': row['body_part'],
                    'Mean Error (px)': f"{row['mean_error']:.2f} ± {row['std_error']:.2f}",
                    'Frames': int(row['n_frames']),
                    'Body Parts': 1
                })
        
        # Calculate overall statistics for the summary
        if not stats_df.empty:
            overall_stats = stats_df.groupby('model').agg({
                'mean_error': ['mean', 'std', 'count'],
                'n_frames': 'sum'
            }).reset_index()
        else:
            # Create empty stats if no body part data
            overall_stats = pd.DataFrame(columns=['model', 'mean_error', 'n_frames'])
        
        # Create a summary row for each model
        for _, row in overall_stats.iterrows():
            model = row[('model', '')]
            summary_rows.append({
                'Model': model,
                'Type': 'Overall',
                'Body Part': 'All',
                'Mean Error (px)': f"{row[('mean_error', 'mean')]:.2f} ± {row[('mean_error', 'std')]:.2f}",
                'Frames': int(row[('n_frames', 'sum')]),
                'Body Parts': int(row[('mean_error', 'count')])
            })
        
        # Add body part specific statistics if we have data
        if not stats_df.empty:
            # Get unique body parts from the stats DataFrame
            unique_body_parts = stats_df['body_part'].unique()
            
            for part in unique_body_parts:
                part_stats = stats_df[stats_df['body_part'] == part].groupby('model').agg({
                    'mean_error': ['mean', 'std'],
                    'n_frames': 'sum'
                }).reset_index()
                
                for _, row in part_stats.iterrows():
                    model = row[('model', '')]
                    summary_rows.append({
                        'Model': model,
                        'Type': 'By Body Part',
                        'Body Part': part,
                        'Mean Error (px)': f"{row[('mean_error', 'mean')]:.2f} ± {row[('mean_error', 'std')]:.2f}",
                        'Frames': int(row[('n_frames', 'sum')]),
                        'Body Parts': 1
                    })
        
        # Create and save the summary DataFrame
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(csv_path, index=False)
        
        # Print comprehensive summary to console
        print("\n" + "="*70)
        print("POSE ESTIMATION ANALYSIS SUMMARY")
        print("="*70)
        
        # Print overall statistics
        print("\nOVERALL PERFORMANCE")
        print("-" * 50)
        print(overall_stats[['model', 'mean_error']].rename(columns={
            'model': 'Model',
            ('mean_error', 'mean'): 'Mean Error (px)',
            ('mean_error', 'std'): 'Std Dev (px)',
            ('mean_error', 'count'): 'Body Parts',
            ('n_frames', 'sum'): 'Total Frames'
        }).to_string(index=False, float_format='%.2f'))
        
        # Print best and worst performing body parts
        if not stats_df.empty:
            # Best performing body parts (lowest mean error)
            best_parts = stats_df.loc[stats_df.groupby('body_part')['mean_error'].idxmin()]\
                               .sort_values('mean_error')\
                               .head(5)
            
            print("\nBEST PERFORMING BODY PARTS (LOWEST ERROR)")
            print("-" * 50)
            print(best_parts[['body_part', 'mean_error', 'std_error', 'model']]\
                  .rename(columns={
                      'body_part': 'Body Part',
                      'mean_error': 'Mean Error (px)',
                      'std_error': 'Std Dev (px)',
                      'model': 'Model'
                  })\
                  .to_string(index=False, float_format='%.2f'))
            
            # Worst performing body parts (highest mean error)
            worst_parts = stats_df.loc[stats_df.groupby('body_part')['mean_error'].idxmax()]\
                                .sort_values('mean_error', ascending=False)\
                                .head(5)
            
            print("\nWORST PERFORMING BODY PARTS (HIGHEST ERROR)")
            print("-" * 50)
            print(worst_parts[['body_part', 'mean_error', 'std_error', 'model']]\
                  .rename(columns={
                      'body_part': 'Body Part',
                      'mean_error': 'Mean Error (px)',
                      'std_error': 'Std Dev (px)',
                      'model': 'Model'
                  })\
                  .to_string(index=False, float_format='%.2f'))
            
            # Print model comparison
            print("\nMODEL COMPARISON")
            print("-" * 50)
            model_comparison = stats_df.groupby('model').agg({
                'mean_error': ['mean', 'std', 'count'],
                'n_frames': 'sum'
            })
            
            print(model_comparison.rename(columns={
                'mean': 'Mean Error (px)',
                'std': 'Std Dev (px)',
                'count': 'Body Parts',
                'sum': 'Total Frames'
            }).to_string(float_format='%.2f'))
        
        print("\n" + "="*70)
        print(f"Detailed results saved to: {csv_path}")
        print("="*70)


def main():
    """Main function to run the analysis from the command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare pose estimation results.')
    parser.add_argument('--dlc_dir', type=str, required=True,
                       help='Path to the DLC project directory')
    parser.add_argument('--sleap_dir', type=str, required=True,
                       help='Path to the SLEAP export directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save analysis results')
    parser.add_argument('--video', type=str, nargs='+',
                       help='Specific video(s) to analyze (optional)')
    parser.add_argument('--no_plots', action='store_true',
                       help='Disable generation of plots')
    
    args = parser.parse_args()
    
    try:
        # Initialize the analyzer
        analyzer = PoseAnalyzer(
            dlc_project_dir=args.dlc_dir,
            sleap_dir=args.sleap_dir,
            output_dir=args.output_dir
        )
        
        # Generate a timestamp for the output files
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f'pose_analysis_{timestamp}'
        
        # Run the analysis
        results = analyzer.run_analysis(video_list=args.video)
        
        # Save results
        if results:
            analyzer.save_results(results, filename=output_filename)
            
            # Print location of output files
            print("\nAnalysis complete!")
            print(f"Results saved to: {os.path.abspath(args.output_dir)}")
            print(f"Plots saved to: {os.path.abspath(analyzer.plots_dir)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
