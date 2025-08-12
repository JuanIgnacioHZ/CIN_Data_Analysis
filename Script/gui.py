#!/usr/bin/env python3
"""
Pose Analysis GUI

A graphical interface for comparing pose estimation results from DeepLabCut and SLEAP.
"""

import os
import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog,
                            QLineEdit, QMessageBox, QProgressBar, QGroupBox,
                            QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from pose_analysis import PoseAnalyzer

class AnalysisThread(QThread):
    """Worker thread for running the analysis in the background."""
    progress = pyqtSignal(int, str)  # progress percentage, status message
    finished = pyqtSignal(dict)  # results when analysis is complete
    error = pyqtSignal(str)  # error message if something goes wrong
    
    def __init__(self, analyzer, video_list=None):
        super().__init__()
        self.analyzer = analyzer
        self.video_list = video_list or []
        self.results = {}
    
    def run(self):
        """Run the analysis."""
        try:
            self.results = {}
            total_videos = len(self.video_list) if self.video_list else 1
            
            if self.video_list:
                # Analyze specific videos
                for i, video in enumerate(self.video_list):
                    progress = int((i + 1) / total_videos * 100)
                    self.progress.emit(progress, f"Analyzing {video}...")
                    
                    # Analyze the video
                    try:
                        result = self.analyzer.analyze_video(video)
                        self.results[video] = result
                    except Exception as e:
                        self.error.emit(f"Error analyzing {video}: {str(e)}")
                        continue
                    
                    # Small delay to update UI
                    self.msleep(100)
            else:
                # Analyze all videos
                self.progress.emit(50, "Analyzing all videos...")
                self.analyzer.run_analysis()
                
                # Get results from the analyzer
                for video in self.analyzer.labeled_videos:
                    try:
                        result = self.analyzer.analyze_video(video)
                        self.results[video] = result
                    except Exception as e:
                        self.error.emit(f"Error analyzing {video}: {str(e)}")
                        continue
            
            self.progress.emit(100, "Analysis complete!")
            self.finished.emit(self.results)
            
        except Exception as e:
            import traceback
            error_msg = f"Analysis failed: {str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class DirectorySelector(QWidget):
    """Widget for selecting a directory with a label and browse button."""
    
    def __init__(self, label_text, dialog_title):
        super().__init__()
        self.dialog_title = dialog_title
        
        layout = QHBoxLayout()
        self.setLayout(layout)
        
        # Label
        label = QLabel(label_text)
        layout.addWidget(label)
        
        # Directory path display
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        layout.addWidget(self.path_edit, 1)  # Stretch factor 1 to take remaining space
        
        # Browse button
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_directory)
        layout.addWidget(self.browse_btn)
    
    def browse_directory(self):
        """Open a directory selection dialog."""
        directory = QFileDialog.getExistingDirectory(
            self, self.dialog_title, 
            self.path_edit.text() or os.path.expanduser("~")
        )
        if directory:
            self.path_edit.setText(directory)
    
    def get_path(self):
        """Get the selected directory path."""
        return self.path_edit.text()
    
    def set_path(self, path):
        """Set the directory path."""
        self.path_edit.setText(path)


class ResultsTable(QTableWidget):
    """Table widget for displaying analysis results."""
    
    def __init__(self):
        super().__init__(0, 4)  # Video, Model, Avg Error, Std Dev
        self.setHorizontalHeaderLabels(["Video", "Model", "Avg Error (px)", "Body Parts"])
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        
        # Store full results for export
        self.full_results = {}
    
    def update_results(self, results):
        """Update the table with analysis results.
        
        Args:
            results: Dictionary of results from PoseAnalyzer
        """
        
        self.setRowCount(0)  # Clear existing rows
        self.full_results = results or {}
        
        if not results:
            print("No results to display")
            return
            
        # Set headers
        self.setHorizontalHeaderLabels(["Video", "Model", "Average Pixel Error (px)", "Std Dev (px)", "Frames"])
        
        # Debug: Print the structure of the results
        print("\n=== Results Structure ===")
        print(f"Top-level keys: {list(results.keys())}")
        
        row = 0
        
        for video, video_data in results.items():
            if not isinstance(video_data, dict):
                print(f"Skipping non-dict video_data for {video}")
                continue
                
            print(f"\nProcessing video: {video}")
            print(f"Available models: {list(video_data.keys())}")
                
            for model, model_data in video_data.items():
                if not isinstance(model_data, dict):
                    print(f"Skipping non-dict model_data for {model}")
                    continue
                
                print(f"\nProcessing model: {model}")
                print(f"Available keys: {list(model_data.keys())}")
                
                # Get statistics - handle both direct stats and nested statistics key
                if 'statistics' in model_data and isinstance(model_data['statistics'], dict):
                    stats = model_data['statistics']
                    print(f"Found statistics in 'statistics' key")
                else:
                    stats = model_data
                    print("Using model_data directly as stats")
                
                # Debug: Print available statistics keys
                print(f"Available stat keys: {list(stats.keys())}")
                
                # Try to get the average error from different possible locations
                mean_error = None
                std_error = None
                n_frames = 0
                
                # First, try to get data from the _overall key if it exists
                if '_overall' in stats and isinstance(stats['_overall'], dict):
                    overall = stats['_overall']
                    mean_error = overall.get('mean')
                    std_error = overall.get('std', 0)
                    n_frames = overall.get('n', 0)
                    print(f"Found overall stats - mean: {mean_error}, std: {std_error}, n: {n_frames}")
                
                # If no overall stats, try to calculate from individual body parts
                if mean_error is None or np.isnan(mean_error):
                    # Collect all body part data (exclude _overall key)
                    body_parts = {k: v for k, v in stats.items() 
                                if isinstance(v, dict) and k != '_overall'}
                    
                    if body_parts:
                        print(f"Found {len(body_parts)} body parts, calculating stats...")
                        errors = []
                        
                        for part, part_data in body_parts.items():
                            if isinstance(part_data, dict):
                                # Check for mean in the part data
                                if 'mean' in part_data and part_data['mean'] is not None:
                                    try:
                                        errors.append(float(part_data['mean']))
                                        print(f"  {part}: mean={part_data['mean']}")
                                    except (ValueError, TypeError):
                                        continue
                                # If no mean, check if the part data has a 'mean' key directly
                                elif 'mean' in part_data and part_data['mean'] is not None:
                                    try:
                                        errors.append(float(part_data['mean']))
                                        print(f"  {part}: mean={part_data['mean']} (direct)")
                                    except (ValueError, TypeError):
                                        continue
                        
                        if errors:
                            mean_error = float(np.nanmean(errors))
                            std_error = float(np.nanstd(errors, ddof=1) if len(errors) > 1 else 0)
                            n_frames = len(errors)
                            print(f"Calculated from body parts - mean: {mean_error}, std: {std_error}, n: {n_frames}")
                        else:
                            print("No valid mean values found in body parts")
                
                # If we still don't have valid values, try to find any numeric values in the model_data
                if mean_error is None or np.isnan(mean_error):
                    print("No valid mean_error found, searching for numeric values...")
                    for k, v in model_data.items():
                        if isinstance(v, (int, float)) and not np.isnan(v) and v > 0:
                            mean_error = float(v)
                            std_error = 0  # Default std to 0 if we find a single value
                            n_frames = 1
                            print(f"Found numeric value for {k}: {v}")
                            break
                    
                    # If still no valid value, set to NaN
                    if mean_error is None or np.isnan(mean_error):
                        print("No valid numeric values found, using NaN")
                        mean_error = np.nan
                        std_error = np.nan
                        n_frames = 0
                
                # Add a row for each model
                self.insertRow(row)
                
                # Video name
                video_item = QTableWidgetItem(video)
                video_item.setData(Qt.UserRole, video)  # Store original data
                self.setItem(row, 0, video_item)
                
                print(f"Adding row {row}: {video} - {model} - {mean_error:.2f} ± {std_error:.2f} (n={n_frames})")
                
                # Model name
                model_item = QTableWidgetItem(model)
                model_item.setTextAlignment(Qt.AlignCenter)
                self.setItem(row, 1, model_item)
                
                # Average error with standard deviation
                error_text = f"{mean_error:.2f} ± {std_error:.2f}" if not np.isnan(mean_error) else "N/A"
                error_item = QTableWidgetItem(error_text)
                error_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.setItem(row, 2, error_item)
                
                # Body parts summary
                parts_text = ""
                if 'body_parts' in stats and stats['body_parts']:
                    parts_text = ", ".join([
                        f"{k}: {v.get('mean', 'N/A'):.1f}±{v.get('std', 'N/A'):.1f}" 
                        for k, v in stats['body_parts'].items() 
                        if isinstance(v, dict)
                    ])
                parts_item = QTableWidgetItem(parts_text)
                self.setItem(row, 3, parts_item)
                
                # Store the number of frames in the last column
                frames_item = QTableWidgetItem(str(n_frames))
                frames_item.setTextAlignment(Qt.AlignCenter)
                self.setItem(row, 4, frames_item)
                
                row += 1


class PoseAnalysisApp(QMainWindow):
    """Main application window for the Pose Analysis tool."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose Estimation Analysis Tool")
        self.setMinimumSize(800, 600)
        
        # Initialize UI
        self.init_ui()
        
        # Analysis thread
        self.analysis_thread = None
        self.analyzer = None
    
    def init_ui(self):
        """Initialize the user interface."""
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create tabs
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Add tabs
        tabs.addTab(self.create_setup_tab(), "Setup")
        tabs.addTab(self.create_results_tab(), "Results")
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def create_setup_tab(self):
        """Create the setup tab with directory selectors."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add a group for directory selection
        dir_group = QGroupBox("Project Directories")
        dir_layout = QVBoxLayout()
        
        # DLC project directory selector
        self.dlc_selector = DirectorySelector(
            "DLC Project Directory:", 
            "Select DLC Project Directory"
        )
        dir_layout.addWidget(self.dlc_selector)
        
        # SLEAP directory selector
        self.sleap_selector = DirectorySelector(
            "SLEAP Export Directory:", 
            "Select SLEAP Export Directory"
        )
        dir_layout.addWidget(self.sleap_selector)
        
        # Output directory selector
        self.output_selector = DirectorySelector(
            "Output Directory:", 
            "Select Output Directory"
        )
        # Set default output directory
        default_output = os.path.join(os.path.expanduser("~"), "pose_analysis_results")
        self.output_selector.set_path(default_output)
        dir_layout.addWidget(self.output_selector)
        
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)
        
        # Add some vertical space
        layout.addSpacing(20)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Analyze button
        self.analyze_btn = QPushButton("Run Analysis")
        self.analyze_btn.clicked.connect(self.start_analysis)
        layout.addWidget(self.analyze_btn)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        return tab
    
    def create_results_tab(self):
        """Create the results tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Results table
        self.results_table = ResultsTable()
        layout.addWidget(self.results_table)
        
        # Add some buttons for exporting, etc.
        btn_layout = QHBoxLayout()
        
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self.export_results)
        btn_layout.addWidget(export_btn)
        
        clear_btn = QPushButton("Clear Results")
        clear_btn.clicked.connect(self.clear_results)
        btn_layout.addWidget(clear_btn)
        
        layout.addLayout(btn_layout)
        
        return tab
    
    def start_analysis(self):
        """Start the analysis process."""
        # Get directory paths
        dlc_dir = self.dlc_selector.get_path()
        sleap_dir = self.sleap_selector.get_path()
        output_dir = self.output_selector.get_path()
        
        # Validate directories
        if not all([dlc_dir, sleap_dir, output_dir]):
            QMessageBox.warning(self, "Error", "Please select all required directories.")
            return
        
        if not os.path.exists(dlc_dir):
            QMessageBox.warning(self, "Error", f"DLC directory does not exist: {dlc_dir}")
            return
        
        if not os.path.exists(sleap_dir):
            QMessageBox.warning(self, "Error", f"SLEAP directory does not exist: {sleap_dir}")
            return
        
        # Create output directory if it doesn't exist
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not create output directory: {str(e)}")
            return
        
        # Initialize the analyzer with timestamped output
        self.analyzer = PoseAnalyzer(dlc_dir, sleap_dir, output_dir, use_timestamp=True)
        
        # Update the output directory in the UI to show the actual output path being used
        self.output_selector.set_path(self.analyzer.output_dir)
        
        # Update UI
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.statusBar().showMessage("Starting analysis...")
        
        # Start analysis in a separate thread
        self.analysis_thread = AnalysisThread(self.analyzer)
        self.analysis_thread.progress.connect(self.update_progress)
        self.analysis_thread.finished.connect(self.analysis_finished)
        self.analysis_thread.error.connect(self.analysis_error)
        self.analysis_thread.start()
    
    def update_progress(self, value, message):
        """Update the progress bar and status message."""
        self.progress_bar.setValue(value)
        self.statusBar().showMessage(message)
    
    def analysis_finished(self, results):
        """Handle completion of the analysis."""
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.statusBar().showMessage("Analysis complete!")
        
        # Update results table
        self.results_table.update_results(results)
        
        # Switch to results tab
        self.centralWidget().findChild(QTabWidget).setCurrentIndex(1)
    
    def analysis_error(self, error_message):
        """Handle errors during analysis."""
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Analysis Error", error_message)
        self.statusBar().showMessage("Analysis failed")
    
    def export_results(self):
        """Export the analysis results to a file."""
        if not hasattr(self, 'analyzer') or not self.analyzer or not hasattr(self, 'analysis_thread'):
            QMessageBox.warning(self, "Export Error", "No analysis results to export.")
            return
            
        if not self.analysis_thread.results:
            QMessageBox.warning(self, "Export Error", "No analysis results to export.")
            return
            
        # Get output directory
        output_dir = self.output_selector.get_path()
        if not output_dir or not os.path.exists(output_dir):
            QMessageBox.warning(self, "Export Error", "Output directory does not exist.")
            return
            
        try:
            # Create a timestamped subdirectory for this export
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = os.path.join(output_dir, f"export_{timestamp}")
            os.makedirs(export_dir, exist_ok=True)
            
            # Save results to CSV
            import pandas as pd
            
            # Prepare data for CSV export
            rows = []
            for video, video_data in self.analysis_thread.results.items():
                for model, model_data in video_data.items():
                    if model not in ["DLC", "SLEAP"]:
                        continue
                        
                    stats = model_data.get("statistics", {})
                    rows.append({
                        "video": video,
                        "model": model,
                        "mean_error": stats.get("mean_error", 0),
                        "std_error": stats.get("std_error", 0),
                        "num_frames": stats.get("num_frames", 0)
                    })
            
            # Save summary CSV
            if rows:
                df = pd.DataFrame(rows)
                csv_path = os.path.join(export_dir, "summary.csv")
                df.to_csv(csv_path, index=False)
            
            # Copy visualizations if they exist
            for video in self.analyzer.labeled_videos:
                vis_path = os.path.join(self.analyzer.output_dir, f"{video}_model_comparison.png")
                if os.path.exists(vis_path):
                    import shutil
                    shutil.copy2(vis_path, os.path.join(export_dir, os.path.basename(vis_path)))
            
            # Show success message
            QMessageBox.information(
                self, 
                "Export Complete", 
                f"Results exported to:\n{export_dir}"
            )
            
            # Open the export directory
            import subprocess
            if sys.platform.startswith('darwin'):  # macOS
                subprocess.Popen(['open', export_dir])
            elif os.name == 'nt':  # Windows
                os.startfile(export_dir)
            elif os.name == 'posix':  # Linux, etc.
                subprocess.Popen(['xdg-open', export_dir])
                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Export Error", 
                f"Failed to export results: {str(e)}"
            )
    
    def clear_results(self):
        """Clear the results table."""
        self.results_table.update_results({})


def main():
    """Main function to start the application."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show the main window
    window = PoseAnalysisApp()
    window.show()
    
    # Start the event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
