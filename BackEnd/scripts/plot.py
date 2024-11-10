import matplotlib.pyplot as plt
import numpy as np
import BackEnd.scripts.load as sl
import os
import pandas as pd

class Plotting:
    
    def __init__(self, data, score, y_pred, key_change_indices, modelname):
        self.save_path = os.path.join(os.path.dirname(__file__), '../model/logs')
        self.data = data
        self.score = score
        self.y_pred = y_pred
        self.key_change_indices = key_change_indices
        self.modelname = modelname
    
    def draw_lines(self):
        # Visualize Data Points and Outliers
        plt.figure(figsize=(15, 5))
        
        if self.modelname == 'Lof':
            # Plot the anomaly scores in blue
            plt.plot(self.data.index, self.score, color='blue', label='Anomaly Scores')
            
            # Mark key change points with green vertical lines
            for i, idx in enumerate(self.key_change_indices):
                plt.axvline(x=idx, color='green', linestyle='--', label='Key Change' if i == 0 else None, zorder=5)

            self.key_change_indices.insert(0, 0)
            prediction_lines = []
            print_text_list = []
            
            # Iterate through the key change indices and find the last red outlier for each segment
            for i in range(1, len(self.key_change_indices)):
                start_idx = self.key_change_indices[i - 1] if i == 1 else self.key_change_indices[i - 1] + 1
                end_idx = self.key_change_indices[i]

                # Identify outliers in this segment
                segment_outliers = self.data.index[start_idx:end_idx][self.y_pred[start_idx:end_idx] == -1]
                
                if len(segment_outliers) > 0:
                    # Get the last outlier index in the segment
                    last_outlier_idx = segment_outliers[-1]
                    
                    # Mark the last outlier with a red vertical line
                    plt.axvline(x=last_outlier_idx, color='red', linestyle='--', label='Anomaly' if i == 0 else None, zorder=5)

                    # Calculate the percentage of early anomaly detection
                    anomaly_detection_percentage = ((last_outlier_idx - start_idx) / (end_idx - start_idx)) * 100
                    prediction_lines.append(anomaly_detection_percentage)

                    print_text = f"Key change at {start_idx} to {end_idx}: Anomaly detected {anomaly_detection_percentage:.2f}% early at index {last_outlier_idx}"
                    print_text_list.append(print_text)
                    
            prediction_mean = np.mean(prediction_lines) if prediction_lines else 0
            print_text_list.append(f"Average early anomaly detection: {prediction_mean:.2f}%")
            
            
        else:
            # Set a threshold for anomaly detection (e.g., using the 95th percentile)
            threshold = np.percentile(self.y_pred, 97)
            anomalies = self.y_pred >= threshold
            anomalies_indices = []

            # Plot the anomaly scores
            plt.plot(self.data.index[0:], self.score, label='Anomaly Score', color='blue')
            
            for key in self.data['key'].unique():
                key_data = self.data[self.data['key'] == key]

                anomalies_key = anomalies[key_data.index]
                first_anomaly_idx = key_data.index[anomalies_key].min() if anomalies_key.any() else None

                # Append to the corresponding lists
                if first_anomaly_idx is not None:
                    anomalies_indices.append(first_anomaly_idx)
            
            # Plot vertical lines for each key
            for i, idx in enumerate(self.key_change_indices):
                plt.axvline(x=self.data.index[idx], color='green', linestyle='--', label='Key Change' if i == 0 else None, zorder=5)

            for i, idx in enumerate(anomalies_indices):
                plt.axvline(x=idx, color='red', linestyle='--', label='Anomaly' if i == 0 else None)
                
            self.key_change_indices.insert(0, 0)
            prediction_lines = []
            print_text_list = []
            
            for i in range(1, len(self.key_change_indices)):
                start_idx = self.key_change_indices[i - 1] if i == 1 else self.key_change_indices[i - 1] + 1
                end_idx = self.key_change_indices[i]

                # Find red lines within this range
                red_lines_in_range = [idx for idx in anomalies_indices if start_idx <= idx <= end_idx]

                # Calculate the percentage for each red line within the range
                if red_lines_in_range:
                    range_length = end_idx - start_idx
                    for red_idx in red_lines_in_range:
                        percentage_location = (red_idx - start_idx) / range_length * 100
                        prediction_lines.append(percentage_location)
                        print_text = f"Red line at index {red_idx} is at {percentage_location:.2f}% of the range from {start_idx} to {end_idx}."
                        print_text_list.append(print_text)
                        
            prediction_mean = np.mean(prediction_lines)
            print_text_list.append(f"Average early anomaly detection: {prediction_mean:.2f}%")
        
        # Save
        self.save_plot(plt)
        self.save_text(print_text_list)
        
    def save_plot(self, plt):
        # Ensure the save directory exists
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # Add title, labels, and legend
        plt.title('Anomaly Detection Results with Key Change and Early Detection')
        plt.xlabel('Index')
        plt.ylabel('Anomaly Score')
        plt.legend(loc='upper left')
        plt.tight_layout()
        
        # Define the save path for the plot image
        path = os.path.join(self.save_path, f'anomaly_detection_plot_{self.modelname}.png')
        
        # Save the plot
        plt.savefig(path)
        plt.close()  # Close the plot to free memory
        
    def save_text(self, log_text_list):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # Convert log messages into DataFrame
        df = pd.DataFrame(log_text_list, columns=['log'])
        
        # Define the save path for the model's log CSV
        path = os.path.join(self.save_path, f'{self.modelname}.csv')
        
        # If the CSV file already exists, delete it
        if os.path.exists(path):
            os.remove(path)
        
        # Save the new DataFrame to CSV
        df.to_csv(path, index=False)