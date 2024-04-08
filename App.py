import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from main import logistic_model, windowSize, normalize_features
import numpy as np


def preprocess_and_predict(df, model, window_size=100):

    roll_data = pd.DataFrame(df).rolling(window_size).median().dropna()

    roll_data = roll_data.iloc[:, ]


    # Normalize the data
    # norm_data = scaler.fit_transform(roll_data)

    features = pd.DataFrame(columns=['mean', 'std', 'max', 'min', 'variance', 'skewness', 'kurtosis', 'range', 'median', 'rms'])

    features['mean'] = df.mean()
    features['std'] = df.std()
    features['max'] = df.max()
    features['min'] = df.min()
    features['variance'] = df.var()
    features['skewness'] = df.skew()
    features['kurtosis'] = df.kurt()
    features['range'] = df.max() - df.min()
    features['median'] = df.median()
    features['rms'] = np.sqrt(np.mean(df**2))

    normalized_features = normalize_features(features)

    dataNew = roll_data.iloc[:, 2:4]

    # Predict using the logistic regression model
    predictions = model.predict(dataNew)

    # Ensure 'Predicted' column exists in the DataFrame
    df['Predicted'] = np.nan

    # Use .loc to avoid SettingWithCopyWarning, ensuring we're modifying the DataFrame directly
    df.loc[df.index[window_size - 1:], 'Predicted'] = predictions
    return df

class ActivityClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Activity Classifier")
        self.root.geometry("500x250")  # Adjust size as needed

        # Upload Button
        self.upload_button = tk.Button(root, text="Upload CSV Data", command=self.upload_data)
        self.upload_button.pack(pady=20)

        # Classify Button
        self.classify_button = tk.Button(root, text="Classify Activities", command=self.classify)
        self.classify_button.pack(pady=10)

        # Label for displaying the uploaded file name
        self.file_label = tk.Label(root, text="", fg="blue")
        self.file_label.pack(pady=5)

        self.filename = ""

    def upload_data(self):
        self.filename = filedialog.askopenfilename(title="Select a CSV file", filetypes=[("CSV files", "*.csv")])
        if self.filename:
            # Update the label with the name of the file
            self.file_label.config(text="Uploaded: " + self.filename.split('/')[-1])
            messagebox.showinfo("Info", "File uploaded successfully!")

    def classify(self):
        if not self.filename:
            messagebox.showwarning("Warning", "Please upload a CSV file first!")
            return

        data = pd.read_csv(self.filename)

        # Preprocess and predict
        updated_df = preprocess_and_predict(data, logistic_model, windowSize)

        # Asks the user where to save the CSV
        f = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if f is None:  # If the user cancelled the dialog, f will be None
            return

        # Save the DataFrame to the chosen file path
        updated_df.to_csv(f, index=False)

        # Notify the user
        messagebox.showinfo("Success", "Results saved to CSV successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    app = ActivityClassifierApp(root)
    root.mainloop()