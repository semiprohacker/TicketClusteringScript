import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
from ticket_clusterer import train_model, predict_with_model

class TicketClustererGUI:
    def __init__(self, master):
        self.master = master
        master.title("Ticket Clusterer")
        master.geometry("800x600")

        self.file_path = ""

        # --- Main Frame ---
        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- File Selection ---
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        file_frame.pack(fill=tk.X, pady=5)

        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        browse_button = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_button.pack(side=tk.RIGHT)

        # --- Controls ---
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        self.train_button = ttk.Button(control_frame, text="Train Model", command=self.start_training_thread)
        self.train_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        self.predict_button = ttk.Button(control_frame, text="Predict with Model", command=self.start_prediction_thread)
        self.predict_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        # --- Log ---
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.log_text = tk.Text(log_frame, wrap=tk.WORD, state="disabled")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # --- Scrollbar for Log ---
        scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        self.log_text['yscrollcommand'] = scrollbar.set
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def log(self, message):
        """Appends a message to the log text area."""
        self.master.after(0, self._log, message)

    def _log(self, message):
        """Helper function to log from other threads."""
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.config(state="disabled")
        self.log_text.see(tk.END)

    def browse_file(self):
        """Opens a file dialog to select an Excel file."""
        file_path = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=(("Excel files", "*.xlsx"), ("All files", "*.*"))
        )
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=os.path.basename(file_path))
            self.log(f"Selected file: {self.file_path}")

    def set_buttons_state(self, state):
        """Disables or enables the buttons."""
        self.train_button.config(state=state)
        self.predict_button.config(state=state)

    def start_training_thread(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select a file first.")
            return
        
        self.set_buttons_state("disabled")
        thread = threading.Thread(target=self.train_model)
        thread.start()

    def train_model(self):
        """Placeholder for model training logic."""
        self.log("--- Starting Model Training ---")
        success, message = train_model(self.file_path, self.log)
        if success:
            self.log("--- Model Training Finished Successfully ---")
        else:
            self.log(f"--- Model Training Failed: {message} ---")
        self.master.after(0, self.set_buttons_state, "normal")


    def start_prediction_thread(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select a file first.")
            return
        
        self.set_buttons_state("disabled")
        thread = threading.Thread(target=self.predict_with_model)
        thread.start()

    def predict_with_model(self):
        """Placeholder for prediction logic."""
        self.log("--- Starting Prediction ---")
        success, message = predict_with_model(self.file_path, self.log)
        if success:
            self.log("--- Prediction Finished Successfully ---")
        else:
            self.log(f"--- Prediction Failed: {message} ---")
        self.master.after(0, self.set_buttons_state, "normal")


if __name__ == '__main__':
    root = tk.Tk()
    app = TicketClustererGUI(root)
    root.mainloop()
