import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import threading
import fitz  # PyMuPDF

# 
def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return "" # Hata durumunda boş string döndürecek

# summarizer.pydan fonksiyonları import etme
from summarizer import summarize_10k_report, summarize_8k_report

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ReportSummarizerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Annual & Current Report Summarizer")
        self.geometry("500x400")

        #dosya yolu
        self.selected_file = None
        self.report_type = ctk.StringVar(value="10-K")

        # Başlık
        self.title_label = ctk.CTkLabel(self, text="AI Report Summarizer", font=("Arial", 22, "bold"))
        self.title_label.pack(pady=20)

        # Report tipi seçim
        self.radio_10k = ctk.CTkRadioButton(self, text="10-K Report", variable=self.report_type, value="10-K")
        self.radio_10k.pack(pady=5)
        self.radio_8k = ctk.CTkRadioButton(self, text="8-K Report", variable=self.report_type, value="8-K")
        self.radio_8k.pack(pady=5)

        # Dosya seçme butonu
        self.select_button = ctk.CTkButton(self, text="Select PDF File", command=self.select_file)
        self.select_button.pack(pady=15)

        # Generate butonu
        self.generate_button = ctk.CTkButton(self, text="Generate Report", command=self.generate_report_threaded)
        self.generate_button.pack(pady=15)

        # Status label
        self.status_label = ctk.CTkLabel(self, text="", font=("Arial", 14))
        self.status_label.pack(pady=10)

    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("PDF Files", "*.pdf")]
        )
        if file_path:
            self.selected_file = file_path
            self.status_label.configure(text=f"Selected: {os.path.basename(file_path)}")

    def generate_report_threaded(self):
        # Thread kullanarak UI donmasını önlüyoz
        thread = threading.Thread(target=self.generate_report)
        thread.start()

    def generate_report(self):
        if not self.selected_file:
            messagebox.showerror("Error", "Please select a PDF file first.")
            return

        self.status_label.configure(text="Processing... Please wait.")

        # PDFten metin çıkarma
        extracted_text = extract_text_from_pdf(self.selected_file)

        # Metin boş veya çok kısaysa hata alma dummy pdfler için
        if len(extracted_text.strip()) < 50: # avg karakter sayısını tespit edersem düzenlerim
            messagebox.showerror("Error", "The selected PDF is empty or does not contain enough text for analysis.")
            self.status_label.configure(text="❌ Error occurred")
            return

        try:
            if self.report_type.get() == "10-K":
                summarize_10k_report(self.selected_file)
            else:
                summarize_8k_report(self.selected_file)

            self.status_label.configure(text="✅ Report generated successfully!")
            messagebox.showinfo("Success", "Report has been generated and saved.")
        except Exception as e:
            self.status_label.configure(text="❌ Error occurred")
            messagebox.showerror("Error", f"An error occurred:\n{e}")

if __name__ == "__main__":
    app = ReportSummarizerApp()
    app.mainloop()
