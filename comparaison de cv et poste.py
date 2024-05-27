import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from tkinter.ttk import Progressbar
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
from langdetect import detect

# Charger le modèle et le tokenizer LaBSE
model_name = "sentence-transformers/LaBSE"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Fonction pour générer les embeddings
def generate_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).pooler_output
    return embeddings

# Pré-traitement des données
def preprocess_text(text):
    return text.strip().lower()

# Fonction pour extraire le texte des fichiers PDF
def extract_text_from_pdfs(pdf_paths):
    texts = []
    for pdf_path in pdf_paths:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            texts.append(preprocess_text(text))
    return texts

# Fonction pour lancer le traitement
def process_data():
    cv_texts = cv_text_area.get("1.0", tk.END).strip().split("\n\n")
    job_texts = job_text_area.get("1.0", tk.END).strip().split("\n\n")

    if not cv_texts[0] and cv_pdf_paths:
        cv_texts = extract_text_from_pdfs(cv_pdf_paths)
    if not job_texts[0] and job_pdf_paths:
        job_texts = extract_text_from_pdfs(job_pdf_paths)

    if not cv_texts or not job_texts:
        messagebox.showerror("Erreur", "Veuillez fournir au moins un CV et une description de poste.")
        return

    # Générer les embeddings pour les CV et les descriptions de postes
    cv_embeddings = generate_embeddings(cv_texts)
    job_embeddings = generate_embeddings(job_texts)

    # Calculer la similarité cosinus
    similarity_matrix = cosine_similarity(cv_embeddings, job_embeddings)

    # Générer le rapport
    report = []
    total_steps = len(cv_texts)
    for i in range(len(cv_texts)):
        best_match_idx = similarity_matrix[i].argmax()
        similarity_score = similarity_matrix[i][best_match_idx] * 100  # Convertir en pourcentage
        report.append({
            "CV": cv_texts[i][:30] + "...",  # Affichage partiel du texte pour la lisibilité
            "Description de poste correspondante": job_texts[best_match_idx][:30] + "...",
            "Score de similarité": similarity_score
        })
        update_progress(i + 1, total_steps)

    report_df = pd.DataFrame(report)
    report_df.to_csv("cv_job_matching_report.csv", index=False)
    display_report(report)

# Fonction pour mettre à jour la barre de progression
def update_progress(current_step, total_steps):
    progress = int((current_step / total_steps) * 100)
    progress_var.set(progress)
    root.update_idletasks()

# Fonction pour afficher le rapport détaillé
def display_report(report):
    report_window = tk.Toplevel(root)
    report_window.title("Rapport d'analyse de correspondance")

    report_text = scrolledtext.ScrolledText(report_window, width=100, height=30)
    report_text.pack(padx=10, pady=10)

    for entry in report:
        report_text.insert(tk.END, f"CV: {entry['CV']}\n")
        report_text.insert(tk.END, f"Description de poste correspondante: {entry['Description de poste correspondante']}\n")
        report_text.insert(tk.END, f"Score de similarité: {entry['Score de similarité']:.2f}%\n")
        report_text.insert(tk.END, "-"*100 + "\n")

    messagebox.showinfo("Succès", "Le rapport a été généré et affiché.")

# Fonctions pour sélectionner les fichiers PDF
def select_cv_pdfs():
    global cv_pdf_paths
    cv_pdf_paths = filedialog.askopenfilenames(filetypes=[("PDF files", "*.pdf")])
    cv_pdf_label.config(text=f"{len(cv_pdf_paths)} fichiers sélectionnés")

def select_job_pdfs():
    global job_pdf_paths
    job_pdf_paths = filedialog.askopenfilenames(filetypes=[("PDF files", "*.pdf")])
    job_pdf_label.config(text=f"{len(job_pdf_paths)} fichiers sélectionnés")

# Création de l'interface utilisateur
root = tk.Tk()
root.title("Analyse de CV et d'Offres d'Emploi")

cv_frame = tk.Frame(root)
cv_frame.pack(padx=10, pady=5, fill="x")
cv_button = tk.Button(cv_frame, text="Sélectionner des fichiers PDF pour les CV", command=select_cv_pdfs)
cv_button.pack(side="left")
cv_pdf_label = tk.Label(cv_frame, text="")
cv_pdf_label.pack(side="left", padx=5)
cv_text_area = scrolledtext.ScrolledText(cv_frame, width=60, height=10)
cv_text_area.pack(padx=5, pady=5)
cv_text_area.insert(tk.END, "Copiez et collez le texte des CV ici, ou sélectionnez des fichiers PDF.")

job_frame = tk.Frame(root)
job_frame.pack(padx=10, pady=5, fill="x")
job_button = tk.Button(job_frame, text="Sélectionner des fichiers PDF pour les descriptions de postes", command=select_job_pdfs)
job_button.pack(side="left")
job_pdf_label = tk.Label(job_frame, text="")
job_pdf_label.pack(side="left", padx=5)
job_text_area = scrolledtext.ScrolledText(job_frame, width=60, height=10)
job_text_area.pack(padx=5, pady=5)
job_text_area.insert(tk.END, "Copiez et collez le texte des descriptions de postes ici, ou sélectionnez des fichiers PDF.")

progress_var = tk.IntVar()
progress_bar = Progressbar(root, variable=progress_var, maximum=100)
progress_bar.pack(pady=10, fill="x")

process_button = tk.Button(root, text="Lancer le traitement", command=process_data)
process_button.pack(pady=10)

root.mainloop()
