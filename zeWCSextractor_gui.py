#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FITS Header Extractor GUI
-------------------------

Petit outil avec interface Tkinter pour :
- sélectionner un dossier contenant des fichiers FIT/FITS/FTS
- parcourir les fichiers (option récursive)
- extraire tous les headers de tous les HDU
- écrire le résultat dans un fichier texte

Dépendance :
    pip install astropy
"""

from __future__ import annotations

import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from astropy.io import fits
except Exception:
    fits = None


FITS_EXTENSIONS = {".fit", ".fits", ".fts"}


def collect_fits_files(folder: Path, recursive: bool) -> list[Path]:
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    files: list[Path] = []
    for p in iterator:
        try:
            if p.is_file() and p.suffix.lower() in FITS_EXTENSIONS:
                files.append(p)
        except Exception:
            continue
    return sorted(files, key=lambda x: str(x).lower())


def header_to_lines(file_path: Path) -> list[str]:
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append(f"FILE: {file_path}")
    lines.append("=" * 100)

    try:
        with fits.open(file_path, ignore_missing_simple=True) as hdul:
            lines.append(f"Number of HDU(s): {len(hdul)}")
            lines.append("")

            for idx, hdu in enumerate(hdul):
                lines.append("-" * 100)
                lines.append(f"HDU {idx}: {type(hdu).__name__}")
                lines.append("-" * 100)

                header = getattr(hdu, "header", None)
                if header is None or len(header) == 0:
                    lines.append("[Empty header]")
                    lines.append("")
                    continue

                for card in header.cards:
                    try:
                        keyword = str(card.keyword)
                        value = repr(card.value)
                        comment = str(card.comment) if card.comment is not None else ""
                        if comment:
                            lines.append(f"{keyword} = {value} / {comment}")
                        else:
                            lines.append(f"{keyword} = {value}")
                    except Exception:
                        lines.append(str(card))

                lines.append("")
    except Exception as exc:
        lines.append(f"[ERROR] Unable to read file: {exc}")
        lines.append(traceback.format_exc())
        lines.append("")

    lines.append("")
    return lines


class FitsHeaderExtractorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("FITS Header Extractor")
        self.root.geometry("760x520")
        self.root.minsize(700, 480)

        self.selected_folder = tk.StringVar()
        self.output_file = tk.StringVar()
        self.recursive = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        title = ttk.Label(
            main,
            text="Extraction des headers FITS vers un fichier texte",
            font=("Segoe UI", 12, "bold"),
        )
        title.pack(anchor="w", pady=(0, 12))

        folder_frame = ttk.LabelFrame(main, text="Dossier source", padding=10)
        folder_frame.pack(fill="x", pady=(0, 10))

        ttk.Entry(folder_frame, textvariable=self.selected_folder).pack(
            side="left", fill="x", expand=True, padx=(0, 8)
        )
        ttk.Button(folder_frame, text="Parcourir…", command=self.choose_folder).pack(side="left")

        output_frame = ttk.LabelFrame(main, text="Fichier texte de sortie", padding=10)
        output_frame.pack(fill="x", pady=(0, 10))

        ttk.Entry(output_frame, textvariable=self.output_file).pack(
            side="left", fill="x", expand=True, padx=(0, 8)
        )
        ttk.Button(output_frame, text="Choisir…", command=self.choose_output_file).pack(side="left")

        options_frame = ttk.LabelFrame(main, text="Options", padding=10)
        options_frame.pack(fill="x", pady=(0, 10))

        ttk.Checkbutton(
            options_frame,
            text="Inclure les sous-dossiers",
            variable=self.recursive,
        ).pack(anchor="w")

        buttons = ttk.Frame(main)
        buttons.pack(fill="x", pady=(0, 10))

        ttk.Button(buttons, text="Extraire les headers", command=self.run_extraction).pack(
            side="left"
        )
        ttk.Button(buttons, text="Quitter", command=self.root.destroy).pack(side="right")

        log_frame = ttk.LabelFrame(main, text="Journal", padding=10)
        log_frame.pack(fill="both", expand=True)

        self.log_widget = tk.Text(log_frame, wrap="word", height=18)
        self.log_widget.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_widget.yview)
        scroll.pack(side="right", fill="y")
        self.log_widget.configure(yscrollcommand=scroll.set)

        self.log("Application prête.")

    def log(self, message: str) -> None:
        self.log_widget.insert("end", message + "\n")
        self.log_widget.see("end")
        self.root.update_idletasks()

    def choose_folder(self) -> None:
        folder = filedialog.askdirectory(title="Choisir le dossier contenant les FITS")
        if folder:
            self.selected_folder.set(folder)
            if not self.output_file.get().strip():
                default_output = Path(folder) / "fits_headers_export.txt"
                self.output_file.set(str(default_output))

    def choose_output_file(self) -> None:
        filename = filedialog.asksaveasfilename(
            title="Choisir le fichier texte de sortie",
            defaultextension=".txt",
            filetypes=[("Fichier texte", "*.txt"), ("Tous les fichiers", "*.*")],
            initialfile="fits_headers_export.txt",
        )
        if filename:
            self.output_file.set(filename)

    def run_extraction(self) -> None:
        if fits is None:
            messagebox.showerror(
                "Astropy manquant",
                "Le module astropy n'est pas installé.\n\nInstalle-le avec :\n\npip install astropy",
            )
            return

        folder_text = self.selected_folder.get().strip()
        output_text = self.output_file.get().strip()

        if not folder_text:
            messagebox.showwarning("Dossier manquant", "Choisis un dossier source.")
            return

        if not output_text:
            messagebox.showwarning("Sortie manquante", "Choisis un fichier texte de sortie.")
            return

        folder = Path(folder_text)
        output_path = Path(output_text)

        if not folder.exists() or not folder.is_dir():
            messagebox.showerror("Dossier invalide", "Le dossier sélectionné est invalide.")
            return

        self.log("=" * 60)
        self.log(f"Dossier source : {folder}")
        self.log(f"Fichier sortie : {output_path}")
        self.log(f"Mode récursif : {'oui' if self.recursive.get() else 'non'}")
        self.log("Recherche des fichiers FITS...")

        try:
            files = collect_fits_files(folder, self.recursive.get())
        except Exception as exc:
            messagebox.showerror("Erreur", f"Erreur pendant la recherche des fichiers :\n{exc}")
            return

        if not files:
            self.log("Aucun fichier FIT/FITS/FTS trouvé.")
            messagebox.showinfo("Aucun fichier", "Aucun fichier FIT/FITS/FTS trouvé.")
            return

        self.log(f"{len(files)} fichier(s) trouvé(s).")
        self.log("Extraction des headers...")

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with output_path.open("w", encoding="utf-8") as fh:
                fh.write("FITS HEADER EXPORT\n")
                fh.write(f"Source folder: {folder}\n")
                fh.write(f"Recursive: {self.recursive.get()}\n")
                fh.write(f"Files found: {len(files)}\n\n")

                for idx, file_path in enumerate(files, start=1):
                    self.log(f"[{idx}/{len(files)}] {file_path.name}")
                    lines = header_to_lines(file_path)
                    fh.write("\n".join(lines))
                    fh.write("\n")

            self.log("Extraction terminée.")
            self.log(f"Fichier créé : {output_path}")
            messagebox.showinfo(
                "Terminé",
                f"Extraction terminée.\n\n{len(files)} fichier(s) traité(s).\n\nSortie :\n{output_path}",
            )
        except Exception as exc:
            self.log(f"Erreur : {exc}")
            self.log(traceback.format_exc())
            messagebox.showerror("Erreur", f"Erreur pendant l'extraction :\n{exc}")


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    FitsHeaderExtractorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
