"""Genere les rapports HTML professionnels depuis les notebooks."""
import subprocess
import sys
import os
from datetime import datetime

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOTEBOOKS = os.path.join(BASE, "notebooks")
OUTPUT = os.path.join(BASE, "rapports")

date_fr = datetime.now().strftime("%d/%m/%Y")

CSS = """
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: "Segoe UI", Arial, sans-serif;
  font-size: 13.5px;
  line-height: 1.7;
  color: #1a1a2e;
  background: #dde1e9;
}

/* ── Bandeau en-tete ── */
.rapport-header {
  background: #1a1a2e;
  color: #ffffff;
  padding: 44px 72px 34px;
  border-bottom: 4px solid #4a90d9;
  border-radius: 6px 6px 0 0;
}
.rapport-header h1 {
  font-size: 22px;
  font-weight: 700;
  letter-spacing: 0.5px;
  margin-bottom: 8px;
}
.rapport-header .meta {
  font-size: 12px;
  color: #a8b8cc;
  display: flex;
  gap: 32px;
  flex-wrap: wrap;
}
.rapport-header .meta span { border-right: 1px solid #3a4a5a; padding-right: 32px; }
.rapport-header .meta span:last-child { border-right: none; }

/* ── Cadre A4 avec ombre ── */
.page-a4 {
  width: 210mm;
  min-height: 297mm;
  margin: 36px auto;
  background: #ffffff;
  box-shadow: 0 4px 24px rgba(0,0,0,0.18), 0 1px 6px rgba(0,0,0,0.10);
  border-radius: 2px;
}

/* ── Corps du rapport ── */
body.jp-Notebook {
  background: #dde1e9 !important;
  padding: 0 !important;
  margin: 0 !important;
}

main {
  padding: 56px 72px 80px !important;
  background: transparent;
  max-width: 100%;
}

/* ── Cellules ── */
.jp-Cell, .cell {
  margin-bottom: 20px;
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
}

/* Masquer UNIQUEMENT les inputs des cellules de CODE */
.jp-CodeCell .jp-Cell-inputWrapper,
.jp-CodeCell .jp-InputArea,
.jp-CodeCell .jp-InputPrompt,
.jp-CodeCell .jp-Cell-inputCollapser,
.jp-CodeCell .jp-Cell-outputCollapser,
.input_area,
.input .prompt,
.out_prompt_overlay,
.jp-OutputPrompt,
div[class*="CodeMirror"],
.highlight pre { display: none !important; }

/* Masquer les separateurs de cellules */
.jp-Cell-inputCollapser, .jp-Cell-outputCollapser { display: none !important; }

.jp-OutputArea, .output_area { margin: 0; padding: 0; }
.jp-OutputArea-output, .output_subarea { padding: 0 !important; overflow: visible !important; }

/* ── Markdown visible ── */
.jp-MarkdownCell .jp-Cell-inputWrapper,
.jp-MarkdownCell .jp-InputArea { display: block !important; }

.jp-MarkdownOutput, .text_cell_render { padding: 4px 0 !important; }

.jp-MarkdownOutput h1, .text_cell_render h1 {
  font-size: 19px; font-weight: 700; color: #1a1a2e;
  border-bottom: 2px solid #4a90d9; padding-bottom: 6px;
  margin: 36px 0 14px;
}
.jp-MarkdownOutput h2, .text_cell_render h2 {
  font-size: 15px; font-weight: 700; color: #1a1a2e;
  border-left: 4px solid #4a90d9; padding-left: 10px;
  margin: 26px 0 10px;
}
.jp-MarkdownOutput h3, .text_cell_render h3 {
  font-size: 13.5px; font-weight: 600; color: #2c3e6e;
  margin: 18px 0 6px;
}
.jp-MarkdownOutput p, .text_cell_render p {
  margin-bottom: 9px; text-align: justify;
}
.jp-MarkdownOutput ul, .jp-MarkdownOutput ol,
.text_cell_render ul, .text_cell_render ol {
  padding-left: 22px; margin-bottom: 9px;
}
.jp-MarkdownOutput li, .text_cell_render li { margin-bottom: 3px; }
.jp-MarkdownOutput strong, .text_cell_render strong { color: #1a1a2e; }
.jp-MarkdownOutput hr, .text_cell_render hr {
  border: none; border-top: 1px solid #e0e3ec; margin: 20px 0;
}

/* Bloc observation / note */
.jp-MarkdownOutput blockquote, .text_cell_render blockquote {
  background: #f0f4fb;
  border-left: 4px solid #4a90d9;
  padding: 10px 16px;
  margin: 12px 0;
  border-radius: 0 4px 4px 0;
  color: #2c3e6e;
}

/* ── Tableaux ── */
table {
  width: 100%; border-collapse: collapse;
  margin: 14px 0; font-size: 12.5px;
  background: #ffffff; border-radius: 5px;
  overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.09);
}
thead tr { background: #1a1a2e; color: #ffffff; }
th {
  padding: 9px 13px; text-align: left;
  font-weight: 600; font-size: 11.5px;
  letter-spacing: 0.4px; text-transform: uppercase;
}
td { padding: 8px 13px; border-bottom: 1px solid #e8eaf0; color: #1a1a2e; }
tbody tr:nth-child(even) { background: #f7f9fc; }
tbody tr:hover { background: #edf1fb; }

/* ── Images / graphiques ── */
.jp-OutputArea-output img, .output_png img, .output_jpeg img {
  max-width: 100%; height: auto; display: block;
  margin: 14px auto; border-radius: 3px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.09);
}

/* ── Sorties texte ── */
.jp-OutputArea-output pre, .output_text pre {
  background: #f3f5fb; border-left: 3px solid #4a90d9;
  border-radius: 3px; padding: 11px 15px;
  font-size: 11.5px; color: #2c3e6e;
  overflow-x: auto; white-space: pre-wrap;
  margin: 10px 0;
}

/* ── Pied de page ── */
.rapport-footer {
  background: #1a1a2e; color: #a8b8cc;
  text-align: center; padding: 20px 72px;
  font-size: 11px; border-top: 3px solid #4a90d9;
  border-radius: 0 0 2px 2px;
}
.rapport-footer span { margin: 0 14px; }

/* ── Impression ── */
@media print {
  body { background: #fff; }
  .page-a4 { box-shadow: none; margin: 0; width: 100%; }
  .rapport-header, .rapport-footer {
    -webkit-print-color-adjust: exact; print-color-adjust: exact;
  }
  .jp-Cell { page-break-inside: avoid; }
}
</style>
"""

HEADER_TPL = """
<div class="rapport-header">
  <h1>{titre}</h1>
  <div class="meta">
    <span>Type : Rapport d'analyse</span>
    <span>Date : {date}</span>
    <span>Statut : Confidentiel</span>
  </div>
</div>
"""

FOOTER_TPL = """
<div class="rapport-footer">
  <span>{date}</span>
  <span>&nbsp;|&nbsp;</span>
  <span>Usage interne &mdash; Diffusion restreinte</span>
</div>
"""

rapports = [
    {
        "notebook": "01_fraude.ipynb",
        "sortie": "rapport_detection_fraude.html",
        "titre": "Rapport &mdash; Detection de fraude bancaire",
    },
    {
        "notebook": "02_segmentation.ipynb",
        "sortie": "rapport_segmentation_client.html",
        "titre": "Rapport &mdash; Segmentation client",
    },
]

for r in rapports:
    nb_path = os.path.join(NOTEBOOKS, r["notebook"])
    out_path = os.path.join(OUTPUT, r["sortie"])

    # 1. Generer le HTML brut sans input
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "html",
        "--no-input",
        "--output", out_path,
        nb_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERREUR] {r['notebook']}\n{result.stderr[-2000:]}")
        continue

    # 2. Post-traitement : injecter CSS + header + footer
    with open(out_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Injecter CSS juste avant </head>
    html = html.replace("</head>", CSS + "\n</head>", 1)

    # Injecter header + ouvrir cadre A4 juste apres <body>
    body_tag_end = html.find(">", html.find("<body"))
    if body_tag_end != -1:
        html = (
            html[: body_tag_end + 1]
            + '\n<div class="page-a4">\n'
            + HEADER_TPL.format(titre=r["titre"], date=date_fr)
            + html[body_tag_end + 1 :]
        )

    # Fermer cadre A4 + injecter footer juste avant </body>
    html = html.replace(
        "</body>",
        FOOTER_TPL.format(date=date_fr) + "\n</div>\n</body>",
        1,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = round(os.path.getsize(out_path) / 1024, 1)
    print(f"[OK] {r['sortie']} ({size_kb} Ko)")
