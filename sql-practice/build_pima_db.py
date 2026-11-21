"""Reconstruit la base d'entrainement SQL (pima.db) a partir du CSV du repo.

Portable : marche depuis n'importe quel PC du moment que le repo est clone.
Usage :  python build_pima_db.py
Puis   :  sqlite3 pima.db      (dans ce meme dossier)
"""
import csv
import os
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "..", "pima-predict", "data", "pima.csv")
DB_PATH = os.path.join(HERE, "pima.db")

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

con = sqlite3.connect(DB_PATH)
cur = con.cursor()

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

cols = ", ".join(f'"{c}" REAL' for c in header)
cur.execute(f"CREATE TABLE pima ({cols})")
cur.executemany(
    f"INSERT INTO pima VALUES ({','.join('?' * len(header))})", rows
)
con.commit()

n = cur.execute("SELECT COUNT(*) FROM pima").fetchone()[0]
print(f"OK -> {DB_PATH}")
print(f"table 'pima' : {n} lignes, colonnes = {header}")
con.close()
