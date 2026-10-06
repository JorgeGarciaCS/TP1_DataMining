"""Verifica referencias contra la fuente primaria y guarda la evidencia en JSON.

- DOI: content negotiation en https://doi.org (sirve para Crossref y DataCite).
- arXiv: API oficial export.arxiv.org.
- JMLR: se descarga la página oficial del artículo y se busca el título.
Uso: python scripts/verify_references.py
"""
import json
import re
import datetime as dt
from pathlib import Path

import requests

OUT = Path("docs/jueces/ronda_1b/verificacion_referencias.json")

DOIS = {
    "REF-01": "10.24432/C5MC89",
    "REF-02": "10.3390/data7110146",
    "REF-03": "10.1016/j.patter.2023.100804",
    "REF-04": "10.1145/3458723",
    "REF-06": "10.1016/j.neuroimage.2017.06.061",
    "REF-07": "10.1371/journal.pone.0118432",
    "REF-09": "10.1016/j.patter.2024.101046",
    "REF-10": "10.1038/s42256-019-0048-x",
    "REF-11": "10.1145/3292500.3330701",
    "REF-13": "10.1145/2382577.2382579",
}
ARXIV = {"REF-04b": "1803.09010", "REF-12": "1705.07874", "REF-14": "2207.08815", "REF-09b": "2108.02497"}
JMLR = {"REF-05": ("https://jmlr.org/papers/v12/pedregosa11a.html", "Scikit-learn: Machine Learning in Python")}
SEARCH = {"REF-08": "The receiver operating characteristic curve accurately assesses imbalanced datasets"}


def doi_meta(doi):
    r = requests.get(f"https://doi.org/{doi}", headers={"Accept": "application/vnd.citationstyles.csl+json"}, timeout=30)
    r.raise_for_status()
    j = r.json()
    year = (j.get("issued") or {}).get("date-parts", [[None]])[0][0]
    authors = [f"{a.get('family', '')}, {a.get('given', '')}".strip(", ") for a in j.get("author", [])]
    return {"title": j.get("title"), "year": year, "authors": authors, "container": j.get("container-title"),
            "volume": j.get("volume"), "issue": j.get("issue"), "page": j.get("page"), "doi": doi}


def arxiv_meta(aid):
    r = requests.get(f"http://export.arxiv.org/api/query?id_list={aid}", timeout=30)
    r.raise_for_status()
    t = r.text
    entry = t[t.find("<entry>"):]
    title = re.search(r"<title>(.*?)</title>", entry, re.S).group(1)
    pub = re.search(r"<published>(\d{4})", entry).group(1)
    authors = re.findall(r"<name>(.*?)</name>", entry)
    return {"title": " ".join(title.split()), "year": int(pub), "authors": authors, "arxiv": aid}


def jmlr_meta(url, title):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return {"url": url, "title_found": title.lower() in r.text.lower(), "snippet": re.sub(r"\s+", " ", re.sub("<[^>]+>", " ", r.text))[:300]}


def crossref_search(q):
    r = requests.get("https://api.crossref.org/works", params={"query.title": q, "rows": 3}, timeout=30)
    r.raise_for_status()
    items = r.json()["message"]["items"]
    return [{"title": i.get("title"), "doi": i.get("DOI"), "container": i.get("container-title"),
             "year": (i.get("issued") or {}).get("date-parts", [[None]])[0][0],
             "authors": [f"{a.get('family', '')}, {a.get('given', '')}" for a in i.get("author", [])][:6]} for i in items]


def main():
    res = {"fecha": dt.datetime.now().isoformat(timespec="seconds"), "items": {}}
    for k, d in DOIS.items():
        try:
            res["items"][k] = doi_meta(d)
        except Exception as e:  # se registra el fallo, no se inventa
            res["items"][k] = {"error": str(e), "doi": d}
    for k, a in ARXIV.items():
        try:
            res["items"][k] = arxiv_meta(a)
        except Exception as e:
            res["items"][k] = {"error": str(e), "arxiv": a}
    for k, (u, t) in JMLR.items():
        try:
            res["items"][k] = jmlr_meta(u, t)
        except Exception as e:
            res["items"][k] = {"error": str(e), "url": u}
    for k, q in SEARCH.items():
        try:
            res["items"][k] = {"crossref_search": crossref_search(q)}
        except Exception as e:
            res["items"][k] = {"error": str(e)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in res["items"].items():
        if "error" in v:
            print(k, "ERROR", v["error"][:80])
        elif "crossref_search" in v:
            for i in v["crossref_search"]:
                print(k, "SEARCH", i["year"], i["doi"], (i["title"] or [""])[0][:90], i["authors"][:2])
        elif "title_found" in v:
            print(k, "JMLR title_found=", v["title_found"])
        else:
            t = v["title"] if isinstance(v["title"], str) else (v["title"] or [""])
            print(k, v["year"], "|", str(t)[:95], "|", v["authors"][:3], "|", v.get("container"), v.get("volume"), v.get("page"))


if __name__ == "__main__":
    main()
