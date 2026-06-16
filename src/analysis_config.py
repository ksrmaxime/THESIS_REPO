# src/analysis_config.py
"""Reference data for scripts/run_analysis.py.

Maps every literal `keyword` value produced by tag_keywords.py (i.e. every
entry of DEPARTMENTS / ADMIN_UNITS / INDEPENDENT_AGENCIES / COUNCILLORS in
src/download_src.py) to:
  - a target_type  ("Department" | "Admin Unit" | "Independent Agency" | "Minister")
  - a canonical parent department code (one of the 7 below), when applicable

Also provides the federal councillor -> party map (needed for E2/E3, absent
from the rest of the pipeline) and a parser for the SOURCE_CAT_STD tokens
produced by scripts/standardize_run5.py.
"""
from __future__ import annotations

import re

from src.run5_prompts import _COUNCIL_COMPOSITIONS, get_council_for_date

# ---------------------------------------------------------------------------
# Canonical department codes (identical to standardize_run5.DEPT_SINGLE_MAP
# values). "DEFR/WBF" is used as the canonical code for the whole 2000-2025
# span even though the department was named "DFE/EVD" before the late-2012
# reshuffle — the council-composition dicts in run5_prompts.py key it as
# "DFE/EVD" pre-2010 and "DEFR/WBF" from 2010 on, so the alias table below
# bridges the two when looking up the minister in office.
# ---------------------------------------------------------------------------
DEPARTMENT_CODES: list[str] = [
    "DFAE/EDA", "DFI/EDI", "DFJP/EJPD", "DDPS/VBS",
    "DFF/EFD", "DEFR/WBF", "DETEC/UVEK",
]

DEPT_LABELS: dict[str, str] = {
    "DFAE/EDA":   "Affaires étrangères (DFAE/EDA)",
    "DFI/EDI":    "Intérieur (DFI/EDI)",
    "DFJP/EJPD":  "Justice et police (DFJP/EJPD)",
    "DDPS/VBS":   "Défense (DDPS/VBS)",
    "DFF/EFD":    "Finances (DFF/EFD)",
    "DEFR/WBF":   "Économie, formation, recherche (DEFR/WBF)",
    "DETEC/UVEK": "Environnement, transports, énergie, communication (DETEC/UVEK)",
}

_DEPT_COMPOSITION_KEY_ALIASES: dict[str, list[str]] = {
    "DEFR/WBF": ["DEFR/WBF", "DFE/EVD"],
}


def minister_for_canonical_dept(dept_code: str, pubtime) -> str | None:
    """Return the name of the federal councillor heading *dept_code* at *pubtime*."""
    comp = get_council_for_date(pubtime)
    for key in _DEPT_COMPOSITION_KEY_ALIASES.get(dept_code, [dept_code]):
        if key in comp:
            return comp[key]
    return None


# ---------------------------------------------------------------------------
# keyword (DEPARTMENTS, src/download_src.py) -> canonical department code
# ---------------------------------------------------------------------------
DEPARTMENT_TERMS: dict[str, str] = {
    "VBS": "DDPS/VBS", "DDPS": "DDPS/VBS",
    "Eidgenössische Departement für Verteidigung, Bevölkerungsschutz und Sport": "DDPS/VBS",
    "Département fédéral de la défense, de la protection de la population et des sports": "DDPS/VBS",

    "EDA": "DFAE/EDA", "DFAE": "DFAE/EDA",
    "Eidgenössische Departement für auswärtige Angelegenheiten": "DFAE/EDA",
    "Département fédéral des affaires étrangères": "DFAE/EDA",

    "UVEK": "DETEC/UVEK", "DETEC": "DETEC/UVEK",
    "Eidgenössische Departement für Umwelt, Verkehr, Energie und Kommunikation": "DETEC/UVEK",
    "Département fédéral de l'environnement, des transports, de l'énergie et de la communication": "DETEC/UVEK",

    "EJPD": "DFJP/EJPD", "DFJP": "DFJP/EJPD",
    "Eidgenössische Justiz- und Polizeidepartement": "DFJP/EJPD",
    "Département fédéral de justice et police": "DFJP/EJPD",

    "EDI": "DFI/EDI", "DFI": "DFI/EDI",
    "Eidgenössische Departement des Innern": "DFI/EDI",
    "Département fédéral de l'intérieur": "DFI/EDI",

    "EFD": "DFF/EFD", "DFF": "DFF/EFD",
    "Eidgenössische Finanzdepartement": "DFF/EFD",
    "Département fédéral des finances": "DFF/EFD",

    "WBF": "DEFR/WBF", "DEFR": "DEFR/WBF",
    "Eidgenössische Departement für Wirtschaft, Bildung und Forschung": "DEFR/WBF",
    "Département fédéral de l'économie, de la formation et de la recherche": "DEFR/WBF",
}

# ---------------------------------------------------------------------------
# keyword (ADMIN_UNITS, src/download_src.py) -> canonical parent department code
# ---------------------------------------------------------------------------
ADMIN_UNIT_TERMS: dict[str, str] = {
    # ── DFAE / EDA ──────────────────────────────────────────────────────────
    "SG-DFAE": "DFAE/EDA", "GS-EDA": "DFAE/EDA",
    "Direction du droit international public": "DFAE/EDA", "DDIP": "DFAE/EDA",
    "Direktion für Völkerrecht": "DFAE/EDA",
    "Direction consulaire": "DFAE/EDA",
    "Konsularische Direktion": "DFAE/EDA",
    "Direction du développement et de la coopération": "DFAE/EDA", "DDC": "DFAE/EDA",
    "Direktion für Entwicklung und Zusammenarbeit": "DFAE/EDA", "DEZA": "DFAE/EDA",
    "Direction des ressources": "DFAE/EDA",
    "Direktion für Ressourcen": "DFAE/EDA",

    # ── DFI / EDI ───────────────────────────────────────────────────────────
    "SG-DFI": "DFI/EDI", "GS-EDI": "DFI/EDI",
    "Bureau fédéral de l'égalité entre femmes et hommes": "DFI/EDI", "BFEG": "DFI/EDI",
    "Eidgenössisches Büro für die Gleichstellung von Frau und Mann": "DFI/EDI", "EBG": "DFI/EDI",
    "Office fédéral de la culture": "DFI/EDI", "OFC": "DFI/EDI",
    "Bundesamt für Kultur": "DFI/EDI", "BAK": "DFI/EDI",
    "Archives fédérales suisses": "DFI/EDI", "AFS": "DFI/EDI",
    "Schweizerisches Bundesarchiv": "DFI/EDI",
    "Office fédéral de météorologie et de climatologie": "DFI/EDI", "MétéoSuisse": "DFI/EDI",
    "Bundesamt für Meteorologie und Klimatologie": "DFI/EDI", "MeteoSchweiz": "DFI/EDI",
    "Office fédéral de la santé publique": "DFI/EDI", "OFSP": "DFI/EDI",
    "Bundesamt für Gesundheit": "DFI/EDI", "BAG": "DFI/EDI",
    "Office fédéral de la sécurité alimentaire et des affaires vétérinaires": "DFI/EDI", "OSAV": "DFI/EDI",
    "Bundesamt für Lebensmittelsicherheit und Veterinärwesen": "DFI/EDI", "BLV": "DFI/EDI",
    "Office fédéral de la statistique": "DFI/EDI", "OFS": "DFI/EDI",
    "Bundesamt für Statistik": "DFI/EDI", "BFS": "DFI/EDI",
    "Office fédéral des assurances sociales": "DFI/EDI", "OFAS": "DFI/EDI",
    "Bundesamt für Sozialversicherungen": "DFI/EDI", "BSV": "DFI/EDI",

    # ── DFJP / EJPD ─────────────────────────────────────────────────────────
    "SG-DFJP": "DFJP/EJPD", "GS-EJPD": "DFJP/EJPD",
    "Secrétariat d'État aux migrations": "DFJP/EJPD", "SEM": "DFJP/EJPD",
    "Staatssekretariat für Migration": "DFJP/EJPD",
    "Office fédéral de la justice": "DFJP/EJPD", "OFJ": "DFJP/EJPD",
    "Bundesamt für Justiz": "DFJP/EJPD",
    "Office fédéral de la police": "DFJP/EJPD", "fedpol": "DFJP/EJPD",
    "Bundesamt für Polizei": "DFJP/EJPD",
    "Service Surveillance de la correspondance par poste et télécommunication": "DFJP/EJPD", "Service SCPT": "DFJP/EJPD",
    "Dienst Überwachung Post- und Fernmeldeverkehr": "DFJP/EJPD", "ÜPF": "DFJP/EJPD",

    # ── DDPS / VBS ──────────────────────────────────────────────────────────
    "SG-DDPS": "DDPS/VBS", "GS-VBS": "DDPS/VBS",
    "Office fédéral de la protection de la population": "DDPS/VBS", "OFPP": "DDPS/VBS",
    "Bundesamt für Bevölkerungsschutz": "DDPS/VBS", "BABS": "DDPS/VBS",
    "Office fédéral de l'armement": "DDPS/VBS", "armasuisse": "DDPS/VBS",
    "Bundesamt für Rüstung": "DDPS/VBS",
    "Office fédéral de topographie": "DDPS/VBS", "swisstopo": "DDPS/VBS",
    "Bundesamt für Landestopografie": "DDPS/VBS",
    "Office fédéral du sport": "DDPS/VBS", "OFSPO": "DDPS/VBS",
    "Bundesamt für Sport": "DDPS/VBS", "BASPO": "DDPS/VBS",
    "Office fédéral de la cybersécurité": "DDPS/VBS", "OFCS": "DDPS/VBS",
    "Bundesamt für Cybersicherheit": "DDPS/VBS", "BACS": "DDPS/VBS",
    "Secrétariat d'État à la politique de sécurité": "DDPS/VBS", "SEPOS": "DDPS/VBS",
    "Staatssekretariat für Sicherheitspolitik": "DDPS/VBS",
    "Armée suisse": "DDPS/VBS", "Schweizer Armee": "DDPS/VBS",
    "Service de renseignement de la Confédération": "DDPS/VBS", "SRC": "DDPS/VBS",
    "Nachrichtendienst des Bundes": "DDPS/VBS", "NDB": "DDPS/VBS",
    "Office de l'auditeur en chef": "DDPS/VBS", "OAC": "DDPS/VBS",
    "Oberauditorat": "DDPS/VBS",

    # ── DFF / EFD ───────────────────────────────────────────────────────────
    "SG-DFF": "DFF/EFD", "GS-EFD": "DFF/EFD",
    "Secrétariat d'État aux questions financières internationales": "DFF/EFD", "SFI": "DFF/EFD",
    "Staatssekretariat für internationale Finanzfragen": "DFF/EFD", "SIF": "DFF/EFD",
    "Administration fédérale des finances": "DFF/EFD", "AFF": "DFF/EFD",
    "Eidgenössische Finanzverwaltung": "DFF/EFD", "EFV": "DFF/EFD",
    "Office fédéral du personnel": "DFF/EFD", "OFPER": "DFF/EFD",
    "Eidgenössisches Personalamt": "DFF/EFD", "EPA": "DFF/EFD",
    "Administration fédérale des contributions": "DFF/EFD", "AFC": "DFF/EFD",
    "Eidgenössische Steuerverwaltung": "DFF/EFD", "ESTV": "DFF/EFD",
    "Office fédéral de la douane et de la sécurité des frontières": "DFF/EFD", "OFDF": "DFF/EFD",
    "Bundesamt für Zoll und Grenzsicherheit": "DFF/EFD", "BAZG": "DFF/EFD",
    "Office fédéral de l'informatique et de la télécommunication": "DFF/EFD", "OFIT": "DFF/EFD",
    "Bundesamt für Informatik und Telekommunikation": "DFF/EFD",
    "Office fédéral des constructions et de la logistique": "DFF/EFD", "OFCL": "DFF/EFD",
    "Bundesamt für Bauten und Logistik": "DFF/EFD", "BBL": "DFF/EFD",

    # ── DEFR / WBF ──────────────────────────────────────────────────────────
    "SG-DEFR": "DEFR/WBF", "GS-WBF": "DEFR/WBF",
    "Secrétariat d'État à l'économie": "DEFR/WBF", "SECO": "DEFR/WBF",
    "Staatssekretariat für Wirtschaft": "DEFR/WBF",
    "Secrétariat d'État à la formation, à la recherche et à l'innovation": "DEFR/WBF", "SEFRI": "DEFR/WBF",
    "Staatssekretariat für Bildung, Forschung und Innovation": "DEFR/WBF", "SBFI": "DEFR/WBF",
    "Office fédéral de l'agriculture": "DEFR/WBF", "OFAG": "DEFR/WBF",
    "Bundesamt für Landwirtschaft": "DEFR/WBF", "BLW": "DEFR/WBF",
    "Office fédéral pour l'approvisionnement économique du pays": "DEFR/WBF", "OFAE": "DEFR/WBF",
    "Bundesamt für wirtschaftliche Landesversorgung": "DEFR/WBF",
    "Office fédéral du logement": "DEFR/WBF", "OFL": "DEFR/WBF",
    "Bundesamt für Wohnungswesen": "DEFR/WBF", "BWO": "DEFR/WBF",
    "Office fédéral du service civil": "DEFR/WBF", "CIVI": "DEFR/WBF",
    "Bundesamt für Zivildienst": "DEFR/WBF", "ZIVI": "DEFR/WBF",

    # ── DETEC / UVEK ────────────────────────────────────────────────────────
    "SG-DETEC": "DETEC/UVEK", "GS-UVEK": "DETEC/UVEK",
    "Office fédéral des transports": "DETEC/UVEK", "OFT": "DETEC/UVEK",
    "Bundesamt für Verkehr": "DETEC/UVEK", "BAV": "DETEC/UVEK",
    "Office fédéral de l'aviation civile": "DETEC/UVEK", "OFAC": "DETEC/UVEK",
    "Bundesamt für Zivilluftfahrt": "DETEC/UVEK", "BAZL": "DETEC/UVEK",
    "Office fédéral de l'énergie": "DETEC/UVEK", "OFEN": "DETEC/UVEK",
    "Bundesamt für Energie": "DETEC/UVEK", "BFE": "DETEC/UVEK",
    "Office fédéral des routes": "DETEC/UVEK", "OFROU": "DETEC/UVEK",
    "Bundesamt für Strassen": "DETEC/UVEK", "ASTRA": "DETEC/UVEK",
    "Office fédéral de la communication": "DETEC/UVEK", "OFCOM": "DETEC/UVEK",
    "Bundesamt für Kommunikation": "DETEC/UVEK", "BAKOM": "DETEC/UVEK",
    "Office fédéral de l'environnement": "DETEC/UVEK", "OFEV": "DETEC/UVEK",
    "Bundesamt für Umwelt": "DETEC/UVEK", "BAFU": "DETEC/UVEK",
    "Office fédéral du développement territorial": "DETEC/UVEK", "ARE": "DETEC/UVEK",
    "Bundesamt für Raumentwicklung": "DETEC/UVEK",
}

# ---------------------------------------------------------------------------
# keyword (INDEPENDENT_AGENCIES, src/download_src.py) -> no parent department
# ---------------------------------------------------------------------------
INDEPENDENT_AGENCY_TERMS: set[str] = {
    "Inspection fédérale de la sécurité nucléaire", "IFSN",
    "Eidgenössisches Nuklearsicherheitsinspektorat", "ENSI",
    "Inspection fédérale des installations à courant fort",
    "Eidgenössisches Starkstrominspektorat", "ESTI",
    "Service suisse d'enquête de sécurité", "SESE",
    "Schweizerische Sicherheitsuntersuchungsstelle", "SUST",
    "Commission fédérale de l'électricité", "EICom",
    "Eidgenössische Elektrizitätskommission", "ElCom",
    "Commission fédérale de la communication", "ComCom",
    "Eidgenössische Kommunikationskommission",
    "Autorité indépendante d'examen des plaintes en matière de radio-télévision", "AIEP",
    "Unabhängige Beschwerdeinstanz für Radio und Fernsehen", "UBI",
    "Commission fédérale de la poste", "PostCom",
    "Eidgenössische Postkommission",
    "Commission des chemins de fer", "RailCom",
    "Kommission für den Eisenbahnverkehr",
    "Surveillance des prix", "SPR",
    "Preisüberwachung", "PUE",
    "Commission de la concurrence", "COMCO",
    "Wettbewerbskommission", "WEKO",
    "Domaine des Écoles polytechniques fédérales", "domaine des EPF",
    "Bereich der Eidgenössischen Technischen Hochschulen", "ETH-Bereich",
    "Haute école fédérale en formation professionnelle", "HEFP",
    "Eidgenössisches Hochschulinstitut für Berufsbildung", "EHB",
    "Agence suisse pour l'encouragement de l'innovation", "Innosuisse",
    "Schweizerische Agentur für Innovationsförderung",
    "Autorité fédérale de surveillance des marchés financiers", "FINMA",
    "Eidgenössische Finanzmarktaufsicht",
    "Contrôle fédéral des finances", "CDF",
    "Eidgenössische Finanzkontrolle", "EFK",
    "Caisse fédérale de pensions PUBLICA", "Pensionskasse des Bundes PUBLICA", "PUBLICA",
    "Institut fédéral de la propriété intellectuelle", "IPI",
    "Eidgenössisches Institut für Geistiges Eigentum", "IGE",
    "Institut fédéral de métrologie", "METAS",
    "Eidgenössisches Institut für Metrologie",
    "Institut suisse de droit comparé", "ISDC",
    "Schweizerisches Institut für Rechtsvergleichung", "SIR",
    "Autorité fédérale de surveillance en matière de révision", "ASR",
    "Eidgenössische Revisionsaufsichtsbehörde", "RAB",
    "Commission fédérale des maisons de jeu", "CFMJ",
    "Eidgenössische Spielbankenkommission", "ESBK",
    "Commission arbitrale fédérale pour la gestion de droits d'auteur et de droits voisins", "CAF",
    "Eidgenössische Schiedskommission für die Verwertung von Urheberrechten", "ESchK",
    "Commission nationale de prévention de la torture", "CNPT",
    "Nationale Kommission zur Verhütung von Folter", "NKVF",
    "Commission fédérale des migrations", "CFM",
    "Eidgenössische Migrationskommission", "EKM",
    "Institut suisse des produits thérapeutiques", "Swissmedic",
    "Schweizerisches Heilmittelinstitut",
    "Musée national suisse", "MNS",
    "Schweizerisches Nationalmuseum", "SNM",
    "Fondation suisse pour la culture Pro Helvetia",
    "Schweizerische Kulturstiftung Pro Helvetia", "Pro Helvetia",
}

# ---------------------------------------------------------------------------
# Federal councillors (src/download_src.py COUNCILLORS) -> party.
# Controlled vocabulary matches scripts/standardize_run5.py PARTIES, plus
# "BDP" for the two councillors who sat for that party (BDP merged into
# Die Mitte in 2021, after both had left office, so it is kept as its own
# label rather than folded into Die Mitte/Le Centre).
# ---------------------------------------------------------------------------
PARTIES: list[str] = [
    "Die Grünen/Les Verts",
    "Die Mitte/Le Centre",
    "SVP/UDC",
    "FDP/PLR",
    "GLP/PVL",
    "SP/PS",
]

FAR_RIGHT_PARTIES: set[str] = {"SVP/UDC"}

COUNCILLOR_PARTY: dict[str, str] = {
    "Joseph Deiss":            "Die Mitte/Le Centre",   # CVP/PDC at the time
    "Ruth Dreifuss":           "SP/PS",
    "Ruth Metzler-Arnold":     "Die Mitte/Le Centre",   # CVP/PDC at the time
    "Adolf Ogi":               "SVP/UDC",
    "Kaspar Villiger":         "FDP/PLR",
    "Pascal Couchepin":        "FDP/PLR",
    "Moritz Leuenberger":      "SP/PS",
    "Samuel Schmid":           "SVP/UDC",               # left for BDP in mid-2008, near end of term
    "Micheline Calmy-Rey":     "SP/PS",
    "Christoph Blocher":       "SVP/UDC",
    "Hans-Rudolf Merz":        "FDP/PLR",
    "Doris Leuthard":          "Die Mitte/Le Centre",   # CVP/PDC at the time
    "Eveline Widmer-Schlumpf": "BDP",                   # elected as SVP, founded BDP weeks later
    "Ueli Maurer":             "SVP/UDC",
    "Didier Burkhalter":       "FDP/PLR",
    "Simonetta Sommaruga":     "SP/PS",
    "Johann Schneider-Ammann": "FDP/PLR",
    "Alain Berset":            "SP/PS",
    "Guy Parmelin":            "SVP/UDC",
    "Ignazio Cassis":          "FDP/PLR",
    "Karin Keller-Sutter":     "FDP/PLR",
    "Viola Amherd":            "Die Mitte/Le Centre",   # CVP/PDC at the time
    "Elisabeth Baume-Schneider": "SP/PS",
    "Albert Rösti":            "SVP/UDC",
    "Beat Jans":               "SP/PS",
    "Martin Pfister":          "Die Mitte/Le Centre",
}


# ---------------------------------------------------------------------------
# keyword -> {target_type, parent_dept, councillor_name}
# ---------------------------------------------------------------------------
def classify_keyword(keyword: str, pubtime=None) -> dict:
    """Classify a `keyword` value (run3 target entity) into the paper's target typology."""
    if keyword in COUNCILLOR_PARTY:
        comp = get_council_for_date(pubtime) if pubtime is not None else {}
        dept = next((d for d, name in comp.items() if name == keyword), None)
        if dept and dept not in DEPARTMENT_CODES:
            # normalise the pre-2010 "DFE/EVD" label to its canonical code
            dept = "DEFR/WBF" if dept == "DFE/EVD" else dept
        return {
            "target_type": "Minister",
            "parent_dept": dept,
            "councillor_name": keyword,
        }
    if keyword in DEPARTMENT_TERMS:
        return {
            "target_type": "Department",
            "parent_dept": DEPARTMENT_TERMS[keyword],
            "councillor_name": None,
        }
    if keyword in ADMIN_UNIT_TERMS:
        return {
            "target_type": "Admin Unit",
            "parent_dept": ADMIN_UNIT_TERMS[keyword],
            "councillor_name": None,
        }
    if keyword in INDEPENDENT_AGENCY_TERMS:
        return {
            "target_type": "Independent Agency",
            "parent_dept": None,
            "councillor_name": None,
        }
    return {"target_type": "Unknown", "parent_dept": None, "councillor_name": None}


# ---------------------------------------------------------------------------
# SOURCE_CAT_STD token parser (scripts/standardize_run5.py output vocabulary)
# ---------------------------------------------------------------------------
_SIMPLE_SOURCE_CATEGORIES: set[str] = {
    "Civil Servant", "Interest Group", "Journalist", "General Public",
    "System", "City Entity", "Cantonal Entity", "State-Owned Company",
}

_CF_RE = re.compile(r"^CF \((.+)\)$")
_FD_RE = re.compile(r"^FD \((.+)\)$")
_PARL_PARTY_RE = re.compile(r"^Parliamentary \((.+)\)$")


def parse_source_token(token: str) -> dict:
    """Parse a single ' | '-separated token of SOURCE_CAT_STD into structured fields."""
    token = token.strip()
    empty = {"broad_category": "Other", "party": None, "councillor_name": None, "dept": None}
    if not token:
        return empty

    if token == "FC":
        return {"broad_category": "Federal Council", "party": None, "councillor_name": None, "dept": None}
    if token == "Parliamentarians":
        return {"broad_category": "Parliamentary", "party": None, "councillor_name": None, "dept": None}
    if token == "Federal Administration":
        return {"broad_category": "Federal Administration", "party": None, "councillor_name": None, "dept": None}

    m = _CF_RE.match(token)
    if m:
        inner = m.group(1)
        if " — " in inner:
            name, dept = (p.strip() for p in inner.split(" — ", 1))
        else:
            name, dept = inner.strip(), None
        return {
            "broad_category": "Federal Councillor",
            "party": COUNCILLOR_PARTY.get(name),
            "councillor_name": name,
            "dept": dept,
        }

    m = _FD_RE.match(token)
    if m:
        return {"broad_category": "Federal Department", "party": None, "councillor_name": None, "dept": m.group(1).strip()}

    m = _PARL_PARTY_RE.match(token)
    if m:
        return {"broad_category": "Parliamentary", "party": m.group(1).strip(), "councillor_name": None, "dept": None}

    if token in _SIMPLE_SOURCE_CATEGORIES:
        return {"broad_category": token, "party": None, "councillor_name": None, "dept": None}

    return empty
