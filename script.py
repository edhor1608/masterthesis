# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:42:08 2024

@author: edhor
"""
# Import der notwendigen Bibliotheken
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import logging
from scipy.stats import ttest_rel, ttest_ind
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.anova import AnovaRM
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

# ------------------- Logging-Konfiguration -------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Laden der Daten
logging.debug("Laden der Daten aus der CSV-Datei.")
data = pd.read_csv(
    "data.csv",
    sep=";",
    encoding="latin1",
    skiprows=[1],
)

# erstellte DataFrame mit Demografische Daten (hardcoded) [Alter, Geschlecht, Bildungsstand] Alter: (Min: 17, Max: 61, Median: 28), Geschlecht: (männlich: 14, weiblich: 5)
demo_data = pd.DataFrame(
    {
        "Alter": [
            26,
            28,
            22,
            25,
            21,
            28,
            26,
            17,
            29,
            35,
            28,
            49,
            52,
            25,
            47,
            18,
            34,
            61,
            28,
        ],
        "Geschlecht": [
            "m",
            "m",
            "w",
            "m",
            "w",
            "w",
            "m",
            "w",
            "m",
            "m",
            "w",
            "m",
            "m",
            "m",
            "m",
            "m",
            "m",
            "m",
            "m",
        ],
        "Bildungsstand": [
            "Bachelor",
            "Master",
            "Bachelor",
            "Abitur",
            "Abitur",
            "Abitur",
            "Bachelor",
            "Schüler",
            "Master",
            "Abitur",
            "Realschule",
            "Realschule",
            "Diplom",
            "Abitur",
            "Realschule",
            "Schüler",
            "Bachelor",
            "Abitur",
            "Abitur",
        ],
    }
)

# Erstelle DataFrame fuer die Analysedaten fuer den finalen Bericht
export_data = pd.DataFrame()

# Datenvorverarbeitung
caseBlacklistIds = [76, 38, 31]
logging.debug(f"Anzahl der urspruenglichen Datenzeilen: {len(data)}")

# Entfernen von Faellen mit ungueltigen IDs
data = data[~data["CASE"].isin(caseBlacklistIds)]
logging.debug(f"Anzahl der Datenzeilen nach Entfernen ungueltiger IDs: {len(data)}")

# Ersetzen von fehlerhaften Werten
data.loc[data["CASE"] == 101, "QUESTNNR"] = data.loc[
    data["CASE"] == 101, "QUESTNNR"
].replace("qnr4", "qnr6")

data.loc[data["CASE"] == 103, "QUESTNNR"] = data.loc[
    data["CASE"] == 103, "QUESTNNR"
].replace("qnr5", "qnr3")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# --------------------------------------------------------------
# ------------------- Vorbereitung -----------------------------
# --------------------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Mapping der Original-Spaltennamen zu aussagekraeftigen Namen
column_mappings = {
    # Persoenliche Codes
    "PC01_01": "PC_Vorname_Mutter",
    "PC02_01": "PC_Geburtstag_Mutter",
    "PC03_01": "PC_Eigener_Geburtstag",
    # City Car Driving (CD) Variablen
    "CD01_01": "CD_Fahrzeit",
    "CD01_02": "CD_Anzahl_Fussgaenger",
    "CD01_03": "CD_Anzahl_Gegenverkehr",
    "CD01_04": "CD_Anzahl_Spurwechsel",
    "CD01_05": "CD_Anzahl_Notbremsung",
    "CD01_06": "CD_Anzahl_Unfall",
    "CD01_07": "CD_Anzahl_Kontrollverlust",
    "CD01_08": "CD_Erfolg_Fussgaenger",
    "CD01_09": "CD_Erfolg_Gegenverkehr",
    "CD01_10": "CD_Erfolg_Spurwechsel",
    "CD01_11": "CD_Erfolg_Notbremsung",
    # CD02 Skalen-Items
    "CD02_01": "CD_Kurvenverhalten",
    "CD02_02": "CD_Stresslevel",
    "CD02_03": "CD_Anpassung_Verkehrsbedingungen",
    "CD02_04": "CD_Vorausschauendes_Fahren",
    "CD02_05": "CD_Reaktionszeiten",
    "CD02_06": "CD_Spurhaltung",
    "CD02_07": "CD_Spurwechselmanoever",
    "CD02_08": "CD_Stressbewaeltigung",
    # Nordschleife (NS) Variablen
    "NS02_01": "NS_Fahrzeit",
    "NS02_03": "NS_Anzahl_Unfall",
    "NS02_08": "NS_Anzahl_Kontrollverlust",
    "NS02_10": "NS_Unfall_verhindert",
    "NS02_06": "NS_Anzahl_Bremsfehler",
    "NS02_07": "NS_Anzahl_Beschleunigungsfehler",
    # NS01 Skalen-Items
    "NS01_01": "NS_Kurvenverhalten_Allgemein",
    "NS01_02": "NS_Stresslevel",
    "NS01_03": "NS_Anpassung_Wetterbedingungen",
    "NS01_04": "NS_Vorausschauendes_Fahren",
    "NS01_05": "NS_Reaktionszeiten",
    "NS01_09": "NS_Kurvenverhalten_Eng",
    "NS01_10": "NS_Kurvenverhalten_Schnell",
    "NS01_11": "NS_Fahrzeugkontrolle_Trocken",
    "NS01_12": "NS_Fahrzeugkontrolle_Nass",
    "NS01_13": "NS_Bremsen_Trocken",
    "NS01_14": "NS_Bremsen_Nass",
    "NS01_15": "NS_Beschleunigen_Trocken",
    "NS01_16": "NS_Beschleunigen_Nass",
    "NS01_17": "NS_Risikoverhalten",
    # ST Items
    "ST01": "ST_Praesentation",  # 1 = VR-Brille, 2 = Monitor
    "ST02": "ST_Eyetracking",  # 1 = Ja, 2 = Nein
    "ST03": "ST_Analyse",  # 1 = Ja, 2 = Nein
    # SF01 Selbstvertrauen beim Fahren (Skalen) # 1 = stimme gar nicht zu, 7 = stimme voll und ganz zu
    "SF01_01": "SF_Fuehle_mich_sicher_und_souveraen",
    "SF01_02": "SF_unterschiedliche_Fahrsituationen_bewaeltigen",
    "SF01_03": "SF_Ueberblick_in_komplexen_Fahrsituationen",
    "SF01_04": "SF_Gefahrensituationen_erkennen",
    "SF01_05": "SF_Fahrzeugbeherrschung_in_unerwarteten_Situationen",
    "SF01_06": "SF_Einschaezung_der_richtigen_Geschwindigkeit_und_Lenkbewegung",
    "SF01_07": "SF_Fahrweise_an_Witterungsbedingungen_anpassen",
    "SF01_08": "SF_Reaktion_auf_Hindernisse",
    "SF01_09": "SF_Sicher_und_souveraen_in_unterschiedlichen_Witterungsbedingungen",
    "SF01_10": "SF_Kennen_von_besonderheiten_unterschiedlicher_Antriebsarten",
    # SF02 Einschätzung von spezifischen Fahrsituationen (1-5) 1 = sehr unsicher, 5 = sehr sicher
    "SF02_01": "SF_Fahren_bei_Naesse",
    "SF02_02": "SF_Ueberholen_bei_hoher_Geschwindigkeit",
    "SF02_03": "SF_Fahren_bei_dichtem_Verkehr",
    "SF02_04": "SF_Fahren_in_engen_Kurven",
    "SF02_05": "SF_Fahren_schlechten_Sichtverhaeltnissen",
    # SF04 Veränderung Selbstvertrauen
    "SF04": "SF_Veraenderung_Selbstvertrauen",  # 1 = weniger Selbstvertrauen, 2 = gleichbleibend, 3 = mehr Selbstvertrauen
    # SF05 Veränderung Fahrtechnik
    "SF05": "SF_Verbaesserung_Fahrtechnik",  # 1 = Ja, Verbeserung, 2 = Unsicher, 3 = Nein
    # SF06 Anpassung an Wetterbedingungen
    "SF06": "SF_Anpassung_Wetterbedingungen",  # 1 = Ja, ich habe mich verbessert, 2 = Keine Veränderung, 3 = Nein, ich fühle mich unsicherer
    # SF07 Erfahrung mit Autofahren
    "SF07": "SF_Erfahrung_Autofahren",  # 1 = noch nie Auto gefahren, 2 = mache gerade meinen Führerschein, 3 = 1-3 Jahre, 4 = 4-10 Jahre, 5 = mehr als 10 Jahre
    # SF08 Trainingserfahrungen (Welche Trainings-/Ausbildungsmethode haben Sie bereits genutzt?)
    "SF08": "SF_Trainingserfahrungen",  # 1 = Fahrsicherheitstraining, 2 = Verkehrsübungsplätze, 3 = Rennlizenz, 4 = Fahrsimulator, 5 = Anderes:
    # SS01 Erfahrungen mit Simracing
    "SS01": "SS_Erfahrungen_Simracing",  # 1 = Ich bin weniger als 10 Std. gefahren, 2 = Ich bin 10-50 Std. gefahren, 3 = Ich bin mehr als 50 Std. gefahren
    # SS02 Faehigkeiten
    "SS02": "SS_Einschaetzung_Faehigkeiten",  # 1 = Sehr schlecht, 2 = Schlecht, 3 = Durchschnittlich, 4 = Gut, 5 = Sehr gut
    # SS03 Wahrnehmung von SimRacing als Trainingstool (1-5: stimme gar nicht zu - stimme voll und ganz zu)
    "SS03_01": "SS_SimRacing_kann_helfen_Fahrfaehigkeiten_zu_verbessern",
    "SS03_02": "SS_SimRacing_bietet_realistische_Fahrbedingungen",
    "SS03_03": "SS_SimRacing_ist_ein_gut_um_Reaktionszeiten_zu_verbessern",
    "SS03_04": "SS_SimRacing_ist_ein_gut_um_Lenktechnik_zu_verbessern",
    "SS03_05": "SS_SimRacing_ist_ein_gut_um_bei_hohen_Geschwindigkeiten_zurechtzukommen",
    "SS03_06": "SS_SimRacing_ist_ein_gut_um_bei_extremen_Wetterbedingungen_besser_umzugehen",
    # SS04 Relevanz für reales Fahren (1-5) 1 = keine, 5 = starke
    "SS04_01": "SS_Reaktionszeiten",
    "SS04_02": "SS_Kurvenfahren",
    "SS04_03": "SS_Bremsverhalten",
    "SS04_04": "SS_Verhalten_bei_schwierigen_Wetterbedingungen",
    # ET Einschätzung Training (1-5) 1 = stimme gar nicht zu, 5 = stimme voll zu
    # ET01 Akzeptanz und Bewertung des Trainings
    "ET01_01": "ET_SimRacing_Training_geholfen_Fahrfaehigkeiten_zu_verbessern",
    "ET01_02": "ET_SimRacing_Training_anderen_Personen_als_empfehlen",
    "ET01_03": "ET_SimRacing_Training_zufrieden",
    "ET01_04": "ET_SimRacing_Training_steigert_Motivation_zu_Ueben",
    "ET01_05": "ET_SimRacing_Training_als_nuetzlivh_fuer_reale_Fahrsituationen",
    "ET01_06": "ET_SimRacing_Training_hat_Faehigkeiten_zur_Konzentration_verbessert",
    # ET02 Einfluss des Präsentationsmediums (1-5) 1 = stimme gar nicht zu, 5 = stimme voll zu
    "ET02_01": "ET_Praesentationsmedium_erleichter_konzentration_auf_Training",
    "ET02_02": "ET_Praesentationsmedium_realitaetsnah",
    "ET02_03": "ET_Praesentationsmedium_Trainingsinhalte_verstaendlich",
    "ET02_04": "ET_Praesentationsmedium_bevorzugen_für_Training",
    # ET03 Immersion und Realitätsnähe (1-5) 1 = stimme gar nicht zu, 5 = stimme voll zu
    "ET03_01": "ET_Immersion_vollstaendig_eingetaucht",
    "ET03_02": "ET_Immersion_fahrpyhsik_realistisch",
    "ET03_03": "ET_Immersion_training_gefaesselt",
    "ET03_04": "ET_Immersion_realistisch_simulation_als_real",
    "ET03_05": "ET_Immersion_training_vermittelt_realistische_fahrersituationen",
    "ET03_06": "ET_Immersion_interaktivitaet_und_steuerung_verstaerkt_realismus",
    # ET04 Feedback und Lernfortschritt (1-5) 1 = stimme gar nicht zu, 5 = stimme voll zu
    "ET04_01": "ET_Feedback_hilfreich_um_Leistungen_zu_verbessern",
    "ET04_02": "ET_Feedback_klar_und_verstaendlich",
    "ET04_03": "ET_Feedback_Lernfortschritt_gut_nachvollziehbar",
    "ET04_04": "ET_Feedback_angeregt_Fahrverhalten_zu_ueberdenken",
    "ET04_05": "ET_Feedback_geholfen_an_Schwierigkeitsgrad_anzupassen",
    "ET04_06": "ET_Feedback_Fahrfaehigkeiten_durch_Training_kontinuierlich_verbessert",
    # ET05 Didaktik und Struktur (1-5) 1 = stimme gar nicht zu, 5 = stimme voll zu
    "ET05_01": "ET_Didaktik_lerninhalte_verstaendlich",
    "ET05_02": "ET_Didaktik_training_gut_strukturiert",
    "ET05_03": "ET_Didaktik_training_auf_persoenliche_Niveau_angepasst",
    "ET05_04": "ET_Didaktik_training_in_kleinen_Schritten",
    "ET05_05": "ET_Didaktik_gestaltung_hat_motivation_gesteigert",
    "ET05_06": "ET_Didaktik_balance_zwischen_theorie_praxis",
    "ET05_07": "ET_Didaktik_inhalte_verstaendlich_und_nachvollziehbar",
    "ET05_08": "ET_Didaktik_struktur_flexibel_um_individuelle_beduerfnisse",
}

# Umbenennen der Spalten im DataFrame
data.rename(columns=column_mappings, inplace=True)


def convert_to_numeric(series):
    series_converted = pd.to_numeric(
        series.astype(str).str.replace(",", "."), errors="coerce"
    )
    return series_converted


# Liste der numerischen Spalten, die konvertiert werden muessen
numeric_cols = [
    "CD_Fahrzeit",
    "NS_Fahrzeit",
]

# Konvertieren der numerischen Spalten
logging.debug("Konvertierung der numerischen Spalten zu numerischem Datentyp.")
for col in numeric_cols:
    if col in data.columns:
        data[col] = convert_to_numeric(data[col])
    else:
        logging.warning(f"Spalte {col} nicht in den Daten gefunden.")

# Erstellen des Identifikators
logging.debug("Erstellen des Personal Codes als eindeutigen Identifikator.")
data["Personal_Code"] = (
    (
        data["PC_Vorname_Mutter"].fillna("")
        + data["PC_Geburtstag_Mutter"].fillna("")
        + data["PC_Eigener_Geburtstag"].fillna("")
    )
    .astype(str)
    .str.strip()
    .str.lower()
)

# Ersetzen von -9 durch NaN in allen numerischen Variablen
logging.debug("Ersetzen von -9 durch NaN in allen numerischen Variablen.")
data = data.replace(-9, np.nan)

# Erstellen der Pre-Test- und Post-Test-Datensaetze
logging.debug("Erstellen der Pre-Test- und Post-Test-Datensaetze.")
pre_questionnaire = data[data["QUESTNNR"] == "qnr2"]
pre_test = data[data["QUESTNNR"] == "qnr3"]

# Post-Test: qnr5 (Testergebnisse) und qnr6 (Fragebogen)
post_test = data[data["QUESTNNR"] == "qnr5"]
post_questionnaire = data[data["QUESTNNR"] == "qnr6"]

# Zusammenfuehren der Pre-Test-Daten
logging.debug("Zusammenfuehren der Pre-Test-Daten.")
pre_combined = pd.merge(
    pre_questionnaire,
    pre_test,
    on="Personal_Code",
    suffixes=("_preQ", "_preT"),
)

# Zusammenfuehren der Post-Test-Daten
logging.debug("Zusammenfuehren der Post-Test-Daten.")
post_combined = pd.merge(
    post_questionnaire,
    post_test,
    on="Personal_Code",
    suffixes=("_postQ", "_postT"),
)

# Zusammenfuehren der Pre- und Post-Daten
logging.debug("Zusammenfuehren der Pre- und Post-Test-Daten.")
full_data = pd.merge(pre_combined, post_combined, on="Personal_Code")

# ueberpruefung der zusammengefuehrten Daten
logging.info(f"Anzahl der Teilnehmer mit vollstaendigen Daten: {len(full_data)}")

# ------------------- Settings -------------------

full_data["ST_Praesentation_post"] = full_data["ST_Praesentation_postQ"]

# with column not found in the data set NaN is filled in the column for the post
if "ST_Eyetracking_postQ" not in full_data.columns:
    full_data["ST_Eyetracking_postQ"] = np.nan
else:
    full_data["ST_Eyetracking_postQ"] = full_data["ST_Eyetracking_postQ"]

full_data["ST_Analyse_post"] = full_data["ST_Analyse_postQ"]

# ------------------- Definitionen -------------------

# Kilometer pro Runde Nordschleife
kilometerPerRoundNS = 20.832


# Funktion zur Berechnung der Quoten
def berechne_quote(numerator, denominator):
    valid_mask = (denominator != 0) & (~np.isnan(numerator)) & (~np.isnan(denominator))
    result = np.full_like(numerator, np.nan, dtype=np.double)
    result[valid_mask] = np.round(numerator[valid_mask] / denominator[valid_mask], 2)
    return result


# Funktion zur Berechnung der prozentualen Veränderung
def berechne_prozentuale_veraenderung(
    pre_values, post_values, diff_values, invert=False
):
    # Sicherstellen, dass die Eingaben Series sind
    pre_values = pd.Series(pre_values)
    diff_values = pd.Series(diff_values)

    # Berechnung des Ergebnisses unter Berücksichtigung von Division durch Null
    result = np.where(
        pre_values == 0,
        np.where(diff_values == 0, 0, np.where(diff_values > 0, np.inf, -np.inf)),
        (diff_values / pre_values.abs()) * 100,
    )

    result = pd.Series(result, index=pre_values.index)

    if invert:
        result = -result

    # Optional: Behandlung von Inf-Werten
    # result.replace([np.inf, -np.inf], np.nan, inplace=True)

    return result


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------- Generelle Statistiken -------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

# Demo-Statistiken
logging.debug("Berechnung der Demografie-Statistiken.")
demo_stats = demo_data.describe(include="all")
logging.info("Demografie-Statistiken:")
logging.info(demo_stats.to_string())

# Visualisierung der Demografie-Statistiken als seperate Plots
plt.figure(figsize=(8, 6))
sns.histplot(
    demo_data["Alter"],
    bins=5,
    color="forestgreen",
)
plt.title("Alter")
plt.xlabel("Alter")
plt.ylabel("Anzahl der Teilnehmer")
# plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(
    x="Geschlecht",
    data=demo_data,
    order=["m", "w"],
    palette=["lightgreen", "forestgreen"],
)
plt.title("Geschlecht")
plt.xlabel("Geschlecht")
plt.ylabel("Anzahl der Teilnehmer")
# plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(
    x="Bildungsstand",
    data=demo_data,
    order=["Schüler", "Realschule", "Abitur", "Bachelor", "Master", "Diplom"],
    palette=["lightgreen", "forestgreen"],
)
plt.title("Bildungsstand")
plt.xlabel("Bildungsstand")
plt.ylabel("Anzahl der Teilnehmer")
# plt.show()


# Gruppen Anzahlen
# Präsentationsmedium Gruppen: VR-Brille, Monitor ohne Eyetracking, Monitor mit Eyetracking
präsentationsmedium = {
    "VR-Brille": full_data["ST_Praesentation_post"].value_counts().get(1, 0),
    "Monitor ohne Eyetracking": full_data["ST_Eyetracking_postQ"]
    .value_counts()
    .get(2, 0),
    "Monitor mit Eyetracking": full_data["ST_Eyetracking_postQ"]
    .value_counts()
    .get(1, 0),
}

eyetracking_analyse = {
    "VR mit Analyse": full_data[
        (full_data["ST_Praesentation_post"] == 1) & (full_data["ST_Analyse_postQ"] == 1)
    ].shape[0],
    "VR ohne Analyse": full_data[
        (full_data["ST_Praesentation_post"] == 1) & (full_data["ST_Analyse_postQ"] == 2)
    ].shape[0],
}

# Visualisierung der Gruppenanzahlen
plt.figure(figsize=(10, 5))
sns.barplot(
    x=list(präsentationsmedium.keys()),
    y=list(präsentationsmedium.values()),
    palette=["lightgreen", "forestgreen", "darkgreen"],
)
plt.title("Präsentationsmedium")
plt.ylabel("Anzahl der Teilnehmer")
plt.ylim(0, 19)
plt.yticks(range(0, 20, 1))
# plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(
    x="ST_Analyse_post",
    data=full_data,
    palette=["lightgreen", "forestgreen"],
)
plt.ylim(0, 19)
plt.yticks(range(0, 20, 1))
plt.xticks([0, 1], ["Ja", "Nein"])
plt.xlabel("")
plt.ylabel("Anzahl der Teilnehmer")
plt.title("Analyse")
# plt.show()

# Fahrerfahrung
fahrerfahrung = {
    "noch nie Auto gefahren": full_data["SF_Erfahrung_Autofahren_postQ"]
    .value_counts()
    .get(1, 0),
    "gerade Führerschein": full_data["SF_Erfahrung_Autofahren_postQ"]
    .value_counts()
    .get(2, 0),
    "1-3 Jahre": full_data["SF_Erfahrung_Autofahren_postQ"].value_counts().get(3, 0),
    "4-10 Jahre": full_data["SF_Erfahrung_Autofahren_postQ"].value_counts().get(4, 0),
    "mehr als 10 Jahre": full_data["SF_Erfahrung_Autofahren_postQ"]
    .value_counts()
    .get(5, 0),
}

# Visualisierung der Fahrerfahrung
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.barplot(
    x=list(fahrerfahrung.keys()),
    y=list(fahrerfahrung.values()),
    ax=ax,
    palette=["lightgreen", "forestgreen", "darkgreen", "green"],
)
ax.set_title("Fahrerfahrung")
ax.set_ylabel("Anzahl der Teilnehmer")
# plt.show()

# Erfahrung mit Simracing
simracing_erfahrung = {
    "noch keine Erfahrung": full_data["SS_Erfahrungen_Simracing_preQ"]
    .value_counts()
    .get(-1, 0),
    "weniger als 10 Std.": full_data["SS_Erfahrungen_Simracing_preQ"]
    .value_counts()
    .get(2, 0),
    "10-50 Std.": full_data["SS_Erfahrungen_Simracing_preQ"].value_counts().get(3, 0),
    "mehr als 50 Std.": full_data["SS_Erfahrungen_Simracing_preQ"]
    .value_counts()
    .get(4, 0),
}

# Visualisierung der Erfahrung mit Simracing
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.barplot(
    x=list(simracing_erfahrung.keys()),
    y=list(simracing_erfahrung.values()),
    ax=ax,
    palette=["lightgreen", "forestgreen", "darkgreen", "green"],
)
ax.set_title("Erfahrung mit Simracing")
ax.set_ylabel("Anzahl der Teilnehmer")
# plt.show()

# Speichern der Demografie-Statistiken in csv-Datei
demo_stats.to_csv("demo_stats.csv")

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------- Berechnung der Differenzen -------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

logging.debug("Berechnung der Differenzen fuer City Car Driving (CD) Daten.")

# Erfolgquoten und Differenzen fuer City Car Driving (CD) Daten
# Beispiel: CD_Quote_Fussgaenger
full_data["CD_Quote_Fussgaenger_pre"] = berechne_quote(
    full_data["CD_Erfolg_Fussgaenger_preT"], full_data["CD_Anzahl_Fussgaenger_preT"]
)
full_data["CD_Quote_Fussgaenger_post"] = berechne_quote(
    full_data["CD_Erfolg_Fussgaenger_postT"], full_data["CD_Anzahl_Fussgaenger_postT"]
)
full_data["CD_Quote_Fussgaenger_diff"] = (
    full_data["CD_Quote_Fussgaenger_post"] - full_data["CD_Quote_Fussgaenger_pre"]
)
full_data["CD_Quote_Fussgaenger_post_mean"] = full_data[
    "CD_Quote_Fussgaenger_post"
].mean()

# Teilnehmer, fuer die die Differenz berechnet wurde
participants_with_diff = full_data.loc[
    full_data["CD_Quote_Fussgaenger_diff"].notna(),
    [
        "Personal_Code",
        "CD_Quote_Fussgaenger_pre",
        "CD_Quote_Fussgaenger_post",
        "CD_Quote_Fussgaenger_diff",
        "CD_Quote_Fussgaenger_post_mean",
    ],
]
logging.debug(
    f"CD_Quote_Fussgaenger_diff berechnet fuer {len(participants_with_diff)} Teilnehmer."
)
logging.info("Zwischenausgabe CD_Quote_Fussgaenger:")
logging.info(participants_with_diff.to_string(index=False))

# CD_Quote_Gegenverkehr
full_data["CD_Quote_Gegenverkehr_pre"] = berechne_quote(
    full_data["CD_Erfolg_Gegenverkehr_preT"], full_data["CD_Anzahl_Gegenverkehr_preT"]
)
full_data["CD_Quote_Gegenverkehr_post"] = berechne_quote(
    full_data["CD_Erfolg_Gegenverkehr_postT"], full_data["CD_Anzahl_Gegenverkehr_postT"]
)
full_data["CD_Quote_Gegenverkehr_diff"] = (
    full_data["CD_Quote_Gegenverkehr_post"] - full_data["CD_Quote_Gegenverkehr_pre"]
)
full_data["CD_Quote_Gegenverkehr_post_mean"] = full_data[
    "CD_Quote_Gegenverkehr_post"
].mean()

participants_with_diff = full_data.loc[
    full_data["CD_Quote_Gegenverkehr_diff"].notna(),
    [
        "Personal_Code",
        "CD_Quote_Gegenverkehr_pre",
        "CD_Quote_Gegenverkehr_post",
        "CD_Quote_Gegenverkehr_diff",
        "CD_Quote_Gegenverkehr_post_mean",
    ],
]
logging.debug(
    f"CD_Quote_Gegenverkehr_diff berechnet fuer {len(participants_with_diff)} Teilnehmer."
)
logging.info("Zwischenausgabe CD_Quote_Gegenverkehr:")
logging.info(participants_with_diff.to_string(index=False))

# CD_Quote_Spurwechsel
full_data["CD_Quote_Spurwechsel_pre"] = berechne_quote(
    full_data["CD_Erfolg_Spurwechsel_preT"], full_data["CD_Anzahl_Spurwechsel_preT"]
)
full_data["CD_Quote_Spurwechsel_post"] = berechne_quote(
    full_data["CD_Erfolg_Spurwechsel_postT"], full_data["CD_Anzahl_Spurwechsel_postT"]
)
full_data["CD_Quote_Spurwechsel_diff"] = (
    full_data["CD_Quote_Spurwechsel_post"] - full_data["CD_Quote_Spurwechsel_pre"]
)
full_data["CD_Quote_Spurwechsel_post_mean"] = full_data[
    "CD_Quote_Spurwechsel_post"
].mean()

participants_with_diff = full_data.loc[
    full_data["CD_Quote_Spurwechsel_diff"].notna(),
    [
        "Personal_Code",
        "CD_Quote_Spurwechsel_pre",
        "CD_Quote_Spurwechsel_post",
        "CD_Quote_Spurwechsel_diff",
        "CD_Quote_Spurwechsel_post_mean",
    ],
]
logging.debug(
    f"CD_Quote_Spurwechsel_diff berechnet fuer {len(participants_with_diff)} Teilnehmer."
)
logging.info("Zwischenausgabe CD_Quote_Spurwechsel:")
logging.info(participants_with_diff.to_string(index=False))

# CD_Quote_Notbremsung
full_data["CD_Quote_Notbremsung_pre"] = berechne_quote(
    full_data["CD_Erfolg_Notbremsung_preT"], full_data["CD_Anzahl_Notbremsung_preT"]
)
full_data["CD_Quote_Notbremsung_post"] = berechne_quote(
    full_data["CD_Erfolg_Notbremsung_postT"], full_data["CD_Anzahl_Notbremsung_postT"]
)
full_data["CD_Quote_Notbremsung_diff"] = (
    full_data["CD_Quote_Notbremsung_post"] - full_data["CD_Quote_Notbremsung_pre"]
)
full_data["CD_Quote_Notbremsung_post_mean"] = full_data[
    "CD_Quote_Notbremsung_post"
].mean()

participants_with_diff = full_data.loc[
    full_data["CD_Quote_Notbremsung_diff"].notna(),
    [
        "Personal_Code",
        "CD_Quote_Notbremsung_pre",
        "CD_Quote_Notbremsung_post",
        "CD_Quote_Notbremsung_diff",
        "CD_Quote_Notbremsung_post_mean",
    ],
]
logging.debug(
    f"CD_Quote_Notbremsung_diff berechnet fuer {len(participants_with_diff)} Teilnehmer."
)
logging.info("Zwischenausgabe CD_Quote_Notbremsung:")
logging.info(participants_with_diff.to_string(index=False))

# CD_Unfall_Diff und CD_Kontrollverlust_Diff
full_data["CD_Unfall_Diff"] = -1 * (
    full_data["CD_Anzahl_Unfall_postT"] - full_data["CD_Anzahl_Unfall_preT"]
)

full_data["CD_Unfall_pct_change"] = berechne_prozentuale_veraenderung(
    full_data["CD_Anzahl_Unfall_preT"],
    full_data["CD_Anzahl_Unfall_postT"],
    full_data["CD_Unfall_Diff"],
    invert=True,
)

participants_with_diff = full_data.loc[
    full_data["CD_Unfall_Diff"].notna(),
    [
        "Personal_Code",
        "CD_Anzahl_Unfall_preT",
        "CD_Anzahl_Unfall_postT",
        "CD_Unfall_Diff",
        "CD_Unfall_pct_change",
    ],
]
logging.debug(
    f"CD_Unfall_Diff berechnet fuer {len(participants_with_diff)} Teilnehmer."
)
logging.info("Zwischenausgabe CD_Unfall:")
logging.info(participants_with_diff.to_string(index=False))

full_data["CD_Kontrollverlust_Diff"] = -1 * (
    full_data["CD_Anzahl_Kontrollverlust_postT"]
    - full_data["CD_Anzahl_Kontrollverlust_preT"]
)
full_data["CD_Kontrollverlust_pct_change"] = berechne_prozentuale_veraenderung(
    full_data["CD_Anzahl_Kontrollverlust_preT"],
    full_data["CD_Anzahl_Kontrollverlust_postT"],
    full_data["CD_Kontrollverlust_Diff"],
    invert=True,
)

participants_with_diff = full_data.loc[
    full_data["CD_Kontrollverlust_Diff"].notna(),
    [
        "Personal_Code",
        "CD_Anzahl_Kontrollverlust_preT",
        "CD_Anzahl_Kontrollverlust_postT",
        "CD_Kontrollverlust_Diff",
        "CD_Kontrollverlust_pct_change",
    ],
]
logging.debug(
    f"CD_Kontrollverlust_Diff berechnet fuer {len(participants_with_diff)} Teilnehmer."
)
logging.info("Zwischenausgabe CD_Kontrollverlust:")
logging.info(participants_with_diff.to_string(index=False))

# CD_UnfallfreiProMinute
full_data["CD_UnfallfreiProMinute_pre"] = 1 - (
    full_data["CD_Anzahl_Unfall_preT"] / full_data["CD_Fahrzeit_preT"]
)
full_data["CD_UnfallfreiProMinute_post"] = 1 - (
    full_data["CD_Anzahl_Unfall_postT"] / full_data["CD_Fahrzeit_postT"]
)
full_data["CD_UnfallfreiProMinute_Diff"] = (
    full_data["CD_UnfallfreiProMinute_post"] - full_data["CD_UnfallfreiProMinute_pre"]
)
full_data["CD_UnfallfreiProMinute_post_mean"] = full_data[
    "CD_UnfallfreiProMinute_post"
].mean()

participants_with_diff = full_data.loc[
    full_data["CD_UnfallfreiProMinute_Diff"].notna(),
    [
        "Personal_Code",
        "CD_UnfallfreiProMinute_pre",
        "CD_UnfallfreiProMinute_post",
        "CD_UnfallfreiProMinute_Diff",
        "CD_UnfallfreiProMinute_post_mean",
    ],
]
logging.debug(
    f"CD_UnfallfreiProMinute_Diff berechnet fuer {len(participants_with_diff)} Teilnehmer."
)
logging.info("Zwischenausgabe CD_UnfallfreiProMinute:")
logging.info(participants_with_diff.to_string(index=False))

logging.debug("Berechnung der Differenzen fuer Nordschleife (NS) Daten.")

# NS_Fahrzeit_Diff
full_data["NS_Fahrzeit_pre"] = full_data["NS_Fahrzeit_preT"]
full_data["NS_Fahrzeit_post"] = full_data["NS_Fahrzeit_postT"]
full_data["NS_Fahrzeit_Diff"] = -1 * (
    full_data["NS_Fahrzeit_post"] - full_data["NS_Fahrzeit_pre"]
)
full_data["NS_Fahrzeit_pct_change"] = berechne_prozentuale_veraenderung(
    full_data["NS_Fahrzeit_pre"],
    full_data["NS_Fahrzeit_post"],
    full_data["NS_Fahrzeit_Diff"],
    invert=False,
)

participants_with_diff = full_data.loc[
    full_data["NS_Fahrzeit_Diff"].notna(),
    [
        "Personal_Code",
        "NS_Fahrzeit_pre",
        "NS_Fahrzeit_post",
        "NS_Fahrzeit_Diff",
        "NS_Fahrzeit_pct_change",
    ],
]
logging.debug(
    f"NS_Fahrzeit_Diff berechnet fuer {len(participants_with_diff)} Teilnehmer."
)
logging.info("Zwischenausgabe NS_Fahrzeit:")
logging.info(participants_with_diff.to_string(index=False))

# NS_Quote_UnfallVerhindert
full_data["NS_Quote_UnfallVerhindert_pre"] = berechne_quote(
    full_data["NS_Unfall_verhindert_preT"],
    full_data["NS_Anzahl_Unfall_preT"]
    + full_data["NS_Anzahl_Kontrollverlust_preT"]
    + full_data["NS_Unfall_verhindert_preT"],
)
full_data["NS_Quote_UnfallVerhindert_post"] = berechne_quote(
    full_data["NS_Unfall_verhindert_postT"],
    full_data["NS_Anzahl_Unfall_postT"]
    + full_data["NS_Anzahl_Kontrollverlust_postT"]
    + full_data["NS_Unfall_verhindert_postT"],
)
full_data["NS_Quote_UnfallVerhindert_Diff"] = (
    full_data["NS_Quote_UnfallVerhindert_post"]
    - full_data["NS_Quote_UnfallVerhindert_pre"]
)
full_data["NS_Quote_UnfallVerhindert_post_mean"] = full_data[
    "NS_Quote_UnfallVerhindert_post"
].mean()

participants_with_diff = full_data.loc[
    full_data["NS_Quote_UnfallVerhindert_Diff"].notna(),
    [
        "Personal_Code",
        "NS_Quote_UnfallVerhindert_pre",
        "NS_Quote_UnfallVerhindert_post",
        "NS_Quote_UnfallVerhindert_Diff",
        "NS_Quote_UnfallVerhindert_post_mean",
    ],
]
logging.debug(
    f"NS_Quote_UnfallVerhindert_Diff berechnet fuer {len(participants_with_diff)} Teilnehmer."
)
logging.info("Zwischenausgabe NS_Quote_UnfallVerhindert:")
logging.info(participants_with_diff.to_string(index=False))

# NS_Quote_UnfallFreiProKM
full_data["NS_Quote_UnfallFreiProKM_pre"] = 1 - (
    (full_data["NS_Anzahl_Unfall_preT"] + full_data["NS_Anzahl_Kontrollverlust_preT"])
    / kilometerPerRoundNS
)
full_data["NS_Quote_UnfallFreiProKM_post"] = 1 - (
    (full_data["NS_Anzahl_Unfall_postT"] + full_data["NS_Anzahl_Kontrollverlust_postT"])
    / kilometerPerRoundNS
)
full_data["NS_Quote_UnfallFreiProKM_Diff"] = (
    full_data["NS_Quote_UnfallFreiProKM_post"]
    - full_data["NS_Quote_UnfallFreiProKM_pre"]
)
full_data["NS_Quote_UnfallFreiProKM_post_mean"] = full_data[
    "NS_Quote_UnfallFreiProKM_post"
].mean()

participants_with_diff = full_data.loc[
    full_data["NS_Quote_UnfallFreiProKM_Diff"].notna(),
    [
        "Personal_Code",
        "NS_Quote_UnfallFreiProKM_pre",
        "NS_Quote_UnfallFreiProKM_post",
        "NS_Quote_UnfallFreiProKM_Diff",
        "NS_Quote_UnfallFreiProKM_post_mean",
    ],
]
logging.debug(
    f"NS_Quote_UnfallFreiProKM_Diff berechnet fuer {len(participants_with_diff)} Teilnehmer."
)
logging.info("Zwischenausgabe NS_Quote_UnfallFreiProKM:")
logging.info(participants_with_diff.to_string(index=False))

# NS_Quote_FehlerFreiProKM
full_data["NS_Quote_FehlerFreiProKM_pre"] = 1 - (
    (
        full_data["NS_Anzahl_Bremsfehler_preT"]
        + full_data["NS_Anzahl_Beschleunigungsfehler_preT"]
    )
    / kilometerPerRoundNS
)
full_data["NS_Quote_FehlerFreiProKM_post"] = 1 - (
    (
        full_data["NS_Anzahl_Bremsfehler_postT"]
        + full_data["NS_Anzahl_Beschleunigungsfehler_postT"]
    )
    / kilometerPerRoundNS
)
full_data["NS_Quote_FehlerFreiProKM_Diff"] = (
    full_data["NS_Quote_FehlerFreiProKM_post"]
    - full_data["NS_Quote_FehlerFreiProKM_pre"]
)
full_data["NS_Quote_FehlerFreiProKM_post_mean"] = full_data[
    "NS_Quote_FehlerFreiProKM_post"
].mean()

participants_with_diff = full_data.loc[
    full_data["NS_Quote_FehlerFreiProKM_Diff"].notna(),
    [
        "Personal_Code",
        "NS_Quote_FehlerFreiProKM_pre",
        "NS_Quote_FehlerFreiProKM_post",
        "NS_Quote_FehlerFreiProKM_Diff",
        "NS_Quote_FehlerFreiProKM_post_mean",
    ],
]
logging.debug(
    f"NS_Quote_FehlerFreiProKM_Diff berechnet fuer {len(participants_with_diff)} Teilnehmer."
)
logging.info("Zwischenausgabe NS_Quote_FehlerFreiProKM:")
logging.info(participants_with_diff.to_string(index=False))

# NS01 Skalen (Bewertung in Skala 1-7)
logging.debug("Berechnung der Differenzen fuer NS01 Skalenwerte.")

ns01_items = [
    "NS_Kurvenverhalten_Allgemein",
    "NS_Stresslevel",
    "NS_Anpassung_Wetterbedingungen",
    "NS_Vorausschauendes_Fahren",
    "NS_Reaktionszeiten",
    "NS_Kurvenverhalten_Eng",
    "NS_Kurvenverhalten_Schnell",
    "NS_Fahrzeugkontrolle_Trocken",
    "NS_Fahrzeugkontrolle_Nass",
    "NS_Bremsen_Trocken",
    "NS_Bremsen_Nass",
    "NS_Beschleunigen_Trocken",
    "NS_Beschleunigen_Nass",
    "NS_Risikoverhalten",
]

for item in ns01_items:
    pre_col = f"{item}_preT"
    post_col = f"{item}_postT"
    diff_col = f"{item}_Diff"
    full_data[diff_col] = full_data[post_col] - full_data[pre_col]
    participants_with_diff = full_data.loc[
        full_data[diff_col].notna(), ["Personal_Code", pre_col, post_col, diff_col]
    ]
    logging.debug(
        f"{diff_col} berechnet fuer {len(participants_with_diff)} Teilnehmer."
    )
    logging.info(f"Zwischenausgabe {item}:")
    logging.info(participants_with_diff.to_string(index=False))

# Aggregation der NS01 Differenzen in uebergreifende Kategorien
logging.debug("Aggregation der NS01 Differenzen in uebergreifende Kategorien.")

# Uebergreifende Skalen
full_data["NS_Kurvenverhalten_Uebergreifend_pre"] = full_data[
    [
        "NS_Kurvenverhalten_Allgemein_preT",
        "NS_Kurvenverhalten_Eng_preT",
        "NS_Kurvenverhalten_Schnell_preT",
    ]
].mean(axis=1)

full_data["NS_Kurvenverhalten_Uebergreifend_post"] = full_data[
    [
        "NS_Kurvenverhalten_Allgemein_postT",
        "NS_Kurvenverhalten_Eng_postT",
        "NS_Kurvenverhalten_Schnell_postT",
    ]
].mean(axis=1)

full_data["NS_Kurvenverhalten_Uebergreifend_Diff"] = full_data[
    [
        "NS_Kurvenverhalten_Allgemein_Diff",
        "NS_Kurvenverhalten_Eng_Diff",
        "NS_Kurvenverhalten_Schnell_Diff",
    ]
].mean(axis=1)
logging.debug("NS_Kurvenverhalten_Uebergreifend_Diff berechnet.")
logging.info("Zwischenausgabe NS_Kurvenverhalten_Uebergreifend_Diff:")
logging.info(
    full_data.loc[
        full_data["NS_Kurvenverhalten_Uebergreifend_Diff"].notna(),
        ["Personal_Code", "NS_Kurvenverhalten_Uebergreifend_Diff"],
    ].to_string(index=False)
)

full_data["NS_Fahrzeugkontrolle_Uebergreifend_pre"] = full_data[
    [
        "NS_Fahrzeugkontrolle_Trocken_preT",
        "NS_Fahrzeugkontrolle_Nass_preT",
    ]
].mean(axis=1)

full_data["NS_Fahrzeugkontrolle_Uebergreifend_post"] = full_data[
    [
        "NS_Fahrzeugkontrolle_Trocken_postT",
        "NS_Fahrzeugkontrolle_Nass_postT",
    ]
].mean(axis=1)

full_data["NS_Fahrzeugkontrolle_Uebergreifend_Diff"] = full_data[
    ["NS_Fahrzeugkontrolle_Trocken_Diff", "NS_Fahrzeugkontrolle_Nass_Diff"]
].mean(axis=1)
logging.debug("NS_Fahrzeugkontrolle_Uebergreifend_Diff berechnet.")
logging.info("Zwischenausgabe NS_Fahrzeugkontrolle_Uebergreifend_Diff:")
logging.info(
    full_data.loc[
        full_data["NS_Fahrzeugkontrolle_Uebergreifend_Diff"].notna(),
        ["Personal_Code", "NS_Fahrzeugkontrolle_Uebergreifend_Diff"],
    ].to_string(index=False)
)

full_data["NS_Beschleunigen_Uebergreifend_pre"] = full_data[
    [
        "NS_Beschleunigen_Trocken_preT",
        "NS_Beschleunigen_Nass_preT",
    ]
].mean(axis=1)

full_data["NS_Beschleunigen_Uebergreifend_post"] = full_data[
    [
        "NS_Beschleunigen_Trocken_postT",
        "NS_Beschleunigen_Nass_postT",
    ]
].mean(axis=1)

full_data["NS_Beschleunigen_Uebergreifend_Diff"] = full_data[
    ["NS_Beschleunigen_Trocken_Diff", "NS_Beschleunigen_Nass_Diff"]
].mean(axis=1)
logging.debug("NS_Beschleunigen_Uebergreifend_Diff berechnet.")
logging.info("Zwischenausgabe NS_Beschleunigen_Uebergreifend_Diff:")
logging.info(
    full_data.loc[
        full_data["NS_Beschleunigen_Uebergreifend_Diff"].notna(),
        ["Personal_Code", "NS_Beschleunigen_Uebergreifend_Diff"],
    ].to_string(index=False)
)

full_data["NS_Bremsen_Uebergreifend_pre"] = full_data[
    [
        "NS_Bremsen_Trocken_preT",
        "NS_Bremsen_Nass_preT",
    ]
].mean(axis=1)

full_data["NS_Bremsen_Uebergreifend_post"] = full_data[
    [
        "NS_Bremsen_Trocken_postT",
        "NS_Bremsen_Nass_postT",
    ]
].mean(axis=1)

full_data["NS_Bremsen_Uebergreifend_Diff"] = full_data[
    ["NS_Bremsen_Trocken_Diff", "NS_Bremsen_Nass_Diff"]
].mean(axis=1)
logging.debug("NS_Bremsen_Uebergreifend_Diff berechnet.")
logging.info("Zwischenausgabe NS_Bremsen_Uebergreifend_Diff:")
logging.info(
    full_data.loc[
        full_data["NS_Bremsen_Uebergreifend_Diff"].notna(),
        ["Personal_Code", "NS_Bremsen_Uebergreifend_Diff"],
    ].to_string(index=False)
)

full_data["NS_Trocken_Uebergreifend_pre"] = full_data[
    [
        "NS_Fahrzeugkontrolle_Trocken_preT",
        "NS_Bremsen_Trocken_preT",
        "NS_Beschleunigen_Trocken_preT",
    ]
].mean(axis=1)

full_data["NS_Nass_Uebergreifend_pre"] = full_data[
    [
        "NS_Fahrzeugkontrolle_Nass_preT",
        "NS_Bremsen_Nass_preT",
        "NS_Beschleunigen_Nass_preT",
    ]
].mean(axis=1)

full_data["NS_Trocken_Uebergreifend_Diff"] = full_data[
    [
        "NS_Fahrzeugkontrolle_Trocken_Diff",
        "NS_Bremsen_Trocken_Diff",
        "NS_Beschleunigen_Trocken_Diff",
    ]
].mean(axis=1)
logging.debug("NS_Trocken_Uebergreifend_Diff berechnet.")
logging.info("Zwischenausgabe NS_Trocken_Uebergreifend_Diff:")
logging.info(
    full_data.loc[
        full_data["NS_Trocken_Uebergreifend_Diff"].notna(),
        ["Personal_Code", "NS_Trocken_Uebergreifend_Diff"],
    ].to_string(index=False)
)

full_data["NS_Nass_Uebergreifend_pre"] = full_data[
    [
        "NS_Fahrzeugkontrolle_Nass_preT",
        "NS_Bremsen_Nass_preT",
        "NS_Beschleunigen_Nass_preT",
    ]
].mean(axis=1)

full_data["NS_Nass_Uebergreifend_post"] = full_data[
    [
        "NS_Fahrzeugkontrolle_Nass_postT",
        "NS_Bremsen_Nass_postT",
        "NS_Beschleunigen_Nass_postT",
    ]
].mean(axis=1)

full_data["NS_Nass_Uebergreifend_Diff"] = full_data[
    [
        "NS_Fahrzeugkontrolle_Nass_Diff",
        "NS_Bremsen_Nass_Diff",
        "NS_Beschleunigen_Nass_Diff",
    ]
].mean(axis=1)
logging.debug("NS_Nass_Uebergreifend_Diff berechnet.")
logging.info("Zwischenausgabe NS_Nass_Uebergreifend_Diff:")
logging.info(
    full_data.loc[
        full_data["NS_Nass_Uebergreifend_Diff"].notna(),
        ["Personal_Code", "NS_Nass_Uebergreifend_Diff"],
    ].to_string(index=False)
)

participants_with_diff = full_data.loc[
    full_data["NS_Fahrzeit_Diff"].notna(),
    [
        "Personal_Code",
        "NS_Fahrzeit_pre",
        "NS_Fahrzeit_post",
        "NS_Fahrzeit_Diff",
        "NS_Fahrzeit_pct_change",
    ],
]
logging.debug(
    f"NS_Fahrzeit_Diff berechnet fuer {len(participants_with_diff)} Teilnehmer."
)
logging.info("Zwischenausgabe NS_Fahrzeit:")
logging.info(participants_with_diff.to_string(index=False))


# CD02 Skalen (Bewertung in Skala 1-7)
logging.debug("Berechnung der Differenzen fuer CD02 Skalenwerte.")

test_diff_items = [
    "CD_Kurvenverhalten",
    "CD_Stresslevel",
    "CD_Anpassung_Verkehrsbedingungen",
    "CD_Vorausschauendes_Fahren",
    "CD_Reaktionszeiten",
    "CD_Spurhaltung",
    "CD_Spurwechselmanoever",
    "CD_Stressbewaeltigung",
    "NS_Kurvenverhalten_Allgemein",
    "NS_Stresslevel",
    "NS_Anpassung_Wetterbedingungen",
    "NS_Vorausschauendes_Fahren",
    "NS_Reaktionszeiten",
    "NS_Kurvenverhalten_Eng",
    "NS_Kurvenverhalten_Schnell",
    "NS_Fahrzeugkontrolle_Trocken",
    "NS_Fahrzeugkontrolle_Nass",
    "NS_Bremsen_Trocken",
    "NS_Bremsen_Nass",
    "NS_Beschleunigen_Trocken",
    "NS_Beschleunigen_Nass",
    "NS_Risikoverhalten",
]

for col in test_diff_items:
    pre_col = f"{col}_preT"
    post_col = f"{col}_postT"
    diff_col = f"{col}_Diff"
    full_data[diff_col] = full_data[post_col] - full_data[pre_col]
    participants_with_diff = full_data.loc[
        full_data[diff_col].notna(), ["Personal_Code", pre_col, post_col, diff_col]
    ]
    logging.debug(
        f"{diff_col} berechnet fuer {len(participants_with_diff)} Teilnehmer."
    )
    logging.debug(f"Zwischenausgabe {col}:")
    logging.debug(participants_with_diff.to_string(index=False))

quest_diff_items = [
    "SF_Fuehle_mich_sicher_und_souveraen",
    "SF_unterschiedliche_Fahrsituationen_bewaeltigen",
    "SF_Ueberblick_in_komplexen_Fahrsituationen",
    "SF_Gefahrensituationen_erkennen",
    "SF_Fahrzeugbeherrschung_in_unerwarteten_Situationen",
    "SF_Einschaezung_der_richtigen_Geschwindigkeit_und_Lenkbewegung",
    "SF_Fahrweise_an_Witterungsbedingungen_anpassen",
    "SF_Reaktion_auf_Hindernisse",
    "SF_Sicher_und_souveraen_in_unterschiedlichen_Witterungsbedingungen",
    "SF_Kennen_von_besonderheiten_unterschiedlicher_Antriebsarten",
    "SF_Fahren_bei_Naesse",
    "SF_Ueberholen_bei_hoher_Geschwindigkeit",
    "SF_Fahren_bei_dichtem_Verkehr",
    "SF_Fahren_in_engen_Kurven",
    "SF_Fahren_schlechten_Sichtverhaeltnissen",
    "SS_SimRacing_kann_helfen_Fahrfaehigkeiten_zu_verbessern",
    "SS_SimRacing_bietet_realistische_Fahrbedingungen",
    "SS_SimRacing_ist_ein_gut_um_Reaktionszeiten_zu_verbessern",
    "SS_SimRacing_ist_ein_gut_um_Lenktechnik_zu_verbessern",
    "SS_SimRacing_ist_ein_gut_um_bei_hohen_Geschwindigkeiten_zurechtzukommen",
    "SS_SimRacing_ist_ein_gut_um_bei_extremen_Wetterbedingungen_besser_umzugehen",
    "SS_Reaktionszeiten",
    "SS_Kurvenfahren",
    "SS_Bremsverhalten",
    "SS_Verhalten_bei_schwierigen_Wetterbedingungen",
]

for col in quest_diff_items:
    pre_col = f"{col}_preQ"
    post_col = f"{col}_postQ"
    diff_col = f"{col}_Diff"
    full_data[diff_col] = full_data[post_col] - full_data[pre_col]
    participants_with_diff = full_data.loc[
        full_data[diff_col].notna(), ["Personal_Code", pre_col, post_col, diff_col]
    ]
    logging.debug(
        f"{diff_col} berechnet fuer {len(participants_with_diff)} Teilnehmer."
    )
    logging.debug(f"Zwischenausgabe {col}:")
    logging.debug(participants_with_diff.to_string(index=False))

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ------------------ Compute Composite Variables ------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Fahrfähigkeiten Scales
fahrfaehigkeiten_scales_pre = [
    "CD_Kurvenverhalten_preT",
    "CD_Reaktionszeiten_preT",
    "CD_Spurhaltung_preT",
    "CD_Spurwechselmanoever_preT",
    "NS_Kurvenverhalten_Allgemein_preT",
    "NS_Kurvenverhalten_Eng_preT",
    "NS_Kurvenverhalten_Schnell_preT",
    "NS_Fahrzeugkontrolle_Trocken_preT",
    "NS_Fahrzeugkontrolle_Nass_preT",
    "NS_Beschleunigen_Trocken_preT",
    "NS_Beschleunigen_Nass_preT",
    "NS_Bremsen_Trocken_preT",
    "NS_Bremsen_Nass_preT",
    "NS_Anpassung_Wetterbedingungen_preT",
    "NS_Reaktionszeiten_preT",
    "SS_SimRacing_kann_helfen_Fahrfaehigkeiten_zu_verbessern_preQ",
    "SS_SimRacing_ist_ein_gut_um_Lenktechnik_zu_verbessern_preQ",
    "SS_SimRacing_ist_ein_gut_um_bei_hohen_Geschwindigkeiten_zurechtzukommen_preQ",
    "SS_SimRacing_ist_ein_gut_um_bei_extremen_Wetterbedingungen_besser_umzugehen_preQ",
]

fahrfaehigkeiten_scales_post = [
    col.replace("_preT", "_postT") for col in fahrfaehigkeiten_scales_pre
]

full_data["Fahrfaehigkeiten_Scales_pre"] = full_data[fahrfaehigkeiten_scales_pre].mean(
    axis=1
)
full_data["Fahrfaehigkeiten_Scales_post"] = full_data[
    fahrfaehigkeiten_scales_post
].mean(axis=1)
full_data["Fahrfaehigkeiten_Scales_Diff"] = (
    full_data["Fahrfaehigkeiten_Scales_post"] - full_data["Fahrfaehigkeiten_Scales_pre"]
)

# Fahrfähigkeiten Quotas
fahrfaehigkeiten_quoten_pre = [
    "NS_Quote_UnfallVerhindert_pre",
    "NS_Quote_UnfallFreiProKM_pre",
    "NS_Quote_FehlerFreiProKM_pre",
]

fahrfaehigkeiten_quoten_post = [
    col.replace("_pre", "_post") for col in fahrfaehigkeiten_quoten_pre
]

full_data["Fahrfaehigkeiten_Quoten_pre"] = full_data[fahrfaehigkeiten_quoten_pre].mean(
    axis=1
)
full_data["Fahrfaehigkeiten_Quoten_post"] = full_data[
    fahrfaehigkeiten_quoten_post
].mean(axis=1)
full_data["Fahrfaehigkeiten_Quoten_Diff"] = (
    full_data["Fahrfaehigkeiten_Quoten_post"] - full_data["Fahrfaehigkeiten_Quoten_pre"]
)

# Selbstvertrauen Scales
selbstvertrauen_scales_pre = [
    "SF_Fuehle_mich_sicher_und_souveraen_preQ",
    "SF_unterschiedliche_Fahrsituationen_bewaeltigen_preQ",
    "SF_Fahrzeugbeherrschung_in_unerwarteten_Situationen_preQ",
    "SF_Sicher_und_souveraen_in_unterschiedlichen_Witterungsbedingungen_preQ",
    "SF_Fahren_bei_Naesse_preQ",
    "SF_Ueberholen_bei_hoher_Geschwindigkeit_preQ",
    "SF_Fahren_bei_dichtem_Verkehr_preQ",
    "SF_Fahren_in_engen_Kurven_preQ",
    "SF_Fahren_schlechten_Sichtverhaeltnissen_preQ",
    "SF_Fahren_bei_Naesse_preQ",
    "SF_Ueberholen_bei_hoher_Geschwindigkeit_preQ",
    "SF_Fahren_bei_dichtem_Verkehr_preQ",
    "SF_Fahren_in_engen_Kurven_preQ",
    "SF_Fahren_schlechten_Sichtverhaeltnissen_preQ",
]

selbstvertrauen_scales_post = [
    col.replace("_preQ", "_postQ") for col in selbstvertrauen_scales_pre
]

full_data["Selbstvertrauen_Scales_pre"] = full_data[selbstvertrauen_scales_pre].mean(
    axis=1
)
full_data["Selbstvertrauen_Scales_post"] = full_data[selbstvertrauen_scales_post].mean(
    axis=1
)
full_data["Selbstvertrauen_Scales_Diff"] = (
    full_data["Selbstvertrauen_Scales_post"] - full_data["Selbstvertrauen_Scales_pre"]
)

# Straßenverkehr Performance Scales
strassenverkehr_performance_scales_pre = [
    "CD_Stresslevel_preT",
    "CD_Anpassung_Verkehrsbedingungen_preT",
    "CD_Vorausschauendes_Fahren_preT",
    "CD_Reaktionszeiten_preT",
    "CD_Spurhaltung_preT",
    "CD_Spurwechselmanoever_preT",
    "CD_Stressbewaeltigung_preT",
    "NS_Stresslevel_preT",
    "NS_Anpassung_Wetterbedingungen_preT",
    "NS_Vorausschauendes_Fahren_preT",
    "NS_Reaktionszeiten_preT",
    "NS_Risikoverhalten_preT",
]

strassenverkehr_performance_scales_post = [
    col.replace("_preT", "_postT") for col in strassenverkehr_performance_scales_pre
]

full_data["Strassenverkehr_Performance_Scales_pre"] = full_data[
    strassenverkehr_performance_scales_pre
].mean(axis=1)
full_data["Strassenverkehr_Performance_Scales_post"] = full_data[
    strassenverkehr_performance_scales_post
].mean(axis=1)
full_data["Strassenverkehr_Performance_Scales_Diff"] = (
    full_data["Strassenverkehr_Performance_Scales_post"]
    - full_data["Strassenverkehr_Performance_Scales_pre"]
)

# Straßenverkehr Performance Quoten
strassenverkehr_quoten_pre = (
    [
        "CD_Quote_Fussgaenger_pre",
        "CD_Quote_Gegenverkehr_pre",
        "CD_Quote_Spurwechsel_pre",
        "CD_Quote_Notbremsung_pre",
    ],
)

strassenverkehr_performance_quoten_pre = [
    "CD_Quote_Fussgaenger_pre",
    "CD_Quote_Gegenverkehr_pre",
    "CD_Quote_Spurwechsel_pre",
    "CD_Quote_Notbremsung_pre",
]

strassenverkehr_performance_quoten_post = [
    "CD_Quote_Fussgaenger_post",
    "CD_Quote_Gegenverkehr_post",
    "CD_Quote_Spurwechsel_post",
    "CD_Quote_Notbremsung_post",
]

full_data["Strassenverkehr_Performance_Quoten_pre"] = full_data[
    strassenverkehr_performance_quoten_pre
].mean(axis=1)
full_data["Strassenverkehr_Performance_Quoten_post"] = full_data[
    strassenverkehr_performance_quoten_post
].mean(axis=1)
full_data["Strassenverkehr_Performance_Quoten_Diff"] = (
    full_data["Strassenverkehr_Performance_Quoten_post"]
    - full_data["Strassenverkehr_Performance_Quoten_pre"]
)

# Nordschleife
nordschleife_pre = [
    "NS_Fahrzeit_pre",
]

nordschleife_post = [
    "NS_Fahrzeit_post",
]

full_data["Nordschleife_pre"] = full_data[nordschleife_pre].mean(axis=1)
full_data["Nordschleife_post"] = full_data[nordschleife_post].mean(axis=1)
full_data["Nordschleife_Diff"] = -1 * (
    full_data["Nordschleife_post"] - full_data["Nordschleife_pre"]
)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------- Reliability Analysis --------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Function to compute Cronbach's alpha
def cronbach_alpha(df):
    df = df.dropna()
    item_vars = df.var(axis=0, ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    n_items = df.shape[1]
    alpha = (n_items / (n_items - 1)) * (1 - (item_vars.sum() / total_var))
    return alpha


# Reliability analysis for Fahrfähigkeiten (Skalen) - Pre
items_fahrfaehigkeiten_pre = full_data[fahrfaehigkeiten_scales_pre]
alpha_fahrfaehigkeiten_pre = cronbach_alpha(items_fahrfaehigkeiten_pre)
print(
    f"Cronbach's alpha for Fahrfähigkeiten (Skalen) - Pre: {alpha_fahrfaehigkeiten_pre:.3f}"
)

# Reliability analysis for Fahrfähigkeiten (Skalen) - Post
items_fahrfaehigkeiten_post = full_data[fahrfaehigkeiten_scales_post]
alpha_fahrfaehigkeiten_post = cronbach_alpha(items_fahrfaehigkeiten_post)
print(
    f"Cronbach's alpha for Fahrfähigkeiten (Skalen) - Post: {alpha_fahrfaehigkeiten_post:.3f}"
)

# Reliability analysis for Selbstvertrauen (Skalen) - Pre
items_selbstvertrauen_pre = full_data[selbstvertrauen_scales_pre]
alpha_selbstvertrauen_pre = cronbach_alpha(items_selbstvertrauen_pre)
print(
    f"Cronbach's alpha for Selbstvertrauen (Skalen) - Pre: {alpha_selbstvertrauen_pre:.3f}"
)

# Reliability analysis for Selbstvertrauen (Skalen) - Post
items_selbstvertrauen_post = full_data[selbstvertrauen_scales_post]
alpha_selbstvertrauen_post = cronbach_alpha(items_selbstvertrauen_post)
print(
    f"Cronbach's alpha for Selbstvertrauen (Skalen) - Post: {alpha_selbstvertrauen_post:.3f}"
)

# Reliability analysis for Straßenverkehr Performance (Skalen) - Pre
items_strassen_pre = full_data[strassenverkehr_performance_scales_pre]
alpha_strassen_pre = cronbach_alpha(items_strassen_pre)
print(
    f"Cronbach's alpha for Straßenverkehr Performance (Skalen) - Pre: {alpha_strassen_pre:.3f}"
)

# Reliability analysis for Straßenverkehr Performance (Skalen) - Post
items_strassen_post = full_data[strassenverkehr_performance_scales_post]
alpha_strassen_post = cronbach_alpha(items_strassen_post)
print(
    f"Cronbach's alpha for Straßenverkehr Performance (Skalen) - Post: {alpha_strassen_post:.3f}"
)

# Speichern der berechneten Werte in CSV-Datei
reliability_data = {
    "Variable": [
        "Fahrfähigkeiten (Skalen) - Pre",
        "Fahrfähigkeiten (Skalen) - Post",
        "Selbstvertrauen (Skalen) - Pre",
        "Selbstvertrauen (Skalen) - Post",
        "Straßenverkehr Performance (Skalen) - Pre",
        "Straßenverkehr Performance (Skalen) - Post",
    ],
    "Cronbach's Alpha": [
        alpha_fahrfaehigkeiten_pre,
        alpha_fahrfaehigkeiten_post,
        alpha_selbstvertrauen_pre,
        alpha_selbstvertrauen_post,
        alpha_strassen_pre,
        alpha_strassen_post,
    ],
}

reliability_df = pd.DataFrame(reliability_data)
reliability_df.to_csv("reliability_analysis.csv", index=False)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ------------------- Exploratory Factor Analysis -----------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Prepare data for EFA - Fahrfähigkeiten (Skalen) - Pre
efa_items_fahr_pre = full_data[fahrfaehigkeiten_scales_pre].dropna()

# Check KMO Measure
kmo_all, kmo_model = calculate_kmo(efa_items_fahr_pre)
print(f"\nFahrfähigkeiten (Skalen) - Pre:")
print(f"KMO Measure: {kmo_model:.3f}")

# Bartlett's Test of Sphericity
chi_square_value, p_value = calculate_bartlett_sphericity(efa_items_fahr_pre)
print(f"Bartlett's Test: chi2 = {chi_square_value:.3f}, p = {p_value:.3f}")

# Determine the number of factors
fa = FactorAnalyzer()
fa.fit(efa_items_fahr_pre)
eigen_values, vectors = fa.get_eigenvalues()

# Create a Scree plot
plt.figure(figsize=(8, 6))
plt.scatter(range(1, efa_items_fahr_pre.shape[1] + 1), eigen_values)
plt.plot(range(1, efa_items_fahr_pre.shape[1] + 1), eigen_values)
plt.title("Scree Plot for Fahrfähigkeiten (Skalen) - Pre")
plt.xlabel("Factors")
plt.ylabel("Eigenvalue")
plt.grid()
# plt.show()

# Decide on the number of factors (e.g., factors with eigenvalue > 1)
n_factors = sum(eigen_values > 1)
print(f"Number of factors to extract: {n_factors}")

# Perform Factor Analysis
fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
fa.fit(efa_items_fahr_pre)

# Factor loadings
loadings = pd.DataFrame(fa.loadings_, index=efa_items_fahr_pre.columns)
print("\nFactor Loadings for Fahrfähigkeiten (Skalen) - Pre:")
print(loadings)

# Similarly, perform EFA for Straßenverkehr Performance (Skalen) - Pre
efa_items_strassen_pre = full_data[strassenverkehr_performance_scales_pre].dropna()

# KMO and Bartlett's Test
kmo_all, kmo_model = calculate_kmo(efa_items_strassen_pre)
print(f"\nStraßenverkehr Performance (Skalen) - Pre:")
print(f"KMO Measure: {kmo_model:.3f}")
chi_square_value, p_value = calculate_bartlett_sphericity(efa_items_strassen_pre)
print(f"Bartlett's Test: chi2 = {chi_square_value:.3f}, p = {p_value:.3f}")

# Factor Analysis
fa = FactorAnalyzer()
fa.fit(efa_items_strassen_pre)
eigen_values, vectors = fa.get_eigenvalues()

plt.figure(figsize=(8, 6))
plt.scatter(range(1, efa_items_strassen_pre.shape[1] + 1), eigen_values)
plt.plot(range(1, efa_items_strassen_pre.shape[1] + 1), eigen_values)
plt.title("Scree Plot for Straßenverkehr Performance (Skalen) - Pre")
plt.xlabel("Factors")
plt.ylabel("Eigenvalue")
plt.grid()
# plt.show()

n_factors = sum(eigen_values > 1)
print(f"Number of factors to extract: {n_factors}")

fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
fa.fit(efa_items_strassen_pre)

loadings = pd.DataFrame(fa.loadings_, index=efa_items_strassen_pre.columns)
print("\nFactor Loadings for Straßenverkehr Performance (Skalen) - Pre:")
print(loadings)

# Speichern der berechneten Werte in CSV-Datei
efa_data = {
    "Variable": [
        "Fahrfähigkeiten (Skalen) - Pre",
        "Straßenverkehr Performance (Skalen) - Pre",
    ],
    "KMO": [kmo_model, kmo_model],
    "Bartlett's Test (chi2)": [chi_square_value, chi_square_value],
    "Bartlett's Test (p)": [p_value, p_value],
    "Number of Factors": [n_factors, n_factors],
}

efa_df = pd.DataFrame(efa_data)
efa_df.to_csv("efa_analysis.csv", index=False)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------- Forschungsfragen Base -------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

general_PrePost = {
    "Composite_Variables": [
        (
            "Fahrfaehigkeiten_Scales_pre",
            "Fahrfaehigkeiten_Scales_post",
            "Fahrfähigkeiten (Skalen)",
        ),
        (
            "Fahrfaehigkeiten_Quoten_pre",
            "Fahrfaehigkeiten_Quoten_post",
            "Fahrfähigkeiten (Quoten)",
        ),
        (
            "Nordschleife_pre",
            "Nordschleife_post",
            "Nordschleife (Rundenzeit)",
        ),
        (
            "Selbstvertrauen_Scales_pre",
            "Selbstvertrauen_Scales_post",
            "Selbstvertrauen (Skalen)",
        ),
        (
            "Strassenverkehr_Performance_Scales_pre",
            "Strassenverkehr_Performance_Scales_post",
            "Straßenverkehr Performance (Skalen)",
        ),
        (
            "Strassenverkehr_Performance_Quoten_pre",
            "Strassenverkehr_Performance_Quoten_post",
            "Straßenverkehr Performance (Quoten)",
        ),
    ],
}


# Now, we can perform the analysis on these composite variables
def perform_composite_analysis(data, composite_vars):
    results = []
    for pre_var, post_var, name in composite_vars:
        # Select the relevant data
        data_anova = data[["Personal_Code", pre_var, post_var]]  # .dropna()

        # Calculate pre and post means
        mean_pre = data_anova[pre_var].mean()
        mean_post = data_anova[post_var].mean()
        mean_diff = mean_post - mean_pre

        # Calculate standard deviation of differences
        differences = data_anova[post_var] - data_anova[pre_var]
        std_diff = differences.std(ddof=1)

        # Calculate Cohen's d for paired samples
        cohen_d = mean_diff / std_diff if std_diff != 0 else np.nan

        # Determine direction of change
        if mean_diff > 0:
            direction = "improvement"
        elif mean_diff < 0:
            direction = "deterioration"
        else:
            direction = "no change"

        # Reshape data into long format for ANOVA
        data_long = pd.melt(
            data_anova,
            id_vars=["Personal_Code"],
            value_vars=[pre_var, post_var],
            var_name="Time_Point",
            value_name="Value",
        )

        # Map 'pre' and 'post' to numerical values
        data_long["Time"] = data_long["Time_Point"].str.extract("(pre|post)$")
        data_long["Time"] = data_long["Time"].map({"pre": 0, "post": 1})

        # Drop missing values
        data_long = data_long.dropna(subset=["Value", "Time"])

        # Convert Value to numeric if not already
        data_long["Value"] = pd.to_numeric(data_long["Value"], errors="coerce")

        # Ensure there are enough observations
        if data_long["Value"].isnull().all():
            print(f"\nNo valid data available for {name}. Skipping analysis.")
            continue

        # ------------------- Repeated Measures ANOVA -------------------
        try:
            anova = AnovaRM(data_long, "Value", "Personal_Code", within=["Time"]).fit()
            print(anova)
            f_value = anova.anova_table["F Value"][0]
            p_value = anova.anova_table["Pr > F"][0]
            df_num = anova.anova_table["Num DF"][0]
            df_den = anova.anova_table["Den DF"][0]
        except Exception as e:
            print(f"ANOVA failed for {name}: {e}")
            continue

        # ------------------- Visualization -------------------
        plt.figure(figsize=(6, 4))
        sns.boxplot(
            x="Time",
            y="Value",
            data=data_long,
            palette=["lightgreen", "forestgreen"],
        )
        plt.title(f"{name}")
        plt.xticks(ticks=[0, 1], labels=["Pre", "Post"])
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.tight_layout()

        # ------------------- Print Results -------------------
        print(f"\nResults for {name}:")
        print(f"Mean Pre-Training: {mean_pre:.3f}")
        print(f"Mean Post-Training: {mean_post:.3f}")
        print(f"Mean Difference: {mean_diff:.3f} ({direction})")
        print(f"Standard Deviation of Differences: {std_diff:.3f}")
        print(f"Cohen's d: {cohen_d:.3f}")
        print(
            f"ANOVA F({int(df_num)}, {int(df_den)}): {f_value:.4f}, p = {p_value:.4f}"
        )
        # print(f"Partial Eta Squared: {partial_eta_squared:.3f}")

        # Append results to the list
        results.append(
            {
                "Variable": name,
                "Mean_Pre": mean_pre,
                "Mean_Post": mean_post,
                "Mean_Diff": mean_diff,
                "Direction": direction,
                "Std_Diff": std_diff,
                "Cohen_d": cohen_d,
                "F_Value": f_value,
                "p_Value": p_value,
            }
        )

        # plt.show()

    return pd.DataFrame(results)


# Prepare the composite variables list
composite_vars = general_PrePost["Composite_Variables"]

# Perform the analysis
results_composite = perform_composite_analysis(full_data, composite_vars)

# Save the results to a CSV file
results_composite.to_csv("results_base.csv", index=False)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------- Forschungsfragen 1 ----------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Spezielle Fahrfaehigkeiten Scales
spezielle_fahrfaehigkeiten_beschleunigen_scales_pre = [
    "NS_Beschleunigen_Trocken_preT",
    "NS_Beschleunigen_Nass_preT",
]

spezielle_fahrfaehigkeiten_beschleunigen_scales_post = [
    col.replace("_preT", "_postT")
    for col in spezielle_fahrfaehigkeiten_beschleunigen_scales_pre
]

spezielle_fahrfaehigkeiten_bremsen_scales_pre = [
    "NS_Bremsen_Trocken_preT",
    "NS_Bremsen_Nass_preT",
]

spezielle_fahrfaehigkeiten_bremsen_scales_post = [
    col.replace("_preT", "_postT")
    for col in spezielle_fahrfaehigkeiten_bremsen_scales_pre
]

spezielle_fahrfaehigkeiten_kurven_scales_pre = [
    "NS_Kurvenverhalten_Allgemein_preT",
    "NS_Kurvenverhalten_Eng_preT",
    "NS_Kurvenverhalten_Schnell_preT",
]

spezielle_fahrfaehigkeiten_kurven_scales_post = [
    col.replace("_preT", "_postT")
    for col in spezielle_fahrfaehigkeiten_kurven_scales_pre
]

spezielle_fahrfaehigkeiten_fahrzeugkontrolle_scales_pre = [
    "NS_Fahrzeugkontrolle_Trocken_preT",
    "NS_Fahrzeugkontrolle_Nass_preT",
]

spezielle_fahrfaehigkeiten_fahrzeugkontrolle_scales_post = [
    col.replace("_preT", "_postT")
    for col in spezielle_fahrfaehigkeiten_fahrzeugkontrolle_scales_pre
]

full_data["Spezielle_Fahrfaehigkeiten_Bremsen_Scales_pre"] = full_data[
    spezielle_fahrfaehigkeiten_bremsen_scales_pre
].mean(axis=1)
full_data["Spezielle_Fahrfaehigkeiten_Bremsen_Scales_post"] = full_data[
    spezielle_fahrfaehigkeiten_bremsen_scales_post
].mean(axis=1)
full_data["Spezielle_Fahrfaehigkeiten_Bremsen_Scales_Diff"] = (
    full_data["Spezielle_Fahrfaehigkeiten_Bremsen_Scales_post"]
    - full_data["Spezielle_Fahrfaehigkeiten_Bremsen_Scales_pre"]
)

full_data["Spezielle_Fahrfaehigkeiten_Kurven_Scales_pre"] = full_data[
    spezielle_fahrfaehigkeiten_kurven_scales_pre
].mean(axis=1)
full_data["Spezielle_Fahrfaehigkeiten_Kurven_Scales_post"] = full_data[
    spezielle_fahrfaehigkeiten_kurven_scales_post
].mean(axis=1)
full_data["Spezielle_Fahrfaehigkeiten_Kurven_Scales_Diff"] = (
    full_data["Spezielle_Fahrfaehigkeiten_Kurven_Scales_post"]
    - full_data["Spezielle_Fahrfaehigkeiten_Kurven_Scales_pre"]
)

full_data["Spezielle_Fahrfaehigkeiten_Beschleunigen_Scales_pre"] = full_data[
    spezielle_fahrfaehigkeiten_beschleunigen_scales_pre
].mean(axis=1)
full_data["Spezielle_Fahrfaehigkeiten_Beschleunigen_Scales_post"] = full_data[
    spezielle_fahrfaehigkeiten_beschleunigen_scales_post
].mean(axis=1)
full_data["Spezielle_Fahrfaehigkeiten_Beschleunigen_Scales_Diff"] = (
    full_data["Spezielle_Fahrfaehigkeiten_Beschleunigen_Scales_post"]
    - full_data["Spezielle_Fahrfaehigkeiten_Beschleunigen_Scales_pre"]
)

full_data["Spezielle_Fahrfaehigkeiten_Fahrzeugkontrolle_Scales_pre"] = full_data[
    spezielle_fahrfaehigkeiten_fahrzeugkontrolle_scales_pre
].mean(axis=1)
full_data["Spezielle_Fahrfaehigkeiten_Fahrzeugkontrolle_Scales_post"] = full_data[
    spezielle_fahrfaehigkeiten_fahrzeugkontrolle_scales_post
].mean(axis=1)
full_data["Spezielle_Fahrfaehigkeiten_Fahrzeugkontrolle_Scales_Diff"] = (
    full_data["Spezielle_Fahrfaehigkeiten_Fahrzeugkontrolle_Scales_post"]
    - full_data["Spezielle_Fahrfaehigkeiten_Fahrzeugkontrolle_Scales_pre"]
)

ff1_PrePost = {
    "Composite_Variables": [
        (
            "Spezielle_Fahrfaehigkeiten_Beschleunigen_Scales_pre",
            "Spezielle_Fahrfaehigkeiten_Beschleunigen_Scales_post",
            "Spezielle Fahrfaehigkeiten (Beschleunigen)",
        ),
        (
            "Spezielle_Fahrfaehigkeiten_Bremsen_Scales_pre",
            "Spezielle_Fahrfaehigkeiten_Bremsen_Scales_post",
            "Spezielle Fahrfaehigkeiten (Bremsen)",
        ),
        (
            "Spezielle_Fahrfaehigkeiten_Kurven_Scales_pre",
            "Spezielle_Fahrfaehigkeiten_Kurven_Scales_post",
            "Spezielle Fahrfaehigkeiten (Kurven)",
        ),
        (
            "Spezielle_Fahrfaehigkeiten_Fahrzeugkontrolle_Scales_pre",
            "Spezielle_Fahrfaehigkeiten_Fahrzeugkontrolle_Scales_post",
            "Spezielle Fahrfaehigkeiten (Fahrzeugkontrolle)",
        ),
    ],
}

# Prepare the composite variables list
composite_vars = ff1_PrePost["Composite_Variables"]

# Perform the analysis
results_composite = perform_composite_analysis(full_data, composite_vars)

# Save the results to a CSV file
results_composite.to_csv("results_ff1.csv", index=False)

# Zusätzliche Auswertungen für Forschungsfrage 2
data_forschungsfrage_5_auswahl = {
    "SF_Veraenderung_Selbstvertrauen": full_data[
        "SF_Veraenderung_Selbstvertrauen_postQ"
    ].replace({1: 1, 2: 0, 3: -1}),
    "SF_Verbaesserung_Fahrtechnik": full_data[
        "SF_Verbaesserung_Fahrtechnik_postQ"
    ].replace({1: 1, 2: 0.5, 3: 0}),
    "SF_Anpassung_Wetterbedingungen": full_data[
        "SF_Anpassung_Wetterbedingungen_postQ"
    ].replace({1: 1, 2: 0, 3: -1}),
}

# Boxplots der data_forschungsfrage_5 Auswahl Items
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=data_forschungsfrage_5_auswahl,
    palette=[
        "lightgreen",
        "mediumseagreen",
        "forestgreen",
    ],
)
plt.title("Auswahl Items (FF2)")
plt.ylabel("Skalenpunkte")
plt.yticks(np.arange(-1, 2, 1))
plt.text(-0.5, -0.9, "weniger Selbstvertrauen")
plt.text(-0.5, 1.1, "mehr Selbstvertrauen")
plt.xticks(rotation=90)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------- Forschungsfragen 3 ----------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Spezielle Straßenverkehr Performance Scales
spezielle_strassenverkehr_performance_stresslevel_scales_pre = [
    "CD_Stresslevel_preT",
    "NS_Stresslevel_preT",
]

spezielle_strassenverkehr_performance_stresslevel_scales_post = [
    col.replace("_preT", "_postT")
    for col in spezielle_strassenverkehr_performance_stresslevel_scales_pre
]

spezielle_strassenverkehr_performance_anpassung_scales_pre = [
    "CD_Anpassung_Verkehrsbedingungen_preT",
    "NS_Anpassung_Wetterbedingungen_preT",
    "NS_Vorausschauendes_Fahren_preT",
    "NS_Risikoverhalten_preT",
]

spezielle_strassenverkehr_performance_anpassung_scales_post = [
    col.replace("_preT", "_postT")
    for col in spezielle_strassenverkehr_performance_anpassung_scales_pre
]

spezielle_strassenverkehr_performance_reaktion_scales_pre = [
    "CD_Reaktionszeiten_preT",
    "NS_Reaktionszeiten_preT",
]

spezielle_strassenverkehr_performance_reaktion_scales_post = [
    col.replace("_preT", "_postT")
    for col in spezielle_strassenverkehr_performance_reaktion_scales_pre
]

spezielle_strassenverkehr_performance_reaktion_quoten_pre = [
    "CD_Quote_Fussgaenger_pre",
    "CD_Quote_Gegenverkehr_pre",
    "CD_Quote_Spurwechsel_pre",
    "CD_Quote_Notbremsung_pre",
]

spezielle_strassenverkehr_performance_reaktion_quoten_post = [
    col.replace("_pre", "_post")
    for col in spezielle_strassenverkehr_performance_reaktion_quoten_pre
]

full_data["Spezielle_Strassenverkehr_Performance_Reaktion_Quoten_pre"] = full_data[
    spezielle_strassenverkehr_performance_reaktion_quoten_pre
].mean(axis=1)
full_data["Spezielle_Strassenverkehr_Performance_Reaktion_Quoten_post"] = full_data[
    spezielle_strassenverkehr_performance_reaktion_quoten_post
].mean(axis=1)
full_data["Spezielle_Strassenverkehr_Performance_Reaktion_Quoten_Diff"] = (
    full_data["Spezielle_Strassenverkehr_Performance_Reaktion_Quoten_post"]
    - full_data["Spezielle_Strassenverkehr_Performance_Reaktion_Quoten_pre"]
)

full_data["Spezielle_Strassenverkehr_Performance_Reaktion_Scales_pre"] = full_data[
    spezielle_strassenverkehr_performance_reaktion_scales_pre
].mean(axis=1)
full_data["Spezielle_Strassenverkehr_Performance_Reaktion_Scales_post"] = full_data[
    spezielle_strassenverkehr_performance_reaktion_scales_post
].mean(axis=1)
full_data["Spezielle_Strassenverkehr_Performance_Reaktion_Scales_Diff"] = (
    full_data["Spezielle_Strassenverkehr_Performance_Reaktion_Scales_post"]
    - full_data["Spezielle_Strassenverkehr_Performance_Reaktion_Scales_pre"]
)

full_data["Spezielle_Strassenverkehr_Performance_Stresslevel_Scales_pre"] = full_data[
    spezielle_strassenverkehr_performance_stresslevel_scales_pre
].mean(axis=1)
full_data["Spezielle_Strassenverkehr_Performance_Stresslevel_Scales_post"] = full_data[
    spezielle_strassenverkehr_performance_stresslevel_scales_post
].mean(axis=1)
full_data["Spezielle_Strassenverkehr_Performance_Stresslevel_Scales_Diff"] = (
    full_data["Spezielle_Strassenverkehr_Performance_Stresslevel_Scales_post"]
    - full_data["Spezielle_Strassenverkehr_Performance_Stresslevel_Scales_pre"]
)

full_data["Spezielle_Strassenverkehr_Performance_Anpassung_Scales_pre"] = full_data[
    spezielle_strassenverkehr_performance_anpassung_scales_pre
].mean(axis=1)
full_data["Spezielle_Strassenverkehr_Performance_Anpassung_Scales_post"] = full_data[
    spezielle_strassenverkehr_performance_anpassung_scales_post
].mean(axis=1)
full_data["Spezielle_Strassenverkehr_Performance_Anpassung_Scales_Diff"] = (
    full_data["Spezielle_Strassenverkehr_Performance_Anpassung_Scales_post"]
    - full_data["Spezielle_Strassenverkehr_Performance_Anpassung_Scales_pre"]
)

ff2_PrePost = {
    "Composite_Variables": [
        (
            "Spezielle_Strassenverkehr_Performance_Reaktion_Quoten_pre",
            "Spezielle_Strassenverkehr_Performance_Reaktion_Quoten_post",
            "Spezielle Straßenverkehr Performance (Reaktion Quoten)",
        ),
        (
            "Spezielle_Strassenverkehr_Performance_Reaktion_Scales_pre",
            "Spezielle_Strassenverkehr_Performance_Reaktion_Scales_post",
            "Spezielle Straßenverkehr Performance (Reaktion Skalen)",
        ),
        (
            "Spezielle_Strassenverkehr_Performance_Stresslevel_Scales_pre",
            "Spezielle_Strassenverkehr_Performance_Stresslevel_Scales_post",
            "Spezielle Straßenverkehr Performance (Stresslevel)",
        ),
        (
            "Spezielle_Strassenverkehr_Performance_Anpassung_Scales_pre",
            "Spezielle_Strassenverkehr_Performance_Anpassung_Scales_post",
            "Spezielle Straßenverkehr Performance (Anpassung)",
        ),
    ],
}

# Prepare the composite variables list
composite_vars = ff2_PrePost["Composite_Variables"]

# Perform the analysis
results_composite = perform_composite_analysis(full_data, composite_vars)

# Save the results to a CSV file
results_composite.to_csv("results_ff3.csv", index=False)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------- Forschungsfrage 4 -----------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Replace 'Ja' (1) with 1 and 'Nein' (2) with 0
full_data["ST_Analyse_post"] = full_data["ST_Analyse_post"].replace({1: 1, 2: 0})

# Composite variables for the main categories
composite_vars = [
    ("Fahrfaehigkeiten_Scales_Diff", "Fahrfähigkeiten (Skalen)"),
    ("Fahrfaehigkeiten_Quoten_Diff", "Fahrfähigkeiten (Quoten)"),
    ("Selbstvertrauen_Scales_Diff", "Selbstvertrauen (Skalen)"),
    ("Strassenverkehr_Performance_Scales_Diff", "Straßenverkehr Performance (Skalen)"),
    ("Strassenverkehr_Performance_Quoten_Diff", "Straßenverkehr Performance (Quoten)"),
]

import scipy.stats as stats


def perform_analysis_between_groups(data, composite_vars, group_var):
    results = []
    for diff_var, name in composite_vars:
        # Remove missing data
        data_clean = data[["Personal_Code", diff_var, group_var]].dropna()
        if data_clean.empty:
            print(f"No data for {name}. Skipping.")
            continue

        # Split data into two groups
        group1_data = data_clean[data_clean[group_var] == 1][diff_var]
        group2_data = data_clean[data_clean[group_var] == 0][diff_var]
        n1 = len(group1_data)
        n2 = len(group2_data)

        if n1 == 0 or n2 == 0:
            print(f"No data for one of the groups in {name}. Skipping.")
            continue

        # Calculate means and standard deviations
        mean1 = group1_data.mean()
        mean2 = group2_data.mean()
        std1 = group1_data.std(ddof=1)
        std2 = group2_data.std(ddof=1)

        # Perform independent samples t-test (Welch's t-test)
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((n1 - 1) * (std1**2) + (n2 - 1) * (std2**2)) / (n1 + n2 - 2)
        )
        cohen_d = (mean1 - mean2) / pooled_std if pooled_std != 0 else np.nan

        # Direction of difference
        if mean1 > mean2:
            direction = "Analysis group improved more"
        elif mean1 < mean2:
            direction = "No-analysis group improved more"
        else:
            direction = "No difference"

        # ------------------- Visualization -------------------
        plt.figure(figsize=(6, 4))
        sns.boxplot(
            x=group_var,
            y=diff_var,
            data=data_clean,
            palette=["lightgreen", "forestgreen"],
        )
        plt.title(f"{name}")
        plt.xticks(ticks=[0, 1], labels=["No Analysis", "Analysis"])
        plt.xlabel("Group")
        plt.ylabel("Difference Score")
        plt.tight_layout()

        # ------------------- Print Results -------------------
        print(f"\nResults for {name}:")
        print(f"Mean Difference (Analysis Group): {mean1:.3f}")
        print(f"Mean Difference (No Analysis Group): {mean2:.3f}")
        print(f"Direction: {direction}")
        print(f"Cohen's d: {cohen_d:.3f}")
        print(f"t({n1 + n2 - 2}) = {t_stat:.4f}, p = {p_value:.4f}")

        # Append results
        results.append(
            {
                "Variable": name,
                "Mean_Analysis_Group": mean1,
                "Mean_No_Analysis_Group": mean2,
                "Direction": direction,
                "Cohen_d": cohen_d,
                "t_stat": t_stat,
                "p_value": p_value,
                "n_Analysis_Group": n1,
                "n_No_Analysis_Group": n2,
            }
        )

        # plt.show()

    return pd.DataFrame(results)


# Prepare the data
group_var = "ST_Analyse_post"

# Ensure that 'ST_Analyse_post' is in the correct format
full_data[group_var] = full_data[group_var].replace({1: 1, 2: 0})

# Perform the analysis
results_ff4 = perform_analysis_between_groups(full_data, composite_vars, group_var)

# Display the results
print("\nSummary of Forschungsfrage 4 Results:")
print(results_ff4)

# Save the results to a CSV file
results_ff4.to_csv("results_ff4.csv", index=False)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------- Forschungsfrage 5 -----------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Replace 'Ja' (1) with 1 and 'Nein' (2) with 2 in ST_Eyetracking_postQ
# Ensure that missing values are handled appropriately
full_data["ST_Eyetracking_postQ"] = full_data["ST_Eyetracking_postQ"].replace(
    {1: 1, 2: 2, np.nan: np.nan}
)


# Function to determine the Eye-Tracking group
def determine_group_et(row):
    if row["ST_Praesentation_post"] == 2 and row["ST_Eyetracking_postQ"] == 1:
        return 1  # Eye-Tracking Group
    elif (row["ST_Praesentation_post"] == 2 and row["ST_Eyetracking_postQ"] == 2) or (
        row["ST_Praesentation_post"] == 1
    ):
        return 0  # No Eye-Tracking Group
    else:
        return np.nan  # Missing data


# Apply the function to create the Group_ET variable
full_data["Group_ET"] = full_data.apply(determine_group_et, axis=1)

# Check the counts in each group
print("Group counts:")
print(full_data["Group_ET"].value_counts(dropna=False))

# Composite variables for the main categories
composite_vars = [
    ("Fahrfaehigkeiten_Scales_Diff", "Fahrfähigkeiten (Skalen)"),
    ("Fahrfaehigkeiten_Quoten_Diff", "Fahrfähigkeiten (Quoten)"),
    ("Selbstvertrauen_Scales_Diff", "Selbstvertrauen (Skalen)"),
    ("Strassenverkehr_Performance_Scales_Diff", "Straßenverkehr Performance (Skalen)"),
    ("Strassenverkehr_Performance_Quoten_Diff", "Straßenverkehr Performance (Quoten)"),
]

import scipy.stats as stats


def perform_analysis_between_groups(data, composite_vars, group_var):
    results = []
    for diff_var, name in composite_vars:
        # Remove missing data
        data_clean = data[["Personal_Code", diff_var, group_var]].dropna()
        if data_clean.empty:
            print(f"No data for {name}. Skipping.")
            continue

        # Split data into two groups
        group1_data = data_clean[data_clean[group_var] == 1][diff_var]
        group2_data = data_clean[data_clean[group_var] == 0][diff_var]
        n1 = len(group1_data)
        n2 = len(group2_data)

        if n1 == 0 or n2 == 0:
            print(f"No data for one of the groups in {name}. Skipping.")
            continue

        # Calculate means and standard deviations
        mean1 = group1_data.mean()
        mean2 = group2_data.mean()
        std1 = group1_data.std(ddof=1)
        std2 = group2_data.std(ddof=1)

        # Perform independent samples t-test (Welch's t-test)
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((n1 - 1) * (std1**2) + (n2 - 1) * (std2**2)) / (n1 + n2 - 2)
        )
        cohen_d = (mean1 - mean2) / pooled_std if pooled_std != 0 else np.nan

        # Direction of difference
        if mean1 > mean2:
            direction = "Eye-Tracking group improved more"
        elif mean1 < mean2:
            direction = "No Eye-Tracking group improved more"
        else:
            direction = "No difference"

        # ------------------- Visualization -------------------
        plt.figure(figsize=(6, 4))
        sns.boxplot(
            x=group_var,
            y=diff_var,
            data=data_clean,
            palette=["lightgreen", "forestgreen"],
        )
        plt.title(f"{name}")
        plt.xticks(ticks=[0, 1], labels=["No Eye-Tracking", "Eye-Tracking"])
        plt.xlabel("Group")
        plt.ylabel("Difference Score")
        plt.tight_layout()

        # ------------------- Print Results -------------------
        print(f"\nResults for {name}:")
        print(f"Mean Difference (Eye-Tracking Group): {mean1:.3f}")
        print(f"Mean Difference (No Eye-Tracking Group): {mean2:.3f}")
        print(f"Direction: {direction}")
        print(f"Cohen's d: {cohen_d:.3f}")
        print(f"t({n1 + n2 - 2}) = {t_stat:.4f}, p = {p_value:.4f}")

        # Append results
        results.append(
            {
                "Variable": name,
                "Mean_EyeTracking_Group": mean1,
                "Mean_No_EyeTracking_Group": mean2,
                "Direction": direction,
                "Cohen_d": cohen_d,
                "t_stat": t_stat,
                "p_value": p_value,
                "n_EyeTracking_Group": n1,
                "n_No_EyeTracking_Group": n2,
            }
        )

        # plt.show()

    return pd.DataFrame(results)


# Prepare the data
group_var = "Group_ET"

# Perform the analysis
results_ff5 = perform_analysis_between_groups(full_data, composite_vars, group_var)

# Save the results to a CSV file
results_ff5.to_csv("results_ff5.csv", index=False)

# Display the results
print("\nSummary of Forschungsfrage 6 Results:")
print(results_ff5)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------- Forschungsfrage 6 -----------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Ensure that the presentation variable is correctly coded
# ST_Praesentation_post: 1 = VR, 2 = Monitor

# Create the Group_Presentation variable
full_data["Group_Presentation"] = full_data["ST_Praesentation_post"].replace(
    {1: "VR", 2: "Monitor"}
)

# Check group counts
print("Group counts:")
print(full_data["Group_Presentation"].value_counts(dropna=False))

# Composite variables for the main categories
composite_vars = [
    ("Fahrfaehigkeiten_Scales_Diff", "Fahrfähigkeiten (Skalen)"),
    ("Fahrfaehigkeiten_Quoten_Diff", "Fahrfähigkeiten (Quoten)"),
    ("Selbstvertrauen_Scales_Diff", "Selbstvertrauen (Skalen)"),
    ("Strassenverkehr_Performance_Scales_Diff", "Straßenverkehr Performance (Skalen)"),
    ("Strassenverkehr_Performance_Quoten_Diff", "Straßenverkehr Performance (Quoten)"),
]

import scipy.stats as stats


def perform_analysis_between_groups(data, composite_vars, group_var):
    results = []
    for diff_var, name in composite_vars:
        # Remove missing data
        data_clean = data[["Personal_Code", diff_var, group_var]].dropna()
        if data_clean.empty:
            print(f"No data for {name}. Skipping.")
            continue

        # Split data into two groups
        group1_data = data_clean[data_clean[group_var] == "VR"][diff_var]
        group2_data = data_clean[data_clean[group_var] == "Monitor"][diff_var]
        n1 = len(group1_data)
        n2 = len(group2_data)

        if n1 == 0 or n2 == 0:
            print(f"No data for one of the groups in {name}. Skipping.")
            continue

        # Calculate means and standard deviations
        mean1 = group1_data.mean()
        mean2 = group2_data.mean()
        std1 = group1_data.std(ddof=1)
        std2 = group2_data.std(ddof=1)

        # Perform independent samples t-test (Welch's t-test)
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((n1 - 1) * (std1**2) + (n2 - 1) * (std2**2)) / (n1 + n2 - 2)
        )
        cohen_d = (mean1 - mean2) / pooled_std if pooled_std != 0 else np.nan

        # Direction of difference
        if mean1 > mean2:
            direction = "VR group improved more"
        elif mean1 < mean2:
            direction = "Monitor group improved more"
        else:
            direction = "No difference"

        # ------------------- Visualization -------------------
        plt.figure(figsize=(6, 4))
        sns.boxplot(
            x=group_var,
            y=diff_var,
            data=data_clean,
            palette=["lightgreen", "forestgreen"],
        )
        plt.title(f"{name}")
        plt.xlabel("Presentation Medium")
        plt.ylabel("Difference Score")
        plt.tight_layout()

        # ------------------- Print Results -------------------
        print(f"\nResults for {name}:")
        print(f"Mean Difference (VR Group): {mean1:.3f}")
        print(f"Mean Difference (Monitor Group): {mean2:.3f}")
        print(f"Direction: {direction}")
        print(f"Cohen's d: {cohen_d:.3f}")
        print(f"t({n1 + n2 - 2}) = {t_stat:.4f}, p = {p_value:.4f}")

        # Append results
        results.append(
            {
                "Variable": name,
                "Mean_VR_Group": mean1,
                "Mean_Monitor_Group": mean2,
                "Direction": direction,
                "Cohen_d": cohen_d,
                "t_stat": t_stat,
                "p_value": p_value,
                "n_VR_Group": n1,
                "n_Monitor_Group": n2,
            }
        )

        # plt.show()

    return pd.DataFrame(results)


# Prepare the data
group_var = "Group_Presentation"

# Perform the analysis
results_ff6_presentation = perform_analysis_between_groups(
    full_data, composite_vars, group_var
)

# Save the results to a CSV file
results_ff6_presentation.to_csv("results_ff6.csv", index=False)

# Display the results
print("\nSummary of Forschungsfrage 6 Results (Presentation Medium):")
print(results_ff6_presentation)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------- Forschungsfrage 7 -----------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Assuming 'full_data' is your DataFrame containing all the data
# and 'data_forschungsfrage_7' is a dictionary containing the relevant columns

# Define the relevant columns for Forschungsfrage 7
data_forschungsfrage_7 = {
    "Bewertung": [
        "ET_SimRacing_Training_geholfen_Fahrfaehigkeiten_zu_verbessern_postQ",
        "ET_SimRacing_Training_anderen_Personen_als_empfehlen_postQ",
        "ET_SimRacing_Training_zufrieden_postQ",
        "ET_SimRacing_Training_steigert_Motivation_zu_Ueben_postQ",
        "ET_SimRacing_Training_als_nuetzlivh_fuer_reale_Fahrsituationen_postQ",
        "ET_SimRacing_Training_hat_Faehigkeiten_zur_Konzentration_verbessert_postQ",
    ],
    "Wahrnehmung_Training": [
        "SS_SimRacing_kann_helfen_Fahrfaehigkeiten_zu_verbessern_preQ",
        "SS_SimRacing_kann_helfen_Fahrfaehigkeiten_zu_verbessern_postQ",
        "SS_SimRacing_bietet_realistische_Fahrbedingungen_preQ",
        "SS_SimRacing_bietet_realistische_Fahrbedingungen_postQ",
        "SS_SimRacing_ist_ein_gut_um_Reaktionszeiten_zu_verbessern_preQ",
        "SS_SimRacing_ist_ein_gut_um_Reaktionszeiten_zu_verbessern_postQ",
        "SS_SimRacing_ist_ein_gut_um_Lenktechnik_zu_verbessern_preQ",
        "SS_SimRacing_ist_ein_gut_um_Lenktechnik_zu_verbessern_postQ",
        "SS_SimRacing_ist_ein_gut_um_bei_hohen_Geschwindigkeiten_zurechtzukommen_preQ",
        "SS_SimRacing_ist_ein_gut_um_bei_hohen_Geschwindigkeiten_zurechtzukommen_postQ",
        "SS_SimRacing_ist_ein_gut_um_bei_extremen_Wetterbedingungen_besser_umzugehen_preQ",
        "SS_SimRacing_ist_ein_gut_um_bei_extremen_Wetterbedingungen_besser_umzugehen_postQ",
    ],
    "Relevanz": [
        "SS_Reaktionszeiten_postQ",
        "SS_Kurvenfahren_postQ",
        "SS_Bremsverhalten_postQ",
        "SS_Verhalten_bei_schwierigen_Wetterbedingungen_postQ",
    ],
}

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# A. Compute Participant-Level Scores
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Compute participant-level mean scores for each category
full_data["Bewertung_Score"] = full_data[data_forschungsfrage_7["Bewertung"]].mean(
    axis=1
)
full_data["Wahrnehmung_Training_Score"] = full_data[
    data_forschungsfrage_7["Wahrnehmung_Training"]
].mean(axis=1)
full_data["Relevanz_Score"] = full_data[data_forschungsfrage_7["Relevanz"]].mean(axis=1)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# B. Perform Descriptive Statistics
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print("Descriptive Statistics for Bewertung_Score:")
print(full_data["Bewertung_Score"].describe())

print("\nDescriptive Statistics for Wahrnehmung_Training_Score:")
print(full_data["Wahrnehmung_Training_Score"].describe())

print("\nDescriptive Statistics for Relevanz_Score:")
print(full_data["Relevanz_Score"].describe())

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# C. Assess Reliability (Cronbach's Alpha)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# def cronbach_alpha(items):
#    items = items.dropna()
#    item_variances = items.var(axis=0, ddof=1)
#    total_score = items.sum(axis=1)
#    total_variance = total_score.var(ddof=1)
#    n_items = items.shape[1]
#    cronbach_alpha_value = (n_items / (n_items - 1)) * (1 - (item_variances.sum() / total_variance))
#    return cronbach_alpha_value

# Bewertung
# items_bewertung = full_data[data_forschungsfrage_7['Bewertung']]
# alpha_bewertung = cronbach_alpha(items_bewertung)
# print(f"\nCronbach's alpha for Bewertung: {alpha_bewertung:.3f}")

# Wahrnehmung_Training
# items_wahrnehmung = full_data[data_forschungsfrage_7['Wahrnehmung_Training']]
# alpha_wahrnehmung = cronbach_alpha(items_wahrnehmung)
# print(f"Cronbach's alpha for Wahrnehmung_Training: {alpha_wahrnehmung:.3f}")

# Relevanz
# items_relevanz = full_data[data_forschungsfrage_7['Relevanz']]
# alpha_relevanz = cronbach_alpha(items_relevanz)
# print(f"Cronbach's alpha for Relevanz: {alpha_relevanz:.3f}")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# D. Conduct Exploratory Factor Analysis (EFA)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Combine all items into one DataFrame
all_items = full_data[
    data_forschungsfrage_7["Bewertung"]
    + data_forschungsfrage_7["Wahrnehmung_Training"]
    + data_forschungsfrage_7["Relevanz"]
].dropna()

# Check the Kaiser-Meyer-Olkin (KMO) Measure
kmo_all, kmo_model = calculate_kmo(all_items)
print(f"\nKMO Measure: {kmo_model:.3f}")

# Bartlett's Test of Sphericity
chi_square_value, p_value = calculate_bartlett_sphericity(all_items)
print(f"Bartlett's Test: chi2 = {chi_square_value:.3f}, p = {p_value:.3f}")

# Determine the number of factors using eigenvalues
fa = FactorAnalyzer(rotation=None)
fa.fit(all_items)
eigen_values, vectors = fa.get_eigenvalues()

# Plot eigenvalues (Scree Plot)
plt.figure(figsize=(8, 6))
plt.scatter(range(1, all_items.shape[1] + 1), eigen_values)
plt.plot(range(1, all_items.shape[1] + 1), eigen_values)
plt.title("Scree Plot")
plt.xlabel("Factors")
plt.ylabel("Eigenvalue")
plt.grid()
# plt.show()

# Decide on the number of factors (e.g., factors with eigenvalues > 1)
n_factors = sum(eigen_values > 1)
print(f"\nNumber of factors to extract: {n_factors}")

# Perform Factor Analysis with Varimax rotation
fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
fa.fit(all_items)

# Get factor loadings
loadings = pd.DataFrame(fa.loadings_, index=all_items.columns)
print("\nFactor Loadings:")
print(loadings)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# E. Analyze Relationships
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Improvement measures (ensure these variables are in your DataFrame)
improvement_measures = [
    "Fahrfaehigkeiten_Scales_Diff",
    "Selbstvertrauen_Scales_Diff",
    "Strassenverkehr_Performance_Scales_Diff",
]

# Create a DataFrame with acceptance scores and improvement measures
analysis_data = full_data[
    ["Bewertung_Score", "Wahrnehmung_Training_Score", "Relevanz_Score"]
    + improvement_measures
].dropna()

# Compute correlation matrix
correlation_matrix = analysis_data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix between Acceptance Scores and Improvement Measures")
# plt.show()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# F. Visualizations
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Histograms of acceptance scores
plt.figure(figsize=(8, 4))
plt.hist(
    full_data["Bewertung_Score"].dropna(),
    bins=10,
    color="mediumseagreen",
    edgecolor="black",
)
plt.title("Verteilung der Bewertung Scores")
plt.xlabel("Bewertung Score")
plt.ylabel("Frequency")
# plt.show()

plt.figure(figsize=(8, 4))
plt.hist(
    full_data["Wahrnehmung_Training_Score"].dropna(),
    bins=10,
    color="forestgreen",
    edgecolor="black",
)
plt.title("Verteilung der Wahrnehmung Training Scores")
plt.xlabel("Wahrnehmung Training Score")
plt.ylabel("Frequency")
# plt.show()

plt.figure(figsize=(8, 4))
plt.hist(
    full_data["Relevanz_Score"].dropna(), bins=10, color="lightgreen", edgecolor="black"
)
plt.title("Verteilung der Relevanz Scores")
plt.xlabel("Relevanz Score")
plt.ylabel("Frequency")
# plt.show()

# Boxplots of acceptance scores
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=full_data[["Bewertung_Score", "Wahrnehmung_Training_Score", "Relevanz_Score"]],
    palette=["lightgreen", "forestgreen", "mediumseagreen"],
)
plt.title("Boxplot der Skalen")
plt.ylabel("Score")
plt.xticks(ticks=[0, 1, 2], labels=["Bewertung", "Wahrnehmung", "Relevanz"])
# plt.show()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# G. Additional Analyses
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Example: Group Comparisons by Gender (assuming 'Gender' column exists)
# Replace 'Gender' with the appropriate column name in your DataFrame

# Ensure 'Gender' column is correctly coded (e.g., 'Male', 'Female')
# If it's numerical, map it accordingly
# full_data['Gender'] = full_data['Gender'].replace({1: 'Male', 2: 'Female'})

# Boxplot of Bewertung Score by Gender
if "Gender" in full_data.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Gender", y="Bewertung_Score", data=full_data)
    plt.title("Bewertung Score by Gender")
    # plt.show()

    # Perform t-test between groups
    male_scores = full_data[full_data["Gender"] == "Male"]["Bewertung_Score"].dropna()
    female_scores = full_data[full_data["Gender"] == "Female"][
        "Bewertung_Score"
    ].dropna()

    t_stat, p_value = stats.ttest_ind(male_scores, female_scores, equal_var=False)
    print(f"\nt-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")

# Regression Analysis: Does Bewertung_Score predict Fahrfaehigkeiten_Scales_Diff?
X = full_data["Bewertung_Score"]
y = full_data["Fahrfaehigkeiten_Scales_Diff"]
regression_data = pd.concat([X, y], axis=1).dropna()

X = regression_data["Bewertung_Score"]
y = regression_data["Fahrfaehigkeiten_Scales_Diff"]
X = sm.add_constant(X)  # Add intercept

model = sm.OLS(y, X).fit()
print("\nRegression Analysis Results:")
print(model.summary())

# Scatter plot with regression line
plt.figure(figsize=(8, 6))
sns.regplot(x="Bewertung_Score", y="Fahrfaehigkeiten_Scales_Diff", data=regression_data)
plt.title("Regression of Fahrfähigkeiten Improvement on Bewertung Score")
plt.xlabel("Bewertung Score")
plt.ylabel("Fahrfähigkeiten Scales Difference")
# plt.show()

# Save the acceptance scores and improvement measures to a CSV file
analysis_data.to_csv("results_ff7.csv", index=False)
print("\nAnalysis data saved to 'forschungsfrage_7_analysis_data.csv'.")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------- Forschungsfrage 8 -----------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Inwiefern können die Ergebnisse aus der Simulation auf die reale Welt übertragen werden?

# Items related to the assessment of training with regard to reality
ss03_items = [
    "SS_SimRacing_kann_helfen_Fahrfaehigkeiten_zu_verbessern_postQ",  # SS03_01
    "SS_SimRacing_bietet_realistische_Fahrbedingungen_postQ",  # SS03_02
    "SS_SimRacing_ist_ein_gut_um_Reaktionszeiten_zu_verbessern_postQ",  # SS03_03
    "SS_SimRacing_ist_ein_gut_um_Lenktechnik_zu_verbessern_postQ",  # SS03_04
    "SS_SimRacing_ist_ein_gut_um_bei_hohen_Geschwindigkeiten_zurechtzukommen_postQ",  # SS03_05
    "SS_SimRacing_ist_ein_gut_um_bei_extremen_Wetterbedingungen_besser_umzugehen_postQ",  # SS03_06
]

# Items explicitly related to the relevance for real driving
ss04_items = [
    "SS_Reaktionszeiten_postQ",  # SS04_01
    "SS_Kurvenfahren_postQ",  # SS04_02
    "SS_Bremsverhalten_postQ",  # SS04_03
    "SS_Verhalten_bei_schwierigen_Wetterbedingungen_postQ",  # SS04_04
]

# Item about improvement of driving skills through training
et01_item = (
    "ET_SimRacing_Training_geholfen_Fahrfaehigkeiten_zu_verbessern_postQ"  # ET01_01
)

# Ensure that the items are in the dataset
all_items = ss03_items + ss04_items + [et01_item]
missing_items = [item for item in all_items if item not in full_data.columns]
if missing_items:
    print(f"Warning: The following items are missing in the dataset: {missing_items}")

# Compute composite score for SS03 items (Perception of training's realism)
full_data["SS03_Composite"] = full_data[ss03_items].mean(axis=1)

# Compute composite score for SS04 items (Relevance for real driving)
full_data["SS04_Composite"] = full_data[ss04_items].mean(axis=1)

# Difference scores already computed in previous analyses
diff_variables = {
    "Fahrfaehigkeiten_Scales_Diff": "Fahrfähigkeiten (Skalen)",
    "Fahrfaehigkeiten_Quoten_Diff": "Fahrfähigkeiten (Quoten)",
    "Selbstvertrauen_Scales_Diff": "Selbstvertrauen (Skalen)",
    "Strassenverkehr_Performance_Scales_Diff": "Straßenverkehr Performance (Skalen)",
    "Strassenverkehr_Performance_Quoten_Diff": "Straßenverkehr Performance (Quoten)",
}

# Combine the data into a single DataFrame
analysis_data = full_data[
    list(diff_variables.keys()) + ["SS03_Composite", "SS04_Composite", et01_item]
]

# Compute correlation matrix
correlation_matrix = analysis_data.corr(method="pearson")

# Extract relevant correlations
relevant_correlations = correlation_matrix.loc[
    ["SS03_Composite", "SS04_Composite", et01_item], list(diff_variables.keys())
]
print("Correlation between reality perception items and difference scores:")
print(relevant_correlations)

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(relevant_correlations, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between Reality Perception and Improvement Measures")
# plt.show()

# save the correlation matrix to a CSV file
relevant_correlations.to_csv("results_ff8_correlation_matrix.csv")

from scipy.stats import pearsonr


# Function to compute correlations and p-values
def compute_correlations(data, items, diff_vars):
    results = []
    for item in items:
        for diff_var in diff_vars:
            # Drop missing values
            data_clean = data[[item, diff_var]].dropna()
            if data_clean.empty:
                continue
            x = data_clean[item]
            y = data_clean[diff_var]
            r_value, p_value = pearsonr(x, y)
            results.append(
                {
                    "Item": item,
                    "Difference_Var": diff_var,
                    "Correlation": r_value,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(results)


# Items to analyze
items_to_analyze = ["SS03_Composite", "SS04_Composite", et01_item]
diff_vars = list(diff_variables.keys())

# Compute correlations
correlation_results = compute_correlations(analysis_data, items_to_analyze, diff_vars)

print("\nCorrelation Results:")
print(correlation_results)

# Save the correlation results to a CSV file
correlation_results.to_csv("results_ff8_correlations.csv", index=False)

# Descriptive statistics for SS04 items
ss04_descriptive = full_data[ss04_items].describe()
print("\nDescriptive Statistics for SS04 Items:")
print(ss04_descriptive)

# Transpose and save to CSV
ss04_descriptive_transposed = ss04_descriptive.transpose()
ss04_descriptive_transposed.to_csv("results_ff8_SS04_Descriptive_Statistics.csv")

# Visualize the distribution of SS04 items
plt.figure(figsize=(10, 6))
sns.boxplot(data=full_data[ss04_items], palette="Set3")
plt.title("Distribution of SS04 Items (Relevance for Real Driving)")
plt.ylabel("Rating")
plt.xticks(rotation=90)
# plt.show()

# Scatter plots between difference scores and composite scores
for item in items_to_analyze:
    for diff_var in diff_vars:
        plt.figure(figsize=(6, 4))
        sns.regplot(
            x=analysis_data[item],
            y=analysis_data[diff_var],
            scatter_kws={"alpha": 0.5},
            line_kws={"color": "red"},
        )
        plt.title(f"{diff_variables[diff_var]} vs. {item}")
        plt.xlabel(item)
        plt.ylabel(diff_variables[diff_var])
        plt.tight_layout()
        # plt.show()

# Create a DataFrame with composite scores and difference scores
export_data_ff8 = analysis_data.copy()

# Save to CSV
export_data_ff8.to_csv("results_ff8_Analysis_Data.csv", index=False)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------- Forschungsfrage 9 -----------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# List of relevant items with full variable names
# ET02: Influence of Presentation Medium
et02_items = {
    "ET_Praesentationsmedium_erleichter_konzentration_auf_Training_postQ": "ET02_01",
    "ET_Praesentationsmedium_realitaetsnah_postQ": "ET02_02",
    "ET_Praesentationsmedium_Trainingsinhalte_verstaendlich_postQ": "ET02_03",
    "ET_Praesentationsmedium_bevorzugen_für_Training_postQ": "ET02_04",
}

# ET03: Immersion and Realism
et03_items = {
    "ET_Immersion_vollstaendig_eingetaucht_postQ": "ET03_01",
    "ET_Immersion_fahrpyhsik_realistisch_postQ": "ET03_02",
    "ET_Immersion_training_gefaesselt_postQ": "ET03_03",
    "ET_Immersion_realistisch_simulation_als_real_postQ": "ET03_04",
    "ET_Immersion_training_vermittelt_realistische_fahrersituationen_postQ": "ET03_05",
    "ET_Immersion_interaktivitaet_und_steuerung_verstaerkt_realismus_postQ": "ET03_06",
}

# ET04: Feedback and Learning Progress
et04_items = {
    "ET_Feedback_hilfreich_um_Leistungen_zu_verbessern_postQ": "ET04_01",
    "ET_Feedback_klar_und_verstaendlich_postQ": "ET04_02",
    "ET_Feedback_Lernfortschritt_gut_nachvollziehbar_postQ": "ET04_03",
    "ET_Feedback_angeregt_Fahrverhalten_zu_ueberdenken_postQ": "ET04_04",
    "ET_Feedback_geholfen_an_Schwierigkeitsgrad_anzupassen_postQ": "ET04_05",
    "ET_Feedback_Fahrfaehigkeiten_durch_Training_kontinuierlich_verbessert_postQ": "ET04_06",
}

# ET05: Didactics and Structure
et05_items = {
    "ET_Didaktik_lerninhalte_verstaendlich_postQ": "ET05_01",
    "ET_Didaktik_training_gut_strukturiert_postQ": "ET05_02",
    "ET_Didaktik_training_auf_persoenliche_Niveau_angepasst_postQ": "ET05_03",
    "ET_Didaktik_training_in_kleinen_Schritten_postQ": "ET05_04",
    "ET_Didaktik_gestaltung_hat_motivation_gesteigert_postQ": "ET05_05",
    "ET_Didaktik_balance_zwischen_theorie_praxis_postQ": "ET05_06",
    "ET_Didaktik_inhalte_verstaendlich_und_nachvollziehbar_postQ": "ET05_07",
    "ET_Didaktik_struktur_flexibel_um_individuelle_beduerfnisse_postQ": "ET05_08",
}

# ST Items: Presentation Medium
st_items = {
    "ST_Praesentation_post": "ST_Praesentation",  # 1 = VR, 2 = Monitor
}

# Extract ET03 items into a DataFrame
et03_df = full_data[list(et03_items.keys())]

# Compute descriptive statistics
et03_descriptive = et03_df.describe()
print("Descriptive Statistics for Immersion and Realism (ET03):")
print(et03_descriptive)

# Compute a composite score for immersion and realism
full_data["Immersion_Realism_Score"] = et03_df.mean(axis=1)

# Ensure that the presentation medium variable is correctly coded
# ST_Praesentation_post: 1 = VR, 2 = Monitor
full_data["Presentation_Medium"] = full_data["ST_Praesentation_post"].replace(
    {1: "VR", 2: "Monitor"}
)

# Compare Immersion_Realism_Score between VR and Monitor groups
import scipy.stats as stats

# Remove missing data
immersion_data = full_data[["Immersion_Realism_Score", "Presentation_Medium"]].dropna()

# Split data into groups
vr_group = immersion_data[immersion_data["Presentation_Medium"] == "VR"][
    "Immersion_Realism_Score"
]
monitor_group = immersion_data[immersion_data["Presentation_Medium"] == "Monitor"][
    "Immersion_Realism_Score"
]

# Perform independent samples t-test (Welch's t-test)
t_stat, p_value = stats.ttest_ind(vr_group, monitor_group, equal_var=False)

print(f"Immersion and Realism Score Comparison:")
print(f"Mean VR Group: {vr_group.mean():.2f}")
print(f"Mean Monitor Group: {monitor_group.mean():.2f}")
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.boxplot(
    x="Presentation_Medium",
    y="Immersion_Realism_Score",
    data=immersion_data,
    palette="Set3",
)
plt.title("Immersion and Realism by Presentation Medium")
plt.xlabel("Presentation Medium")
plt.ylabel("Immersion and Realism Score")
plt.tight_layout()
# plt.show()

# Extract ET04 items into a DataFrame
et04_df = full_data[list(et04_items.keys())]

# Compute descriptive statistics
et04_descriptive = et04_df.describe()
print("Descriptive Statistics for Feedback and Learning Progress (ET04):")
print(et04_descriptive)

# Compute a composite score for feedback evaluation
full_data["Feedback_Score"] = et04_df.mean(axis=1)

# Assuming difference scores have been calculated previously
diff_variables = {
    "Fahrfaehigkeiten_Scales_Diff": "Driving Abilities (Scales)",
    "Selbstvertrauen_Scales_Diff": "Self-Confidence (Scales)",
    "Strassenverkehr_Performance_Scales_Diff": "Traffic Performance (Scales)",
}

# Analyze correlations between Feedback_Score and difference scores
from scipy.stats import pearsonr

feedback_impact_results = []
for diff_var, diff_name in diff_variables.items():
    # Remove missing data
    data_clean = full_data[["Feedback_Score", diff_var]].dropna()
    if data_clean.empty:
        continue
    correlation, p_value = pearsonr(data_clean["Feedback_Score"], data_clean[diff_var])
    feedback_impact_results.append(
        {
            "Difference_Variable": diff_name,
            "Correlation": correlation,
            "p_value": p_value,
        }
    )

# Convert to DataFrame
feedback_impact_df = pd.DataFrame(feedback_impact_results)
print("Correlation between Feedback Score and Improvement Scores:")
print(feedback_impact_df)

# Descriptive statistics for ET_Feedback_Lernfortschritt_gut_nachvollziehbar_postQ
learning_progress = full_data[
    "ET_Feedback_Lernfortschritt_gut_nachvollziehbar_postQ"
].dropna()
print(f"Mean Learning Progress Score: {learning_progress.mean():.2f}")
print(learning_progress.describe())

# Histogram
plt.figure(figsize=(6, 4))
sns.histplot(learning_progress, bins=5, kde=False)
plt.title("Learning Progress Evaluation")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.tight_layout()
# plt.show()

# Prepare data
learning_progress_data = full_data[
    ["ET_Feedback_Lernfortschritt_gut_nachvollziehbar_postQ", "Presentation_Medium"]
].dropna()

# Perform independent samples t-test
vr_lp = learning_progress_data[learning_progress_data["Presentation_Medium"] == "VR"][
    "ET_Feedback_Lernfortschritt_gut_nachvollziehbar_postQ"
]
monitor_lp = learning_progress_data[
    learning_progress_data["Presentation_Medium"] == "Monitor"
]["ET_Feedback_Lernfortschritt_gut_nachvollziehbar_postQ"]

t_stat_lp, p_value_lp = stats.ttest_ind(vr_lp, monitor_lp, equal_var=False)

print(f"Learning Progress Score Comparison by Presentation Medium:")
print(f"Mean VR Group: {vr_lp.mean():.2f}")
print(f"Mean Monitor Group: {monitor_lp.mean():.2f}")
print(f"t-statistic: {t_stat_lp:.4f}, p-value: {p_value_lp:.4f}")

# Visualization
plt.figure(figsize=(6, 4))
sns.boxplot(
    x="Presentation_Medium",
    y="ET_Feedback_Lernfortschritt_gut_nachvollziehbar_postQ",
    data=learning_progress_data,
    palette="Set2",
)
plt.title("Learning Progress by Presentation Medium")
plt.xlabel("Presentation Medium")
plt.ylabel("Learning Progress Score")
plt.tight_layout()
# plt.show()

# Extract ET05 items into a DataFrame
et05_df = full_data[list(et05_items.keys())]

# Compute descriptive statistics
et05_descriptive = et05_df.describe()
print("Descriptive Statistics for Didactics and Structure (ET05):")
print(et05_descriptive)

# Compute a composite score for didactics and structure
full_data["Didactics_Structure_Score"] = et05_df.mean(axis=1)

# Prepare data
didactics_data = full_data[
    ["Didactics_Structure_Score", "Presentation_Medium"]
].dropna()

# Perform independent samples t-test
vr_ds = didactics_data[didactics_data["Presentation_Medium"] == "VR"][
    "Didactics_Structure_Score"
]
monitor_ds = didactics_data[didactics_data["Presentation_Medium"] == "Monitor"][
    "Didactics_Structure_Score"
]

t_stat_ds, p_value_ds = stats.ttest_ind(vr_ds, monitor_ds, equal_var=False)

print(f"Didactics and Structure Score Comparison by Presentation Medium:")
print(f"Mean VR Group: {vr_ds.mean():.2f}")
print(f"Mean Monitor Group: {monitor_ds.mean():.2f}")
print(f"t-statistic: {t_stat_ds:.4f}, p-value: {p_value_ds:.4f}")

# Visualization
plt.figure(figsize=(6, 4))
sns.boxplot(
    x="Presentation_Medium",
    y="Didactics_Structure_Score",
    data=didactics_data,
    palette="Set1",
)
plt.title("Didactics and Structure by Presentation Medium")
plt.xlabel("Presentation Medium")
plt.ylabel("Didactics and Structure Score")
plt.tight_layout()
plt.show()

# Export descriptive statistics
et03_descriptive.to_csv("Immersion_Realism_Descriptive.csv")
et04_descriptive.to_csv("Feedback_LearningProgress_Descriptive.csv")
et05_descriptive.to_csv("Didactics_Structure_Descriptive.csv")

# Export correlation results
feedback_impact_df.to_csv("Feedback_Impact_Correlation.csv", index=False)

# Export composite scores
composite_scores = full_data[
    [
        "Personal_Code",
        "Immersion_Realism_Score",
        "Feedback_Score",
        "Didactics_Structure_Score",
    ]
]
composite_scores.to_csv("Composite_Scores.csv", index=False)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------- End of Analysis -------------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -----------------------------------------------------------------
# ------------------- Speicherung der Daten -----------------------
# -----------------------------------------------------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

logging.info("Speicherung der Daten in CSV-Dateien.")
# Alle relevanten Ergebnisse und Analysen als CSV ausgeben
full_data.to_csv("full_data_updated_with_new_column_names.csv", index=False)
export_data.to_csv("export_data_updated.csv", index=False)
logging.info("Daten wurden erfolgreich gespeichert.")
