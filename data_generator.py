import random
import uuid
from datetime import datetime, timedelta
from faker import Faker
import pandas as pd
import numpy as np
from pathlib import Path

fake = Faker('en_US')
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# ==========================================================
# CONFIGURATION
# ==========================================================
CONFIG = {
    "start_date": "2022-01-01",
    "end_date": "2025-12-31",
    "min_records": 500000,
    "max_records": 1000000,
    "patient_count": 100000,
    "provider_count": 500,
    "clinic_count": 75,
    "output_dir": str(Path(__file__).parent / "raw_data"),
    # ── Chunked raw-data output ──────────────────────────────────────────────
    # Each large table is written as multiple numbered chunk files so the
    # pipeline gets realistic multi-file ingestion (like real source systems).
    # Small reference tables (clinics, providers, payers) stay as single files.
    "chunk_size": 50000,          # rows per chunk file
    "csv_chunk_ratio": 0.6,       # 60 % of chunks written as CSV, 40 % as Parquet
    #   → PySpark reads the whole folder: spark.read.csv("raw_data/claims/")
}

# ==========================================================
# REFERENCE DATA  — realistic, comprehensive
# ==========================================================
SPECIALTIES = [
    ("Family Medicine",        "207Q00000X"),
    ("Internal Medicine",      "207R00000X"),
    ("Cardiology",             "207RC0000X"),
    ("Orthopedic Surgery",     "207X00000X"),
    ("Pediatrics",             "208000000X"),
    ("Neurology",              "2084N0400X"),
    ("Radiology",              "2085R0202X"),
    ("Dermatology",            "207N00000X"),
    ("Gastroenterology",       "207RG0100X"),
    ("Pulmonology",            "207RP1001X"),
    ("Nephrology",             "207RN0300X"),
    ("Endocrinology",          "207RE0101X"),
    ("Oncology",               "207RX0202X"),
    ("Psychiatry",             "2084P0800X"),
    ("Emergency Medicine",     "207P00000X"),
    ("Anesthesiology",         "207L00000X"),
    ("Obstetrics/Gynecology",  "207V00000X"),
    ("Urology",                "208800000X"),
    ("Ophthalmology",          "207W00000X"),
    ("ENT",                    "207Y00000X"),
]

CREDENTIALS = ["MD", "DO", "NP", "PA", "CRNA", "DDS"]

PAYERS = [
    ("Medicare",                "Government",   "00010", "MCARE"),
    ("Medicaid",                "Government",   "00020", "MCAID"),
    ("Aetna",                   "Commercial",   "60054", "AETNA"),
    ("Blue Cross Blue Shield",  "Commercial",   "00050", "BCBS"),
    ("United Healthcare",       "Commercial",   "87726", "UHC"),
    ("Cigna",                   "Commercial",   "62308", "CIGNA"),
    ("Humana",                  "Commercial",   "61101", "HUMANA"),
    ("Anthem",                  "Commercial",   "00090", "ANTHEM"),
    ("Molina Healthcare",       "Government",   "00100", "MOLINA"),
    ("Tricare",                 "Government",   "TRIC1", "TRIC"),
    ("Kaiser Permanente",       "Commercial",   "00120", "KAISER"),
    ("Oscar Health",            "Commercial",   "00130", "OSCAR"),
    ("WellCare",                "Government",   "00140", "WCARE"),
    ("Centene",                 "Government",   "00150", "CNTNE"),
    ("Self-Pay",                "Self-Pay",     "00000", "SELF"),
    ("Workers Comp",            "Workers Comp", "00160", "WCOMP"),
]

PLAN_TYPES = ["HMO", "PPO", "EPO", "POS", "HDHP", "Medicare Advantage", "Medicaid Managed Care"]

APPOINTMENT_TYPES = [
    "New Patient", "Follow Up", "Annual Physical", "Consultation",
    "Emergency", "Telehealth", "Procedure", "Lab Only", "Pre-Op", "Post-Op",
]

APPOINTMENT_STATUSES_WEIGHTS = (
    ["Completed", "Cancelled", "No Show", "Scheduled", "Rescheduled"],
    [65, 12, 8, 10, 5],
)

CLINIC_TYPES = ["Hospital", "Private Practice", "Urgent Care", "Specialty Clinic", "FQHC", "ASC"]
EMR_SYSTEMS  = ["Epic", "Cerner", "Meditech", "Athena", "eClinicalWorks", "NextGen"]
ACCREDITATIONS = ["Joint Commission", "AAAHC", "DNV", "None"]

# ICD-10: (code, description, chronic, hcc_code)
ICD10_CODES = [
    ("I10",    "Essential Hypertension",                          True,  "85"),
    ("E11.9",  "Type 2 Diabetes Mellitus without complications",  True,  "19"),
    ("E11.65", "Type 2 Diabetes with Hyperglycemia",             True,  "19"),
    ("E11.40", "Type 2 Diabetes with Diabetic Neuropathy",       True,  "18"),
    ("E11.319","Type 2 Diabetes with Diabetic Retinopathy",      True,  "18"),
    ("N18.3",  "Chronic Kidney Disease Stage 3",                 True,  "137"),
    ("N18.4",  "Chronic Kidney Disease Stage 4",                 True,  "136"),
    ("I50.9",  "Heart Failure Unspecified",                      True,  "85"),
    ("I50.32", "Chronic Diastolic Heart Failure",                True,  "85"),
    ("I25.10", "Atherosclerotic Heart Disease",                  True,  "86"),
    ("I48.91", "Unspecified Atrial Fibrillation",                True,  "96"),
    ("J44.1",  "COPD with Acute Exacerbation",                   True,  "111"),
    ("J44.0",  "COPD with Acute Lower Respiratory Infection",    True,  "111"),
    ("J45.41", "Moderate Persistent Asthma with Exacerbation",   True,  "110"),
    ("E78.5",  "Hyperlipidemia Unspecified",                     True,  None),
    ("E66.01", "Morbid Obesity due to Excess Calories",          True,  "22"),
    ("F32.1",  "Major Depressive Disorder Moderate",             True,  "59"),
    ("F41.1",  "Generalized Anxiety Disorder",                   True,  None),
    ("G47.33", "Obstructive Sleep Apnea",                        True,  None),
    ("E03.9",  "Hypothyroidism Unspecified",                     True,  None),
    ("K21.0",  "GERD with Esophagitis",                          True,  None),
    ("M17.11", "Primary Osteoarthritis Right Knee",              True,  None),
    ("M17.12", "Primary Osteoarthritis Left Knee",               True,  None),
    ("M54.5",  "Low Back Pain",                                  False, None),
    ("M54.2",  "Cervicalgia",                                    False, None),
    ("I63.9",  "Cerebral Infarction Unspecified",                False, "100"),
    ("A41.9",  "Sepsis Unspecified",                             False, "2"),
    ("J18.9",  "Pneumonia Unspecified",                          False, None),
    ("J06.9",  "Acute Upper Respiratory Infection",              False, None),
    ("N39.0",  "Urinary Tract Infection",                        False, None),
    ("S72.001A","Fracture Femoral Neck Right Initial",           False, None),
    ("C18.9",  "Malignant Neoplasm Colon Unspecified",           True,  "12"),
    ("C34.10", "Malignant Neoplasm Upper Lobe Lung",             True,  "9"),
    ("C50.911","Malignant Neoplasm Breast Female",               True,  "12"),
    ("U07.1",  "COVID-19",                                       False, None),
    ("Z00.00", "Encounter for General Adult Medical Exam",       False, None),
    ("Z12.11", "Encounter for Screening Colon Cancer",           False, None),
    ("Z23",    "Encounter for Immunization",                     False, None),
    ("Z51.11", "Encounter for Antineoplastic Chemotherapy",      False, None),
    ("R05.9",  "Cough Unspecified",                              False, None),
    ("R51.9",  "Headache Unspecified",                           False, None),
    ("R10.9",  "Unspecified Abdominal Pain",                     False, None),
    ("R55",    "Syncope and Collapse",                           False, None),
    ("R00.0",  "Tachycardia Unspecified",                        False, None),
]

# CPT: (code, description, charge, requires_auth)
CPT_CODES = [
    ("99202", "Office Visit New Patient Low",              110,  False),
    ("99203", "Office Visit New Patient Moderate",         165,  False),
    ("99204", "Office Visit New Patient Moderate-High",    230,  False),
    ("99205", "Office Visit New Patient High",             290,  False),
    ("99211", "Office Visit Established Minimal",           45,  False),
    ("99212", "Office Visit Established Low",               85,  False),
    ("99213", "Office Visit Established Moderate",         130,  False),
    ("99214", "Office Visit Established Moderate-High",    185,  False),
    ("99215", "Office Visit Established High",             250,  False),
    ("99221", "Initial Hospital Care Low",                 220,  False),
    ("99222", "Initial Hospital Care Moderate",            310,  False),
    ("99223", "Initial Hospital Care High",                430,  False),
    ("99231", "Subsequent Hospital Care Low",              110,  False),
    ("99232", "Subsequent Hospital Care Moderate",         165,  False),
    ("99233", "Subsequent Hospital Care High",             230,  False),
    ("99238", "Hospital Discharge Day 30 min",             145,  False),
    ("99283", "Emergency Dept Visit Moderate",             280,  False),
    ("99284", "Emergency Dept Visit Moderate-High",        380,  False),
    ("99285", "Emergency Dept Visit High",                 490,  False),
    ("99381", "Preventive Visit New Patient Infant",       175,  False),
    ("99395", "Preventive Visit Established 18-39",        195,  False),
    ("99396", "Preventive Visit Established 40-64",        215,  False),
    ("99397", "Preventive Visit Established 65+",          235,  False),
    ("93000", "Electrocardiogram ECG",                      95,  False),
    ("93306", "Echocardiography with Doppler",             850,  True),
    ("93454", "Coronary Angiography",                     3200,  True),
    ("33533", "CABG Arterial Single",                    18000,  True),
    ("71046", "Chest X-Ray 2 Views",                       110,  False),
    ("70553", "MRI Brain with Contrast",                  1800,  True),
    ("72148", "MRI Lumbar Spine without Contrast",        1600,  True),
    ("74177", "CT Abdomen Pelvis with Contrast",          1400,  True),
    ("80053", "Comprehensive Metabolic Panel",              65,  False),
    ("85025", "Complete Blood Count with Diff",             45,  False),
    ("83036", "Hemoglobin A1c",                             55,  False),
    ("82947", "Glucose Blood",                              35,  False),
    ("84443", "TSH",                                        75,  False),
    ("80061", "Lipid Panel",                                60,  False),
    ("36415", "Venipuncture",                               25,  False),
    ("90471", "Immunization Administration",                25,  False),
    ("90686", "Influenza Vaccine",                          45,  False),
    ("90460", "Immunization Admin Child",                   25,  False),
    ("20610", "Arthrocentesis Major Joint",                185,  False),
    ("27447", "Total Knee Arthroplasty",                 12000,  True),
    ("27130", "Total Hip Arthroplasty",                  13500,  True),
    ("43239", "EGD with Biopsy",                          1200,  True),
    ("45378", "Colonoscopy Diagnostic",                   1500,  True),
    ("45380", "Colonoscopy with Biopsy",                  1800,  True),
    ("19307", "Mastectomy Modified Radical",              8500,  True),
    ("96413", "Chemotherapy IV Infusion 1 hr",             650,  True),
    ("96415", "Chemotherapy IV Infusion each add hr",      200,  True),
    ("99213", "Office Visit Established Moderate",         130,  False),
]
# deduplicate CPT list
_seen_cpt = set()
_unique_cpt = []
for c in CPT_CODES:
    if c[0] not in _seen_cpt:
        _seen_cpt.add(c[0])
        _unique_cpt.append(c)
CPT_CODES = _unique_cpt

CPT_DICT = {c[0]: c for c in CPT_CODES}

MODIFIERS = ["25", "59", "GT", "95", "TC", "26", "50", "LT", "RT", None]

DENIAL_CODES = [
    ("CO-4",  "Inconsistent Modifier",                    "Administrative"),
    ("CO-11", "Diagnosis Inconsistent with Procedure",    "Clinical"),
    ("CO-16", "Claim Lacks Information",                  "Administrative"),
    ("CO-22", "Coordination of Benefits",                 "COB"),
    ("CO-45", "Charge Exceeds Fee Schedule",              "Contractual"),
    ("CO-97", "Service Included in Another Service",      "Administrative"),
    ("PR-1",  "Deductible Amount",                        "Patient Responsibility"),
    ("PR-2",  "Coinsurance Amount",                       "Patient Responsibility"),
    ("PR-3",  "Copay Amount",                             "Patient Responsibility"),
    ("OA-23", "Payment Adjusted - Timely Filing",         "Timely Filing"),
    ("CO-50", "Non-Covered Service",                      "Clinical"),
    ("CO-167","Diagnosis Not Covered",                    "Clinical"),
    ("CO-29", "Timely Filing Exceeded",                   "Timely Filing"),
    ("CO-109","Claim Not Covered by Payer",               "Administrative"),
    ("CO-18", "Duplicate Claim",                          "Duplicate"),
]

PLACE_OF_SERVICE = {
    "11": "Office",
    "21": "Inpatient Hospital",
    "22": "Outpatient Hospital",
    "23": "Emergency Room",
    "02": "Telehealth",
    "24": "Ambulatory Surgical Center",
    "31": "Skilled Nursing Facility",
}

DRUG_LIST = [
    ("00071015523", "Lisinopril 10mg",      "Lisinopril",    "ACE Inhibitor",      "10mg",  "mg", "Oral",    "Once Daily",   30, 30, 3),
    ("00093505698", "Losartan 50mg",        "Losartan",      "ARB",                "50mg",  "mg", "Oral",    "Once Daily",   30, 30, 3),
    ("00378207205", "Metoprolol 25mg",      "Metoprolol",    "Beta Blocker",       "25mg",  "mg", "Oral",    "Twice Daily",  30, 60, 3),
    ("00071101523", "Atorvastatin 40mg",    "Atorvastatin",  "Statin",             "40mg",  "mg", "Oral",    "Once Daily",   30, 30, 3),
    ("00093505801", "Rosuvastatin 20mg",    "Rosuvastatin",  "Statin",             "20mg",  "mg", "Oral",    "Once Daily",   30, 30, 3),
    ("00093101401", "Metformin 500mg",      "Metformin",     "Biguanide",          "500mg", "mg", "Oral",    "Twice Daily",  30, 60, 5),
    ("00088502001", "Lantus 100u/mL",       "Insulin Glargine","Insulin",          "100u",  "u",  "SubQ",    "Once Daily",   30, 10, 2),
    ("00088250033", "Humalog 100u/mL",      "Insulin Lispro","Insulin",            "100u",  "u",  "SubQ",    "Three Times",  30, 30, 2),
    ("00049490023", "Sertraline 50mg",      "Sertraline",    "SSRI",               "50mg",  "mg", "Oral",    "Once Daily",   30, 30, 5),
    ("00456200063", "Escitalopram 10mg",    "Escitalopram",  "SSRI",               "10mg",  "mg", "Oral",    "Once Daily",   30, 30, 5),
    ("00093083205", "Omeprazole 20mg",      "Omeprazole",    "PPI",                "20mg",  "mg", "Oral",    "Once Daily",   30, 30, 3),
    ("00173068200", "Albuterol 90mcg",      "Albuterol",     "SABA",               "90mcg", "mcg","Inhaled",  "PRN",         30, 200, 2),
    ("00074455490", "Levothyroxine 50mcg",  "Levothyroxine", "Thyroid Hormone",    "50mcg", "mcg","Oral",    "Once Daily",   30, 30, 5),
    ("00069152041", "Amlodipine 5mg",       "Amlodipine",    "CCB",                "5mg",   "mg", "Oral",    "Once Daily",   30, 30, 3),
    ("00071020523", "Furosemide 40mg",      "Furosemide",    "Loop Diuretic",      "40mg",  "mg", "Oral",    "Once Daily",   30, 30, 3),
    ("00056017975", "Warfarin 5mg",         "Warfarin",      "Anticoagulant",      "5mg",   "mg", "Oral",    "Once Daily",   30, 30, 3),
    ("59148001705", "Apixaban 5mg",         "Apixaban",      "DOAC",               "5mg",   "mg", "Oral",    "Twice Daily",  30, 60, 3),
    ("00093101601", "Gabapentin 300mg",     "Gabapentin",    "Anticonvulsant",     "300mg", "mg", "Oral",    "Three Times",  30, 90, 3),
    ("00406051201", "Hydrocodone 5/325mg",  "Hydrocodone",   "Opioid Analgesic",   "5mg",   "mg", "Oral",    "Every 6 hrs",  7,  28, 0),
    ("00093416401", "Amoxicillin 500mg",    "Amoxicillin",   "Antibiotic",         "500mg", "mg", "Oral",    "Three Times",  10, 30, 0),
    ("00069305041", "Azithromycin 250mg",   "Azithromycin",  "Antibiotic",         "250mg", "mg", "Oral",    "Once Daily",   5,  6,  0),
    ("00006001254", "Lisinopril/HCTZ 20mg", "Lisinopril/HCTZ","ACE+Diuretic",     "20mg",  "mg", "Oral",    "Once Daily",   30, 30, 3),
]

LAB_PANELS = {
    "CBC": [
        ("6690-2",  "WBC",         4.5,  11.0, "K/uL"),
        ("789-8",   "RBC",         4.2,   5.8, "M/uL"),
        ("718-7",   "Hemoglobin",  12.0,  17.5, "g/dL"),
        ("4544-3",  "Hematocrit",  36.0,  52.0, "%"),
        ("777-3",   "Platelets",  150.0, 400.0, "K/uL"),
        ("736-9",   "Lymphocytes", 20.0,  40.0, "%"),
        ("5905-5",  "Monocytes",    2.0,  10.0, "%"),
    ],
    "CMP": [
        ("2951-2",  "Sodium",      136.0, 145.0, "mEq/L"),
        ("2823-3",  "Potassium",     3.5,   5.1, "mEq/L"),
        ("2075-0",  "Chloride",     98.0, 107.0, "mEq/L"),
        ("1963-8",  "Bicarbonate",  22.0,  29.0, "mEq/L"),
        ("3094-0",  "BUN",           7.0,  25.0, "mg/dL"),
        ("2160-0",  "Creatinine",    0.6,   1.2, "mg/dL"),
        ("2345-7",  "Glucose",      70.0, 100.0, "mg/dL"),
        ("1742-6",  "ALT",           7.0,  56.0, "U/L"),
        ("1920-8",  "AST",          10.0,  40.0, "U/L"),
        ("1751-7",  "Albumin",       3.5,   5.0, "g/dL"),
        ("2093-3",  "Total Protein", 6.3,   8.2, "g/dL"),
    ],
    "LIPID": [
        ("2093-3",  "Total Cholesterol", 0.0, 200.0, "mg/dL"),
        ("2571-8",  "Triglycerides",     0.0, 150.0, "mg/dL"),
        ("2085-9",  "HDL Cholesterol",  40.0, 999.0, "mg/dL"),
        ("13457-7", "LDL Cholesterol",   0.0, 100.0, "mg/dL"),
    ],
    "OTHER": [
        ("4548-4",  "HbA1c",         4.0,   5.6, "%"),
        ("3016-3",  "TSH",           0.4,   4.0, "mIU/L"),
        ("2857-1",  "PSA",           0.0,   4.0, "ng/mL"),
        ("5902-2",  "PT/INR",        0.8,   1.2, "ratio"),
        ("6598-7",  "Troponin I",    0.0,   0.04,"ng/mL"),
        ("42637-9", "BNP",           0.0, 100.0, "pg/mL"),
        ("5767-9",  "Urinalysis pH", 4.5,   8.0, "pH"),
    ],
}

PERFORMING_LABS = ["Quest Diagnostics", "LabCorp", "Hospital Lab", "In-Office Lab", "BioReference"]

DISCHARGE_DISPOSITIONS = ["Home", "SNF", "Inpatient Rehab", "Home Health", "Hospice", "Expired", "AMA"]

DRG_CODES = [
    ("291", "Heart Failure and Shock with MCC",          12000),
    ("292", "Heart Failure and Shock with CC",            8500),
    ("470", "Major Joint Replacement Lower Extremity",   15000),
    ("193", "Simple Pneumonia with MCC",                  9000),
    ("194", "Simple Pneumonia with CC",                   6500),
    ("871", "Septicemia with MV >96 hrs",                35000),
    ("872", "Septicemia without MV >96 hrs with MCC",    18000),
    ("065", "Intracranial Hemorrhage with MCC",          22000),
    ("392", "Esophagitis GI and Misc Digestive Disorders", 5500),
    ("603", "Cellulitis with MCC",                        7000),
    ("682", "Renal Failure with MCC",                    11000),
    ("683", "Renal Failure with CC",                      7500),
    ("190", "COPD with MCC",                              8000),
    ("191", "COPD with CC",                               5500),
    ("310", "Cardiac Arrhythmia with MCC",                8500),
]

# ==========================================================
# UTILITIES
# ==========================================================
def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))

def make_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:10]}"

def rand_npi():
    return str(random.randint(1000000000, 9999999999))

def rand_ein():
    return f"{random.randint(10,99)}-{random.randint(1000000,9999999)}"

def rand_member_id():
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
    return "".join(random.choices(chars, k=10))

def to_date(val):
    """Safely convert str, date, datetime, or pandas Timestamp to a date object."""
    if val is None:
        return None
    if hasattr(val, 'date') and callable(val.date):
        return val.date()
    if hasattr(val, 'year'):
        return val
    return pd.to_datetime(val).date()

def add_days(val, days):
    """Add days to any date-like value, returning a date object."""
    d = to_date(val)
    if d is None:
        return None
    return (datetime.combine(d, datetime.min.time()) + timedelta(days=days)).date()

def abnormal_flag(value, low, high):
    if value < low * 0.8:
        return "Critical Low"
    elif value < low:
        return "Low"
    elif value > high * 1.2:
        return "Critical High"
    elif value > high:
        return "High"
    return "Normal"


import shutil as _shutil

def _table_dir(name, output_dir):
    p = Path(output_dir) / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_reference_table(df, name, output_dir):
    """Small lookup tables: single CSV + Parquet pair."""
    d = _table_dir(name, output_dir)
    for c in df.columns:
        if df[c].dtype == object and any(k in c.lower() for k in ("date","created_at")):
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime("%Y-%m-%d")
    df.to_csv(d / (name + ".csv"), index=False)
    df.to_parquet(d / (name + ".parquet"), index=False)
    print("  " + name + ": " + str(len(df)) + " rows")


def stream_to_chunks(row_generator, name, output_dir, chunk_size=None):
    """Stream rows to numbered chunk files (CSV or Parquet per chunk)."""
    if chunk_size is None:
        chunk_size = CONFIG["chunk_size"]
    d = _table_dir(name, output_dir)
    csv_ratio = CONFIG["csv_chunk_ratio"]
    total = 0
    part_num = 0
    chunk = []
    _shutil.rmtree(d); d.mkdir(parents=True)
    def _write(rows):
        nonlocal part_num, total
        df = pd.DataFrame(rows)
        for c in df.columns:
            if df[c].dtype == object and any(k in c.lower() for k in ("date","created_at","recorded_at")):
                df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime("%Y-%m-%d")
        fname = name + "_part_" + str(part_num).zfill(3)
        if random.random() < csv_ratio:
            df.to_csv(d / (fname + ".csv"), index=False)
        else:
            df.to_parquet(d / (fname + ".parquet"), index=False)
        total += len(df); part_num += 1
    for row in row_generator:
        chunk.append(row)
        if len(chunk) >= chunk_size:
            _write(chunk); chunk = []
    if chunk: _write(chunk)
    csv_n = len(list(d.glob("*.csv")))
    pq_n  = len(list(d.glob("*.parquet")))
    print("  " + name + ": " + str(total) + " rows -> " + str(part_num) + " chunks (" + str(csv_n) + " CSV + " + str(pq_n) + " Parquet)")
    return total


def load_table(name, output_dir):
    """Reload chunked table into a single DataFrame."""
    d = Path(output_dir) / name
    frames = []
    for f in sorted(d.glob(name + "_part_*.csv")):
        frames.append(pd.read_csv(f, low_memory=False))
    for f in sorted(d.glob(name + "_part_*.parquet")):
        frames.append(pd.read_parquet(f))
    for f in [d / (name+".csv"), d / (name+".parquet")]:
        if f.exists():
            frames.append(pd.read_csv(f, low_memory=False) if f.suffix==".csv" else pd.read_parquet(f))
    if not frames: raise FileNotFoundError("No data for " + repr(name))
    return pd.concat(frames, ignore_index=True)


# GENERATORS
# ==========================================================
def generate_clinics(count):
    rows = []
    for _ in range(count):
        ctype = random.choice(CLINIC_TYPES)
        rows.append({
            "clinic_id":       make_id("CLN"),
            "clinic_name":     fake.company() + " " + random.choice(["Medical Center","Health System","Clinic","Hospital","Physicians Group"]),
            "npi":             rand_npi(),
            "tax_id":          rand_ein(),
            "address":         fake.street_address(),
            "city":            fake.city(),
            "state":           fake.state_abbr(),
            "zip_code":        fake.zipcode(),
            "phone":           fake.phone_number(),
            "fax":             fake.phone_number(),
            "clinic_type":     ctype,
            "bed_count":       random.randint(50, 800) if ctype == "Hospital" else None,
            "accreditation":   random.choice(ACCREDITATIONS),
            "emr_system":      random.choice(EMR_SYSTEMS),
            "network_status":  random.choices(["In-Network","Out-of-Network"], weights=[85,15])[0],
            "created_at":      fake.date_time_between(start_date="-10y", end_date="-1y"),
        })
    return pd.DataFrame(rows)



def generate_providers(count, clinics_df):
    rows = []
    clinic_ids = clinics_df['clinic_id'].tolist()
    for _ in range(count):
        spec = random.choice(SPECIALTIES)
        cred = random.choice(CREDENTIALS)
        rows.append({
            "provider_id":           make_id("PRV"),
            "npi":                   rand_npi(),
            "first_name":            fake.first_name(),
            "last_name":             "Dr. " + fake.last_name(),
            "credential":            cred,
            "specialty":             spec[0],
            "sub_specialty":         random.choice([None, "General", "Interventional", "Pediatric", "Geriatric"]),
            "taxonomy_code":         spec[1],
            "clinic_id":             random.choice(clinic_ids),
            "dea_number":            f"A{fake.lexify('?')}{random.randint(1000000,9999999)}",
            "state_license":         fake.state_abbr() + str(random.randint(100000,999999)),
            "phone":                 fake.phone_number(),
            "email":                 fake.email(),
            "status":                random.choices(["Active","Inactive","Suspended"], weights=[90,8,2])[0],
            "board_certified":       random.choices([True, False], weights=[80,20])[0],
            "years_experience":      random.randint(1, 40),
            "accepting_new_patients":random.choices([True, False], weights=[70,30])[0],
            "created_at":            fake.date_time_between(start_date="-8y", end_date="-1y"),
        })
    return pd.DataFrame(rows)



def generate_payers():
    rows = []
    for p in PAYERS:
        rows.append({
            "payer_id":                    make_id("PAY"),
            "payer_name":                  p[0],
            "payer_type":                  p[1],
            "payer_code":                  p[2],
            "electronic_payer_id":         p[3],
            "contact_number":              fake.phone_number(),
            "claims_address":              fake.address().replace("\n", ", "),
            "timely_filing_limit_days":    random.choice([90, 180, 365]),
            "supports_electronic_claims":  p[1] != "Self-Pay",
        })
    return pd.DataFrame(rows)



def generate_patients(count):
    rows = []
    # Age distribution skewed toward older (Medicare population)
    age_weights = [2]*18 + [5]*22 + [10]*20 + [15]*15 + [20]*10 + [25]*5  # rough weights by decade
    ages = np.random.choice(range(1, 91), size=count,
                            p=np.array([1]*90, dtype=float) / 90)
    # Skew toward 45-85
    ages = np.clip(np.random.normal(loc=58, scale=18, size=count).astype(int), 1, 95)
    genders = np.random.choice(["Male","Female","Non-Binary"], size=count, p=[0.48,0.50,0.02])
    races   = np.random.choice(["White","Black or African American","Asian","Hispanic or Latino",
                                 "American Indian","Native Hawaiian","Two or More Races","Unknown"],
                                size=count, p=[0.60,0.13,0.06,0.13,0.01,0.01,0.03,0.03])
    smoking = np.random.choice(["Never","Former","Current","Unknown"], size=count, p=[0.55,0.25,0.15,0.05])
    bmi_cat = np.random.choice(["Underweight","Normal","Overweight","Obese","Morbidly Obese"],
                                size=count, p=[0.02,0.30,0.33,0.25,0.10])
    emp_status = np.random.choice(["Employed","Unemployed","Retired","Disabled","Student","Unknown"],
                                   size=count, p=[0.45,0.08,0.25,0.10,0.05,0.07])
    now = datetime.now()
    mrns = random.sample(range(10000000, 99999999), count)
    for i in range(count):
        g = genders[i]
        dob = now - timedelta(days=int(ages[i])*365 + random.randint(0,364))
        rows.append({
            "patient_id":         make_id("PAT"),
            "mrn":                mrns[i],
            "first_name":         fake.first_name_male() if g=="Male" else fake.first_name_female(),
            "last_name":          fake.last_name(),
            "dob":                dob.date(),
            "age":                ages[i],
            "gender":             g,
            "ssn_last4":          str(random.randint(1000,9999)),
            "phone":              fake.phone_number(),
            "email":              fake.email(),
            "address":            fake.street_address(),
            "city":               fake.city(),
            "state":              fake.state_abbr(),
            "zip_code":           fake.zipcode(),
            "race":               races[i],
            "ethnicity":          random.choice(["Non-Hispanic","Hispanic"]),
            "marital_status":     random.choice(["Single","Married","Divorced","Widowed","Separated"]),
            "preferred_language": random.choices(["English","Spanish","Mandarin","French","Other"],
                                                  weights=[80,12,2,1,5])[0],
            "employment_status":  emp_status[i],
            "income_bracket":     random.choice(["<$25k","$25k-$50k","$50k-$75k","$75k-$100k",">$100k"]),
            "smoking_status":     smoking[i],
            "bmi_category":       bmi_cat[i],
            "created_at":         fake.date_time_between(start_date="-5y", end_date="-6m"),
            "updated_at":         fake.date_time_between(start_date="-6m", end_date="now"),
        })
    return pd.DataFrame(rows)



def generate_patient_insurance(patients_df, payers_df):
    rows = []
    payer_ids   = payers_df['payer_id'].tolist()
    payer_names = dict(zip(payers_df['payer_id'], payers_df['payer_name']))
    payer_types = dict(zip(payers_df['payer_id'], payers_df['payer_type']))
    for _, pat in patients_df.iterrows():
        # Primary insurance
        pid = random.choice(payer_ids)
        plan_type = random.choice(PLAN_TYPES)
        eff = fake.date_between(start_date='-5y', end_date='-1m')
        rows.append({
            "patient_insurance_id": make_id("PIN"),
            "patient_id":           pat['patient_id'],
            "payer_id":             pid,
            "insurance_priority":   "Primary",
            "member_id":            rand_member_id(),
            "group_number":         str(random.randint(10000, 999999)),
            "plan_name":            payer_names[pid] + f" {plan_type} Plan",
            "plan_type":            plan_type,
            "effective_date":       eff,
            "termination_date":     fake.date_between(start_date='+6m', end_date='+3y'),
            "copay_office":         round(random.choice([0,10,15,20,25,30,35,40,50]), 2),
            "copay_specialist":     round(random.choice([20,30,40,50,60,75]), 2),
            "copay_er":             round(random.choice([100,150,200,250,300,350]), 2),
            "deductible_individual":round(random.choice([0,500,1000,1500,2000,3000,5000,6000,7000]), 2),
            "deductible_family":    round(random.choice([0,1000,2000,3000,5000,8000,12000,14000]), 2),
            "deductible_met":       round(random.uniform(0, 3000), 2),
            "out_of_pocket_max":    round(random.choice([3000,5000,7000,8700,10000,12000]), 2),
            "out_of_pocket_met":    round(random.uniform(0, 5000), 2),
            "coinsurance_pct":      random.choice([0,10,20,30]),
            "requires_referral":    plan_type in ["HMO","Medicaid Managed Care"],
            "requires_prior_auth":  random.choices([True,False], weights=[40,60])[0],
            "created_at":           fake.date_time_between(start_date="-5y", end_date="-1m"),
        })
        # ~30% have secondary insurance
        if random.random() < 0.30:
            pid2 = random.choice([p for p in payer_ids if p != pid])
            rows.append({
                "patient_insurance_id": make_id("PIN"),
                "patient_id":           pat['patient_id'],
                "payer_id":             pid2,
                "insurance_priority":   "Secondary",
                "member_id":            rand_member_id(),
                "group_number":         str(random.randint(10000, 999999)),
                "plan_name":            payer_names[pid2] + " Secondary Plan",
                "plan_type":            random.choice(PLAN_TYPES),
                "effective_date":       eff,
                "termination_date":     fake.date_between(start_date='+6m', end_date='+3y'),
                "copay_office":         0,
                "copay_specialist":     0,
                "copay_er":             0,
                "deductible_individual":0,
                "deductible_family":    0,
                "deductible_met":       0,
                "out_of_pocket_max":    0,
                "out_of_pocket_met":    0,
                "coinsurance_pct":      0,
                "requires_referral":    False,
                "requires_prior_auth":  False,
                "created_at":           fake.date_time_between(start_date="-5y", end_date="-1m"),
            })
    return pd.DataFrame(rows)



def generate_appointments(min_records, max_records, patients_df, providers_df, clinics_df):
    total = random.randint(min_records, max_records)
    print(f"  Generating {total:,} appointments...")
    start_dt = datetime.strptime(CONFIG['start_date'], "%Y-%m-%d")
    end_dt   = datetime.strptime(CONFIG['end_date'],   "%Y-%m-%d")
    delta_days = (end_dt - start_dt).days

    pat_ids  = patients_df['patient_id'].tolist()
    prov_ids = providers_df['provider_id'].tolist()
    prov_clinic = dict(zip(providers_df['provider_id'], providers_df['clinic_id']))

    statuses, s_weights = APPOINTMENT_STATUSES_WEIGHTS
    cancel_reasons = ["Patient Request","Provider Unavailable","Weather","Insurance Issue","No Reason Given",None]
    hours   = [f"{h:02d}:{m:02d}" for h in range(8,18) for m in [0,15,30,45]]
    durations = [15,20,30,45,60,90]

    for _ in range(total):
        prov_id = random.choice(prov_ids)
        status  = random.choices(statuses, weights=s_weights)[0]
        appt_dt = start_dt + timedelta(days=random.randint(0, delta_days))
        yield {
            "appointment_id":     make_id("APT"),
            "patient_id":         random.choice(pat_ids),
            "provider_id":        prov_id,
            "clinic_id":          prov_clinic[prov_id],
            "appointment_date":   appt_dt.date(),
            "appointment_time":   random.choice(hours),
            "appointment_type":   random.choice(APPOINTMENT_TYPES),
            "appointment_status": status,
            "visit_reason":       fake.sentence(nb_words=random.randint(4,8)),
            "duration_minutes":   random.choice(durations),
            "referral_id":        make_id("REF") if random.random() < 0.15 else None,
            "cancelled_reason":   random.choice(cancel_reasons) if status in ["Cancelled","No Show"] else None,
            "created_at":         appt_dt - timedelta(days=random.randint(1,30)),
        }



def generate_encounters(appointments_df, clinics_df):
    completed = appointments_df[appointments_df['appointment_status'] == 'Completed']
    print(f"  Generating {len(completed):,} encounters...")
    clinic_types = dict(zip(clinics_df['clinic_id'], clinics_df['clinic_type']))
    pos_map = {"Hospital":"21","Urgent Care":"23","ASC":"24","FQHC":"11","Private Practice":"11","Specialty Clinic":"22"}
    chief_complaints = [
        "Chest pain and shortness of breath","Uncontrolled blood sugar","Routine follow-up for hypertension",
        "Knee pain worsening over 3 months","Annual wellness exam","Cough and fever for 5 days",
        "Lower back pain radiating to left leg","Palpitations and dizziness","Fatigue and weight gain",
        "Abdominal pain nausea vomiting","Headache and blurred vision","Skin rash and itching",
        "Urinary frequency and burning","Anxiety and insomnia","Follow-up post hospitalization",
    ]
    for _, appt in completed.iterrows():
        ctype = clinic_types.get(appt['clinic_id'], "Private Practice")
        pos   = pos_map.get(ctype, "11")
        is_inpatient = pos == "21"
        los   = random.randint(1,14) if is_inpatient else None
        admit_dt = to_date(appt['appointment_date']) if is_inpatient else None
        disch_dt = add_days(admit_dt, los) if is_inpatient and admit_dt else None
        drg   = random.choice(DRG_CODES) if is_inpatient else None
        yield {
            "encounter_id":          make_id("ENC"),
            "appointment_id":        appt['appointment_id'],
            "patient_id":            appt['patient_id'],
            "provider_id":           appt['provider_id'],
            "clinic_id":             appt['clinic_id'],
            "encounter_date":        appt['appointment_date'],
            "encounter_type":        appt['appointment_type'],
            "chief_complaint":       random.choice(chief_complaints),
            "hpi":                   fake.paragraph(nb_sentences=3),
            "assessment":            fake.sentence(nb_words=12),
            "plan":                  fake.paragraph(nb_sentences=2),
            "place_of_service_code": pos,
            "discharge_disposition": random.choice(DISCHARGE_DISPOSITIONS) if is_inpatient else None,
            "los_days":              los,
            "drg_code":              drg[0] if drg else None,
            "drg_description":       drg[1] if drg else None,
            "drg_base_rate":         drg[2] if drg else None,
            "admit_date":            admit_dt,
            "discharge_date":        disch_dt,
            "created_at":            appt['created_at'],
        }



def generate_vitals(encounters_df):
    print(f"  Generating vitals for {len(encounters_df):,} encounters...")
    for _, enc in encounters_df.iterrows():
        ht  = round(random.gauss(67, 4), 1)
        wt  = round(random.gauss(185, 45), 1)
        wt  = max(90, min(450, wt))
        ht  = max(54, min(80, ht))
        bmi = round(703 * wt / (ht ** 2), 1)
        yield {
            "vital_id":          make_id("VIT"),
            "encounter_id":      enc['encounter_id'],
            "patient_id":        enc['patient_id'],
            "recorded_at":       enc['encounter_date'],
            "height_inches":     ht,
            "weight_lbs":        wt,
            "bmi":               bmi,
            "systolic_bp":       random.randint(90, 190),
            "diastolic_bp":      random.randint(55, 115),
            "heart_rate":        random.randint(48, 130),
            "respiratory_rate":  random.randint(12, 28),
            "temperature_f":     round(random.gauss(98.6, 1.2), 1),
            "oxygen_saturation": random.randint(88, 100),
            "pain_scale":        random.randint(0, 10),
            "recorded_by":       fake.name(),
        }


def generate_diagnoses(encounters_df):
    print(f"  Generating diagnoses...")
    poa_choices = ["Y","N","U","W"]
    for _, enc in encounters_df.iterrows():
        n = random.randint(1, 4)
        selected = random.sample(ICD10_CODES, min(n, len(ICD10_CODES)))
        for idx, diag in enumerate(selected):
            yield {
                "diagnosis_id":          make_id("DGN"),
                "encounter_id":          enc['encounter_id'],
                "patient_id":            enc['patient_id'],
                "icd10_code":            diag[0],
                "diagnosis_description": diag[1],
                "diagnosis_type":        "Primary" if idx == 0 else random.choice(["Secondary","Admitting"]),
                "chronic_flag":          diag[2],
                "hcc_code":              diag[3],
                "poa_indicator":         random.choices(poa_choices, weights=[75,10,10,5])[0],
                "created_at":            enc['encounter_date'],
            }



def generate_procedures(encounters_df):
    print(f"  Generating procedures...")
    revenue_codes = ["0300","0301","0360","0450","0481","0510","0636","0710","0730","0761"]
    for _, enc in encounters_df.iterrows():
        n = random.randint(1, 4)
        selected = random.sample(CPT_CODES, min(n, len(CPT_CODES)))
        for proc in selected:
            units = random.randint(1, 3)
            yield {
                "procedure_id":          make_id("PRC"),
                "encounter_id":          enc['encounter_id'],
                "patient_id":            enc['patient_id'],
                "provider_id":           enc['provider_id'],
                "cpt_code":              proc[0],
                "procedure_description": proc[1],
                "modifier":              random.choice(MODIFIERS),
                "units":                 units,
                "procedure_charge":      round(proc[2] * units * random.uniform(0.95, 1.10), 2),
                "revenue_code":          random.choice(revenue_codes),
                "ndc_code":              random.choice(DRUG_LIST)[0] if random.random() < 0.15 else None,
                "anesthesia_minutes":    random.randint(30,240) if "arthroplasty" in proc[1].lower() or "CABG" in proc[1] else None,
                "created_at":            enc['encounter_date'],
            }



def generate_lab_results(encounters_df):
    print(f"  Generating lab results...")
    enc_with_labs = encounters_df.sample(frac=0.60, random_state=42)
    for _, enc in enc_with_labs.iterrows():
        panels = random.sample(list(LAB_PANELS.keys()), random.randint(1, 3))
        order_dt = to_date(enc['encounter_date'])
        result_dt = add_days(order_dt, random.randint(0, 2))
        for panel in panels:
            for test in LAB_PANELS[panel]:
                loinc, name, low, high, unit = test
                mean = (low + high) / 2
                std  = (high - low) / 4
                val  = round(random.gauss(mean, std), 2)
                flag = abnormal_flag(val, low, high)
                yield {
                    "lab_id":              make_id("LAB"),
                    "encounter_id":        enc['encounter_id'],
                    "patient_id":          enc['patient_id'],
                    "order_date":          order_dt,
                    "result_date":         result_dt,
                    "lab_name":            panel,
                    "loinc_code":          loinc,
                    "test_name":           name,
                    "result_value":        val,
                    "result_unit":         unit,
                    "reference_range_low": low,
                    "reference_range_high":high,
                    "abnormal_flag":       flag,
                    "status":              random.choices(["Final","Preliminary","Corrected"], weights=[90,7,3])[0],
                    "ordering_provider_id":enc['provider_id'],
                    "performing_lab":      random.choice(PERFORMING_LABS),
                }


def generate_medications(encounters_df):
    print(f"  Generating medications...")
    enc_with_meds = encounters_df.sample(frac=0.70, random_state=7)
    pharmacies = ["CVS Pharmacy","Walgreens","Rite Aid","Walmart Pharmacy","Hospital Pharmacy","Mail Order"]
    for _, enc in enc_with_meds.iterrows():
        n_meds = random.randint(1, 4)
        selected_drugs = random.sample(DRUG_LIST, min(n_meds, len(DRUG_LIST)))
        for drug in selected_drugs:
            ndc, brand, generic, drug_class, dose, dose_unit, route, freq, days, qty, refills = drug
            pa_req = drug_class in ["Insulin","DOAC","Opioid Analgesic"] or random.random() < 0.10
            prescribed_dt = to_date(enc['encounter_date'])
            fill_dt = add_days(prescribed_dt, random.randint(0, 3))
            yield {
                "medication_id":       make_id("MED"),
                "encounter_id":        enc['encounter_id'],
                "patient_id":          enc['patient_id'],
                "ndc_code":            ndc,
                "drug_name":           brand,
                "generic_name":        generic,
                "drug_class":          drug_class,
                "dose":                dose,
                "dose_unit":           dose_unit,
                "route":               route,
                "frequency":           freq,
                "days_supply":         days,
                "quantity":            qty,
                "refills":             refills,
                "prescriber_id":       enc['provider_id'],
                "pharmacy_name":       random.choice(pharmacies),
                "rx_number":           str(random.randint(1000000, 9999999)),
                "prescribed_date":     prescribed_dt,
                "fill_date":           fill_dt,
                "status":              random.choices(["Active","Discontinued","On Hold"], weights=[75,20,5])[0],
                "prior_auth_required": pa_req,
                "prior_auth_number":   make_id("PA") if pa_req and random.random() < 0.8 else None,
            }


def generate_prior_authorizations(encounters_df, patients_df, providers_df, payers_df):
    print(f"  Generating prior authorizations...")
    payer_ids= payers_df['payer_id'].tolist()
    enc_sample = encounters_df.sample(frac=0.20, random_state=11)
    for _, enc in enc_sample.iterrows():
        svc_type = random.choice(["Procedure","Medication","Referral","DME","Imaging"])
        req_dt   = to_date(enc['encounter_date'])
        dec_days = random.randint(1, 14)
        dec_dt   = add_days(req_dt, dec_days)
        status   = random.choices(["Approved","Denied","Pending","Expired","Appealed"], weights=[65,15,10,5,5])[0]
        yield {
            "auth_id":            make_id("AUT"),
            "patient_id":         enc['patient_id'],
            "provider_id":        enc['provider_id'],
            "payer_id":           random.choice(payer_ids),
            "service_type":       svc_type,
            "cpt_code":           random.choice(CPT_CODES)[0] if svc_type in ["Procedure","Imaging"] else None,
            "ndc_code":           random.choice(DRUG_LIST)[0] if svc_type == "Medication" else None,
            "icd10_code":         random.choice(ICD10_CODES)[0],
            "request_date":       req_dt,
            "decision_date":      dec_dt,
            "status":             status,
            "auth_number":        str(random.randint(10000000,99999999)) if status == "Approved" else None,
            "approved_units":     random.randint(1,10) if status == "Approved" else None,
            "approved_from_date": dec_dt if status == "Approved" else None,
            "approved_to_date":   add_days(dec_dt, random.choice([30,60,90,180,365])) if status == "Approved" else None,
            "denial_reason":      random.choice(["Medical Necessity","Not Covered","Experimental","Insufficient Documentation"]) if status == "Denied" else None,
            "urgency":            random.choices(["Routine","Urgent","Emergent"], weights=[70,25,5])[0],
            "clinical_notes":     fake.paragraph(nb_sentences=2),
        }



def generate_claims(encounters_df, procedures_df, patient_ins_df, providers_df, clinics_df):
    print(f"  Generating claims for {len(encounters_df):,} encounters...")
    ins_by_patient = {}
    for _, row in patient_ins_df[patient_ins_df['insurance_priority']=='Primary'].iterrows():
        ins_by_patient[row['patient_id']] = row

    sec_ins_by_patient = {}
    for _, row in patient_ins_df[patient_ins_df['insurance_priority']=='Secondary'].iterrows():
        sec_ins_by_patient[row['patient_id']] = row

    proc_charge_by_enc = procedures_df.groupby('encounter_id')['procedure_charge'].sum().to_dict()
    prov_npi   = dict(zip(providers_df['provider_id'], providers_df['npi']))
    clinic_npi = dict(zip(clinics_df['clinic_id'], clinics_df['npi']))

    claim_statuses = ["Paid","Pending","Denied","Rejected","Appealed","Void","Adjusted"]
    claim_weights  = [65, 10, 13, 4, 4, 2, 2]
    claim_types    = ["Professional","Institutional"]
    filing_ind     = ["Electronic","Paper"]

    for _, enc in encounters_df.iterrows():
        ins = ins_by_patient.get(enc['patient_id'])
        if ins is None:
            continue
        sec_ins = sec_ins_by_patient.get(enc['patient_id'])
        total_charge = proc_charge_by_enc.get(enc['encounter_id'], 0)
        status = random.choices(claim_statuses, weights=claim_weights)[0]
        sub_dt = add_days(enc['encounter_date'], random.randint(1, 7))
        adj_dt = add_days(sub_dt, random.randint(5, 45))
        yield {
            "claim_id":                make_id("CLM"),
            "encounter_id":            enc['encounter_id'],
            "patient_id":              enc['patient_id'],
            "provider_id":             enc['provider_id'],
            "clinic_id":               enc['clinic_id'],
            "payer_id":                ins['payer_id'],
            "secondary_payer_id":      sec_ins['payer_id'] if sec_ins is not None else None,
            "claim_number":            str(random.randint(100000000, 999999999)),
            "claim_type":              random.choices(claim_types, weights=[70,30])[0],
            "total_charge":            round(total_charge, 2),
            "claim_status":            status,
            "submission_date":         sub_dt,
            "adjudication_date":       adj_dt,
            "filing_indicator":        random.choices(filing_ind, weights=[92,8])[0],
            "place_of_service":        enc['place_of_service_code'],
            "prior_auth_number":       str(random.randint(10000000,99999999)) if random.random()<0.20 else None,
            "referring_provider_npi":  rand_npi() if random.random()<0.25 else None,
            "rendering_provider_npi":  prov_npi.get(enc['provider_id'], rand_npi()),
            "billing_provider_npi":    clinic_npi.get(enc['clinic_id'], rand_npi()),
            "facility_npi":            clinic_npi.get(enc['clinic_id']) if enc['place_of_service_code'] in ["21","22","23"] else None,
            "original_claim_number":   str(random.randint(100000000,999999999)) if status in ["Adjusted","Void"] else None,
            "created_at":              sub_dt,
        }



def generate_claim_lines(claims_df, procedures_df, diagnoses_df):
    print(f"  Generating claim lines...")
    procs_by_enc = procedures_df.groupby('encounter_id')
    diags_by_enc = diagnoses_df.groupby('encounter_id')
    adj_reason_codes = ["CO-45","PR-1","PR-2","PR-3","CO-97","CO-4","CO-11","OA-23","CO-22"]
    remark_codes = ["N30","N115","M20","MA04","N95","N362",None,None,None]
    for _, claim in claims_df.iterrows():
        enc_id = claim['encounter_id']
        try:
            procs = procs_by_enc.get_group(enc_id)
        except KeyError:
            continue
        try:
            diags = diags_by_enc.get_group(enc_id)
            diag_ptr = ",".join([str(i+1) for i in range(min(4, len(diags)))])
        except KeyError:
            diag_ptr = "1"
        for line_num, (_, proc) in enumerate(procs.iterrows(), 1):
            billed  = proc['procedure_charge']
            allowed = round(billed * random.uniform(0.55, 0.95), 2)
            copay   = round(random.choice([0,10,15,20,25,30,35,40,50]), 2)
            ded     = round(random.uniform(0, 200), 2)
            coins   = round((allowed - copay - ded) * random.uniform(0, 0.20), 2)
            pat_resp= round(copay + ded + coins, 2)
            paid    = round(max(0, allowed - pat_resp), 2)
            adj     = round(billed - allowed, 2)
            yield {
                "claim_line_id":          make_id("CLI"),
                "claim_id":               claim['claim_id'],
                "line_number":            line_num,
                "cpt_code":               proc['cpt_code'],
                "revenue_code":           proc.get('revenue_code'),
                "modifier":               proc['modifier'],
                "diagnosis_pointer":      diag_ptr,
                "service_date":           claim['submission_date'],
                "units":                  proc['units'],
                "billed_amount":          round(billed, 2),
                "allowed_amount":         allowed,
                "paid_amount":            paid,
                "patient_responsibility": pat_resp,
                "coinsurance_amount":     coins,
                "copay_amount":           copay,
                "deductible_amount":      ded,
                "adjustment_amount":      adj,
                "adjustment_reason_code": random.choice(adj_reason_codes),
                "line_status":            claim['claim_status'],
                "remark_code":            random.choice(remark_codes),
            }


def generate_remittances(claims_df, payers_df):
    print(f"  Generating remittances (ERA/835)...")
    payer_names = dict(zip(payers_df['payer_id'], payers_df['payer_name']))
    paid_claims = claims_df[claims_df['claim_status'] == 'Paid']
    # Group paid claims into ERA batches by payer + adjudication_date
    groups = paid_claims.groupby(['payer_id','adjudication_date'])
    rows = []
    remit_map = {}  # claim_id -> remittance_id
    for (payer_id, adj_date), grp in groups:
        remit_id = make_id("REM")
        total_paid = round(grp['total_charge'].sum() * random.uniform(0.60, 0.85), 2)
        rows.append({
            "remittance_id":        remit_id,
            "payer_id":             payer_id,
            "check_number":         str(random.randint(10000000,99999999)),
            "eft_trace_number":     str(random.randint(100000000000,999999999999)),
            "payment_date":         adj_date,
            "payment_method":       random.choices(["EFT","Check"], weights=[85,15])[0],
            "total_payment_amount": total_paid,
            "claim_count":          len(grp),
            "provider_npi":         rand_npi(),
            "provider_name":        fake.company() + " Medical Group",
            "payer_name":           payer_names.get(payer_id, "Unknown"),
            "era_file_name":        f"ERA_{payer_id}_{adj_date}.835",
            "created_at":           adj_date,
        })
        for cid in grp['claim_id']:
            remit_map[cid] = remit_id
    return pd.DataFrame(rows), remit_map



def generate_payments(claims_df, remit_map):
    print(f"  Generating payments...")
    rows = []
    paid = claims_df[claims_df['claim_status'] == 'Paid']
    for _, claim in paid.iterrows():
        remit_id = remit_map.get(claim['claim_id'])
        rows.append({
            "payment_id":          make_id("PMT"),
            "claim_id":            claim['claim_id'],
            "remittance_id":       remit_id,
            "payment_date":        claim['adjudication_date'],
            "payment_source":      random.choices(["ERA","Check","Wire","Secondary Insurance"], weights=[75,15,5,5])[0],
            "paid_amount":         round(claim['total_charge'] * random.uniform(0.55, 0.90), 2),
            "check_number":        str(random.randint(10000000,99999999)) if random.random()<0.15 else None,
            "eft_trace_number":    str(random.randint(100000000000,999999999999)),
            "patient_payment_type":None,
            "posted_date":         claim['adjudication_date'],
            "posted_by":           fake.name(),
            "created_at":          claim['adjudication_date'],
        })
    # Patient payments (copay/deductible) — ~40% of all claims
    patient_claims = claims_df.sample(frac=0.40, random_state=3)
    for _, claim in patient_claims.iterrows():
        rows.append({
            "payment_id":          make_id("PMT"),
            "claim_id":            claim['claim_id'],
            "remittance_id":       None,
            "payment_date":        claim['adjudication_date'],
            "payment_source":      "Patient",
            "paid_amount":         round(random.uniform(10, 350), 2),
            "check_number":        None,
            "eft_trace_number":    None,
            "patient_payment_type":random.choice(["Copay","Deductible","Coinsurance","Self-Pay"]),
            "posted_date":         claim['adjudication_date'],
            "posted_by":           "Front Desk",
            "created_at":          claim['adjudication_date'],
        })
    return pd.DataFrame(rows)


def generate_denials(claims_df):
    print(f"  Generating denials...")
    denied = claims_df[claims_df['claim_status'].isin(['Denied','Rejected'])]
    rows = []
    for _, claim in denied.iterrows():
        dc = random.choice(DENIAL_CODES)
        denial_dt = to_date(claim['adjudication_date'])
        appeal_dl = add_days(denial_dt, random.choice([30,60,90,180]))
        rows.append({
            "denial_id":                  make_id("DNL"),
            "claim_id":                   claim['claim_id'],
            "claim_line_id":              None,
            "denial_code":                dc[0],
            "denial_reason":              dc[1],
            "denial_category":            dc[2],
            "denial_date":                denial_dt,
            "appeal_deadline":            appeal_dl,
            "appeal_status":              random.choices(["Not Appealed","Appealed","Overturned","Upheld","Pending"],
                                                          weights=[40,20,15,15,10])[0],
            "corrected_claim_submitted":  random.choices([True,False], weights=[35,65])[0],
            "root_cause":                 random.choice(["Missing Auth","Wrong Payer","Coding Error","Eligibility Issue",
                                                          "Timely Filing","Duplicate","Medical Necessity","Bundling"]),
            "created_at":                 denial_dt,
        })
    return pd.DataFrame(rows)


def generate_appeals(denials_df, claims_df):
    print(f"  Generating appeals...")
    appealed = denials_df[denials_df['appeal_status'].isin(['Appealed','Overturned','Upheld','Pending'])]
    rows = []
    for _, denial in appealed.iterrows():
        sub_dt = to_date(denial['denial_date'])
        dec_dt = add_days(sub_dt, random.randint(15, 90))
        decision = denial['appeal_status'] if denial['appeal_status'] in ['Overturned','Upheld'] else random.choice(['Overturned','Upheld','Partial','Pending'])
        rows.append({
            "appeal_id":                  make_id("APP"),
            "denial_id":                  denial['denial_id'],
            "claim_id":                   denial['claim_id'],
            "patient_id":                 claims_df.loc[claims_df['claim_id']==denial['claim_id'],'patient_id'].values[0] if len(claims_df.loc[claims_df['claim_id']==denial['claim_id']]) > 0 else None,
            "appeal_type":                random.choices(["First Level","Second Level","External Review","Peer to Peer"],
                                                          weights=[55,25,10,10])[0],
            "submitted_date":             sub_dt,
            "decision_date":              dec_dt if decision != "Pending" else None,
            "decision":                   decision,
            "appeal_reason":              random.choice(["Medical Necessity Supported","Coding Corrected","Auth Obtained",
                                                          "Eligibility Confirmed","Timely Filing Exception","Clinical Documentation"]),
            "supporting_docs_submitted":  random.choices([True,False], weights=[80,20])[0],
            "amount_recovered":           round(random.uniform(50, 5000), 2) if decision == "Overturned" else None,
            "created_at":                 sub_dt,
        })
    return pd.DataFrame(rows)


def generate_eligibility_checks(patients_df, payers_df, appointments_df):
    print(f"  Generating eligibility checks...")
    payer_ids   = payers_df['payer_id'].tolist()
    payer_names = dict(zip(payers_df['payer_id'], payers_df['payer_name']))
    for _, appt in appointments_df.sample(frac=0.85, random_state=5).iterrows():
        pid = random.choice(payer_ids)
        active = random.choices([True,False], weights=[90,10])[0]
        yield {
            "eligibility_id":         make_id("ELG"),
            "patient_id":             appt['patient_id'],
            "payer_id":               pid,
            "check_date":             appt['appointment_date'],
            "check_type":             random.choices(["Real-Time","Batch"], weights=[70,30])[0],
            "response_status":        random.choices(["Active","Inactive","Not Found","Error"], weights=[88,5,4,3])[0],
            "coverage_active":        active,
            "plan_name":              payer_names.get(pid,"Unknown") + " Plan",
            "copay_verified":         round(random.choice([0,10,15,20,25,30,35,40,50]), 2),
            "deductible_verified":    round(random.choice([0,500,1000,1500,2000,3000,5000]), 2),
            "deductible_remaining":   round(random.uniform(0, 3000), 2),
            "out_of_pocket_remaining":round(random.uniform(0, 7000), 2),
            "prior_auth_required":    random.choices([True,False], weights=[30,70])[0],
            "referral_required":      random.choices([True,False], weights=[25,75])[0],
            "transaction_id":         uuid.uuid4().hex[:16].upper(),
            "created_at":             appt['appointment_date'],
        }


# ==========================================================
# MAIN
# ==========================================================
def main():
    out = CONFIG['output_dir']
    start = datetime.now()
    print('=' * 60)
    print('  HEALTHCARE & MEDICAL BILLING DATA GENERATOR')
    print('  Chunked folder-per-table output (CSV + Parquet chunks)')
    print('  Each table -> raw_data/<table>/<table>_part_NNN.csv/.parquet')
    print('  Chunk size:', CONFIG['chunk_size'], 'rows')
    print('=' * 60)

    # Reference tables (small, single file pair)
    print('[1/20] Generating clinics...')
    clinics_df = generate_clinics(CONFIG['clinic_count'])
    write_reference_table(clinics_df, 'clinics', out)

    print('[2/20] Generating providers...')
    providers_df = generate_providers(CONFIG['provider_count'], clinics_df)
    write_reference_table(providers_df, 'providers', out)

    print('[3/20] Generating payers...')
    payers_df = generate_payers()
    write_reference_table(payers_df, 'payers', out)

    print('[4/20] Generating patients...')
    patients_df = generate_patients(CONFIG['patient_count'])
    write_reference_table(patients_df, 'patients', out)

    print('[5/20] Generating patient insurance...')
    patient_ins_df = generate_patient_insurance(patients_df, payers_df)
    write_reference_table(patient_ins_df, 'patient_insurance', out)

    # Large tables: stream to chunk files
    print('[6/20] Generating appointments (streaming)...')
    stream_to_chunks(
        generate_appointments(CONFIG['min_records'], CONFIG['max_records'],
                              patients_df, providers_df, clinics_df),
        'appointments', out)

    print('  Reloading appointments...')
    appointments_df = load_table('appointments', out)
    print('  Loaded', len(appointments_df), 'appointments')

    print('[7/20] Generating encounters (streaming)...')
    stream_to_chunks(generate_encounters(appointments_df, clinics_df), 'encounters', out)
    del appointments_df

    print('  Reloading encounters...')
    encounters_df = load_table('encounters', out)
    print('  Loaded', len(encounters_df), 'encounters')

    print('[8/20] Generating vitals (streaming)...')
    stream_to_chunks(generate_vitals(encounters_df), 'vitals', out)

    print('[9/20] Generating diagnoses (streaming)...')
    stream_to_chunks(generate_diagnoses(encounters_df), 'diagnoses', out)

    print('[10/20] Generating procedures (streaming)...')
    stream_to_chunks(generate_procedures(encounters_df), 'procedures', out)

    print('[11/20] Generating lab results (streaming)...')
    stream_to_chunks(generate_lab_results(encounters_df), 'lab_results', out)

    print('[12/20] Generating medications (streaming)...')
    stream_to_chunks(generate_medications(encounters_df), 'medications', out)

    print('[13/20] Generating prior authorizations (streaming)...')
    stream_to_chunks(
        generate_prior_authorizations(encounters_df, patients_df, providers_df, payers_df),
        'prior_authorizations', out)

    print('  Reloading procedures + diagnoses for claims...')
    procedures_df = load_table('procedures', out)
    diagnoses_df  = load_table('diagnoses', out)

    print('[14/20] Generating claims (streaming)...')
    stream_to_chunks(
        generate_claims(encounters_df, procedures_df, patient_ins_df, providers_df, clinics_df),
        'claims', out)
    del encounters_df

    claims_df = load_table('claims', out)
    print('  Loaded', len(claims_df), 'claims')

    print('[15/20] Generating claim lines (streaming)...')
    stream_to_chunks(
        generate_claim_lines(claims_df, procedures_df, diagnoses_df),
        'claim_lines', out)
    del procedures_df, diagnoses_df

    print('[16/20] Generating remittances...')
    remittances_df, remit_map = generate_remittances(claims_df, payers_df)
    write_reference_table(remittances_df, 'remittances', out)

    print('[17/20] Generating payments...')
    payments_df = generate_payments(claims_df, remit_map)
    write_reference_table(payments_df, 'payments', out)

    print('[18/20] Generating denials...')
    denials_df = generate_denials(claims_df)
    write_reference_table(denials_df, 'denials', out)

    print('[19/20] Generating appeals...')
    appeals_df = generate_appeals(denials_df, claims_df)
    write_reference_table(appeals_df, 'appeals', out)
    del claims_df

    print('[20/20] Generating eligibility checks (streaming)...')
    appointments_df = load_table('appointments', out)
    stream_to_chunks(
        generate_eligibility_checks(patients_df, payers_df, appointments_df),
        'eligibility_checks', out)

    elapsed = (datetime.now() - start).seconds
    print('=' * 60)
    print('  DATA GENERATION COMPLETE')
    print('  Output:', out)
    print('  Time elapsed:', elapsed, 's')
    print('=' * 60)


if __name__ == '__main__':
    main()
