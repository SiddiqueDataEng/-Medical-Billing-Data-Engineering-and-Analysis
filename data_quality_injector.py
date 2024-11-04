"""
Enhanced Data Quality Issue Injector - Healthcare Data
Injects 20+ realistic issue types across all 20 tables.
Output uses the same chunked folder-per-table layout as data_generator.py.

Issue types injected:
  1.  missing_value          - NaN/null critical fields
  2.  invalid_format         - bad phone/email/NPI/date/zip/SSN formats
  3.  out_of_range           - numeric values outside valid bounds
  4.  future_date            - dates set in the future
  5.  past_date              - dates unrealistically far in the past
  6.  impossible_values      - biologically/logically impossible numbers
  7.  wrong_data_type        - strings in numeric cols, numbers in text cols
  8.  special_characters     - injected garbage chars in text fields
  9.  duplicate_records      - exact and near-duplicate rows
  10. inconsistent_data      - field values that contradict each other
  11. logical_inconsistency  - broken cross-column business rules
  12. negative_amounts       - negative charges/payments/quantities
  13. swapped_columns        - two related columns have values exchanged
  14. truncated_text         - text cut off mid-value
  15. case_inconsistency     - random UPPER/lower/mIxEd casing
  16. whitespace_pollution   - leading/trailing/embedded extra spaces
  17. encoding_corruption    - latin-1 lookalike chars replacing ASCII
  18. orphan_records         - foreign keys pointing to non-existent IDs
  19. amount_mismatch        - line totals that don't add up to header
  20. date_sequence_error    - end date before start date
  21. referential_break      - cross-table ID replaced with invalid value
  22. stale_status           - status code inconsistent with dates present
"""

import json, random, uuid, shutil
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    "input_dir":  str(Path(__file__).parent / "raw_data"),
    "output_dir": str(Path(__file__).parent / "raw_data_with_issues"),
    "seed": 42,
    # Rates are deliberately heavy so the pipeline has real work to do
    "missing_rate":          0.06,
    "format_rate":           0.04,
    "range_rate":            0.04,
    "date_rate":             0.03,
    "duplicate_rate":        0.04,
    "type_rate":             0.025,
    "special_rate":          0.03,
    "orphan_rate":           0.025,
    "logic_rate":            0.04,
    "negative_rate":         0.03,
    "swap_rate":             0.02,
    "truncate_rate":         0.03,
    "case_rate":             0.04,
    "whitespace_rate":       0.04,
    "encoding_rate":         0.02,
    "amount_mismatch_rate":  0.03,
    "date_seq_rate":         0.025,
    "stale_status_rate":     0.03,
}

# ============================================================
# I/O HELPERS  (folder-per-table layout)
# ============================================================
def load_table(name, base_dir):
    """Load all chunk files for a table into one DataFrame."""
    d = Path(base_dir) / name
    if not d.exists():
        print("  SKIP (not found):", name)
        return None
    frames = []
    for f in sorted(d.glob("*.csv")):
        frames.append(pd.read_csv(f, low_memory=False))
    for f in sorted(d.glob("*.parquet")):
        frames.append(pd.read_parquet(f))
    if not frames:
        print("  SKIP (empty):", name)
        return None
    df = pd.concat(frames, ignore_index=True)
    print("  Loaded", name + ":", len(df), "rows from", len(frames), "files")
    return df


def save_table(df, name, base_dir, chunk_size=50000):
    """Save DataFrame as chunked files into output folder.
    
    Injected data has mixed types (strings in numeric cols, etc.) so we
    write ALL chunks as CSV — parquet is strict about types and the whole
    point of this data is to be messy for the pipeline to clean.
    A random ~30% of chunks are written as parquet with all-string schema.
    """
    d = Path(base_dir) / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    # Stringify everything — injected data is intentionally dirty
    df = df.astype(str).replace("nan", "").replace("<NA>", "")
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    csv_n = pq_n = 0
    for i, chunk in enumerate(chunks):
        fname = name + "_part_" + str(i).zfill(3)
        if random.random() < 0.6:
            chunk.to_csv(d / (fname + ".csv"), index=False)
            csv_n += 1
        else:
            # All-string parquet — valid, PySpark will cast in silver layer
            chunk.to_parquet(d / (fname + ".parquet"), index=False)
            pq_n += 1
    print("  Saved", name + ":", len(df), "rows ->",
          len(chunks), "chunks (" + str(csv_n) + " CSV + " + str(pq_n) + " Parquet)")

# ============================================================
# ISSUE PRIMITIVES
# ============================================================
class DQ:
    """All issue-injection primitives as static methods."""

    @staticmethod
    def missing(df, cols, rate):
        """Null out values in given columns."""
        df = df.copy()
        for c in cols:
            if c not in df.columns:
                continue
            mask = np.random.random(len(df)) < rate
            df.loc[mask, c] = np.nan
        return df

    @staticmethod
    def invalid_format(df, col, fmt, rate):
        """Replace values with invalid-format strings."""
        if col not in df.columns:
            return df
        df = df.copy()
        bad = {
            "date":  ["2025-13-01", "2024-02-30", "99/99/9999", "not-a-date", "0000-00-00"],
            "phone": ["123", "555-ABC-DEFG", "000-000-0000", "(999)999-9999x", ""],
            "email": ["notanemail", "missing@", "@domain.com", "user@@x.com", "x@.c"],
            "zip":   ["123", "ABCDE", "00000", "123456789", "9999-12345"],
            "ssn":   ["123", "000-00-0000", "XXX-XX-XXXX", "123456789", "999-99-999"],
            "npi":   ["123", "ABC", "0000000000", "12345678901", "NPI-INVALID"],
            "icd":   ["ZZZ.99", "INVALID", "999.999", "A00.999", ""],
            "cpt":   ["ABCDE", "9999", "000000", "INVALID", ""],
        }
        pool = bad.get(fmt, bad["date"])
        mask = np.random.random(len(df)) < rate
        if mask.sum() > 0:
            df.loc[mask, col] = np.random.choice(pool, size=mask.sum())
        return df

    @staticmethod
    def out_of_range(df, col, lo, hi, rate):
        """Push numeric values outside [lo, hi]."""
        if col not in df.columns:
            return df
        df = df.copy()
        mask = np.random.random(len(df)) < rate
        if mask.sum() == 0:
            return df
        span = hi - lo
        outliers = np.where(
            np.random.random(mask.sum()) < 0.5,
            lo - np.random.uniform(span * 0.5, span * 2, mask.sum()),
            hi + np.random.uniform(span * 0.5, span * 2, mask.sum()),
        )
        # Assign as strings so object-dtype columns accept them cleanly
        df.loc[mask, col] = [str(round(float(v), 2)) for v in outliers]
        return df

    @staticmethod
    def future_date(df, col, rate, max_days=730):
        """Set some dates into the future."""
        if col not in df.columns:
            return df
        df = df.copy()
        mask = np.random.random(len(df)) < rate
        if mask.sum() == 0:
            return df
        future_dates = [
            (datetime.now() + timedelta(days=int(d))).strftime("%Y-%m-%d")
            for d in np.random.randint(1, max_days, mask.sum())
        ]
        df.loc[mask, col] = future_dates
        return df

    @staticmethod
    def past_date(df, col, rate, earliest_year=1920):
        """Set some dates unrealistically far in the past."""
        if col not in df.columns:
            return df
        df = df.copy()
        mask = np.random.random(len(df)) < rate
        if mask.sum() == 0:
            return df
        # Use a year well before earliest_year as the lower bound
        lo = max(1900, earliest_year - 60)
        hi = max(lo + 1, earliest_year - 1)
        old_dates = [
            datetime(
                random.randint(lo, hi),
                random.randint(1, 12),
                random.randint(1, 28),
            ).strftime("%Y-%m-%d")
            for _ in range(mask.sum())
        ]
        df.loc[mask, col] = old_dates
        return df

    @staticmethod
    def impossible(df, col, rate):
        """Inject biologically/logically impossible values."""
        if col not in df.columns:
            return df
        df = df.copy()
        pools = {
            "age":              [-5, 150, 200, 999, -1],
            "height_inches":    [10, 120, 200, -50, 0],
            "weight_lbs":       [0, -10, 5, 2000, 5000],
            "bmi":              [0, 5, 100, 200, -20],
            "systolic_bp":      [0, 30, 300, 500, -10],
            "diastolic_bp":     [0, 20, 250, 400, -5],
            "heart_rate":       [0, 10, 300, 500, -20],
            "temperature_f":    [80, 110, 120, -40, 212],
            "oxygen_saturation":[0, 10, 50, 150, 200],
            "pain_scale":       [-1, 15, 20, 100, -5],
            "duration_minutes": [-30, 0, 2000, 9999, -1],
            "units":            [-5, 0, 999, -1, 500],
            "refills":          [-1, -5, 999, -10, 500],
            "quantity":         [-10, 0, 99999, -1, 50000],
        }
        pool = pools.get(col)
        if pool is None:
            return df
        mask = np.random.random(len(df)) < rate
        if mask.sum() > 0:
            df.loc[mask, col] = np.random.choice(pool, size=mask.sum())
        return df

    @staticmethod
    def wrong_type(df, col, rate):
        """Insert wrong-type values (strings in numeric cols, etc.)."""
        if col not in df.columns:
            return df
        df = df.copy()
        mask = np.random.random(len(df)) < rate
        if mask.sum() == 0:
            return df
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(object)
            df.loc[mask, col] = np.random.choice(
                ["TEXT", "N/A", "NULL", "???", "123ABC", "#REF!", ""], size=mask.sum())
        else:
            df.loc[mask, col] = np.random.randint(100000, 999999, size=mask.sum()).astype(str)
        return df

    @staticmethod
    def special_chars(df, cols, rate):
        """Inject garbage characters into text fields."""
        df = df.copy()
        garbage = ["!@#$%", "NULL", "undefined", "N/A", "???", "***", "\x00", "\\n\\t"]
        for c in cols:
            if c not in df.columns:
                continue
            mask = np.random.random(len(df)) < rate
            idxs = np.where(mask)[0]
            for i in idxs:
                val = df.at[i, c]
                if pd.notna(val):
                    s = str(val)
                    pos = random.randint(0, max(0, len(s) - 1))
                    df.at[i, c] = s[:pos] + random.choice(garbage) + s[pos:]
        return df

    @staticmethod
    def duplicates(df, rate):
        """Add duplicate rows (exact + near-duplicate with mangled IDs)."""
        n = int(len(df) * rate)
        if n == 0:
            return df
        idx = np.random.choice(len(df), n, replace=False)
        dups = df.iloc[idx].copy()
        # Mangle ID columns so they look like re-submitted records
        for c in dups.columns:
            if c.endswith("_id") or c.endswith("_number"):
                dups[c] = dups[c].astype(str) + "_DUP"
        return pd.concat([df, dups], ignore_index=True)

    @staticmethod
    def inconsistent(df, col, mapping, rate):
        """Replace values with logically inconsistent alternatives."""
        if col not in df.columns:
            return df
        df = df.copy()
        mask = np.random.random(len(df)) < rate
        idxs = np.where(mask)[0]
        for i in idxs:
            v = df.at[i, col]
            if v in mapping:
                df.at[i, col] = random.choice(mapping[v])
        return df

    @staticmethod
    def negative(df, cols, rate):
        """Flip financial/quantity columns to negative."""
        df = df.copy()
        for c in cols:
            if c not in df.columns:
                continue
            mask = np.random.random(len(df)) < rate
            vals = pd.to_numeric(df.loc[mask, c], errors="coerce").abs() * -1
            df.loc[mask, c] = vals.astype(str)
        return df

    @staticmethod
    def swap_cols(df, c1, c2, rate):
        """Swap values between two columns."""
        if c1 not in df.columns or c2 not in df.columns:
            return df
        df = df.copy()
        mask = np.random.random(len(df)) < rate
        tmp = df.loc[mask, c1].copy()
        df.loc[mask, c1] = df.loc[mask, c2]
        df.loc[mask, c2] = tmp
        return df

    @staticmethod
    def truncate(df, cols, rate):
        """Truncate text values mid-string."""
        df = df.copy()
        for c in cols:
            if c not in df.columns:
                continue
            mask = np.random.random(len(df)) < rate
            idxs = np.where(mask)[0]
            for i in idxs:
                v = df.at[i, c]
                if pd.notna(v):
                    s = str(v)
                    if len(s) > 4:
                        df.at[i, c] = s[:random.randint(2, len(s) - 1)]
        return df

    @staticmethod
    def case_mess(df, cols, rate):
        """Randomly change casing of text values."""
        df = df.copy()
        transforms = [str.upper, str.lower, str.title,
                      lambda s: "".join(c.upper() if i % 2 == 0 else c.lower()
                                        for i, c in enumerate(s))]
        for c in cols:
            if c not in df.columns:
                continue
            mask = np.random.random(len(df)) < rate
            idxs = np.where(mask)[0]
            for i in idxs:
                v = df.at[i, c]
                if pd.notna(v):
                    df.at[i, c] = random.choice(transforms)(str(v))
        return df

    @staticmethod
    def whitespace(df, cols, rate):
        """Add leading/trailing/embedded extra spaces."""
        df = df.copy()
        for c in cols:
            if c not in df.columns:
                continue
            mask = np.random.random(len(df)) < rate
            idxs = np.where(mask)[0]
            for i in idxs:
                v = df.at[i, c]
                if pd.notna(v):
                    s = str(v)
                    choice = random.randint(0, 3)
                    if choice == 0:
                        df.at[i, c] = "  " + s
                    elif choice == 1:
                        df.at[i, c] = s + "   "
                    elif choice == 2:
                        df.at[i, c] = "  " + s + "  "
                    else:
                        mid = len(s) // 2
                        df.at[i, c] = s[:mid] + "  " + s[mid:]
        return df

    @staticmethod
    def encoding_corrupt(df, cols, rate):
        """Replace ASCII chars with lookalike unicode (simulates encoding bugs)."""
        df = df.copy()
        subs = {"a": "a", "e": "e", "i": "i", "o": "o", "u": "u",
                "A": "A", "E": "E", "I": "I", "O": "O", "U": "U",
                "c": "c", "n": "n", "s": "s"}
        for c in cols:
            if c not in df.columns:
                continue
            mask = np.random.random(len(df)) < rate
            idxs = np.where(mask)[0]
            for i in idxs:
                v = df.at[i, c]
                if pd.notna(v):
                    s = str(v)
                    out = []
                    for ch in s:
                        if ch in subs and random.random() < 0.3:
                            out.append(subs[ch])
                        else:
                            out.append(ch)
                    df.at[i, c] = "".join(out)
        return df

    @staticmethod
    def orphan_fk(df, fk_col, rate):
        """Replace foreign key values with non-existent IDs."""
        if fk_col not in df.columns:
            return df
        df = df.copy()
        mask = np.random.random(len(df)) < rate
        if mask.sum() > 0:
            df.loc[mask, fk_col] = [
                "INVALID_" + uuid.uuid4().hex[:8].upper()
                for _ in range(mask.sum())
            ]
        return df

    @staticmethod
    def date_sequence(df, start_col, end_col, rate):
        """Make end_date earlier than start_date."""
        if start_col not in df.columns or end_col not in df.columns:
            return df
        df = df.copy()
        mask = np.random.random(len(df)) < rate
        idxs = np.where(mask)[0]
        for i in idxs:
            s = df.at[i, start_col]
            if pd.notna(s):
                try:
                    sd = pd.to_datetime(s)
                    df.at[i, end_col] = (sd - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d")
                except Exception:
                    pass
        return df

    @staticmethod
    def stale_status(df, status_col, date_col, active_statuses, rate):
        """Set active status on records with old dates (stale/zombie records)."""
        if status_col not in df.columns or date_col not in df.columns:
            return df
        df = df.copy()
        mask = np.random.random(len(df)) < rate
        idxs = np.where(mask)[0]
        for i in idxs:
            df.at[i, status_col] = random.choice(active_statuses)
            df.at[i, date_col] = (datetime.now() - timedelta(days=random.randint(365*3, 365*10))).strftime("%Y-%m-%d")
        return df

    @staticmethod
    def amount_mismatch(df, line_col, total_col, rate):
        """Make a line-level amount not match the header total."""
        if line_col not in df.columns or total_col not in df.columns:
            return df
        df = df.copy()
        mask = np.random.random(len(df)) < rate
        idxs = np.where(mask)[0]
        for i in idxs:
            v = pd.to_numeric(df.at[i, total_col], errors="coerce")
            if pd.notna(v):
                df.at[i, total_col] = str(round(float(v) * random.uniform(1.1, 3.0), 2))
        return df

# ============================================================
# INJECTOR  (per-table rules)
# ============================================================
class Injector:
    def __init__(self, cfg):
        self.cfg = cfg
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])

    def _r(self, key):
        return self.cfg[key]

    # ----------------------------------------------------------
    def inject_clinics(self, df):
        r = self._r
        df = DQ.missing(df, ["fax", "bed_count", "accreditation"], r("missing_rate"))
        df = DQ.invalid_format(df, "npi", "npi", r("format_rate"))
        df = DQ.invalid_format(df, "phone", "phone", r("format_rate"))
        df = DQ.special_chars(df, ["clinic_name", "address"], r("special_rate"))
        df = DQ.whitespace(df, ["clinic_name", "city", "state"], r("whitespace_rate"))
        df = DQ.case_mess(df, ["clinic_type", "network_status"], r("case_rate"))
        df = DQ.encoding_corrupt(df, ["clinic_name"], r("encoding_rate"))
        df = DQ.duplicates(df, r("duplicate_rate"))
        return df

    def inject_providers(self, df):
        r = self._r
        df = DQ.missing(df, ["dea_number", "email", "sub_specialty", "state_license"], r("missing_rate"))
        df = DQ.invalid_format(df, "npi", "npi", r("format_rate"))
        df = DQ.invalid_format(df, "phone", "phone", r("format_rate"))
        df = DQ.invalid_format(df, "email", "email", r("format_rate"))
        df = DQ.out_of_range(df, "years_experience", 0, 60, r("range_rate"))
        df = DQ.impossible(df, "years_experience", r("range_rate") * 0.5)
        df = DQ.special_chars(df, ["first_name", "last_name"], r("special_rate"))
        df = DQ.whitespace(df, ["first_name", "last_name", "specialty"], r("whitespace_rate"))
        df = DQ.case_mess(df, ["specialty", "credential", "status"], r("case_rate"))
        df = DQ.inconsistent(df, "status",
            {"Active": ["Inactive", "Suspended"], "Inactive": ["Active"], "Suspended": ["Active"]},
            r("logic_rate"))
        df = DQ.wrong_type(df, "years_experience", r("type_rate"))
        df = DQ.duplicates(df, r("duplicate_rate"))
        return df

    def inject_payers(self, df):
        r = self._r
        df = DQ.missing(df, ["contact_number", "claims_address"], r("missing_rate"))
        df = DQ.invalid_format(df, "contact_number", "phone", r("format_rate"))
        df = DQ.whitespace(df, ["payer_name", "payer_type"], r("whitespace_rate"))
        df = DQ.case_mess(df, ["payer_type", "payer_name"], r("case_rate"))
        df = DQ.encoding_corrupt(df, ["payer_name"], r("encoding_rate"))
        df = DQ.duplicates(df, r("duplicate_rate") * 0.5)
        return df

    def inject_patients(self, df):
        r = self._r
        df = DQ.missing(df, ["phone", "email", "ssn_last4", "address", "zip_code"], r("missing_rate"))
        df = DQ.invalid_format(df, "phone", "phone", r("format_rate"))
        df = DQ.invalid_format(df, "email", "email", r("format_rate"))
        df = DQ.invalid_format(df, "zip_code", "zip", r("format_rate"))
        df = DQ.invalid_format(df, "ssn_last4", "ssn", r("format_rate"))
        df = DQ.out_of_range(df, "age", 0, 120, r("range_rate"))
        df = DQ.impossible(df, "age", r("range_rate"))
        df = DQ.future_date(df, "dob", r("date_rate"))
        df = DQ.past_date(df, "dob", r("date_rate"), earliest_year=1900)
        df = DQ.special_chars(df, ["first_name", "last_name", "address"], r("special_rate"))
        df = DQ.whitespace(df, ["first_name", "last_name", "city", "state"], r("whitespace_rate"))
        df = DQ.case_mess(df, ["gender", "race", "ethnicity", "smoking_status", "bmi_category"], r("case_rate"))
        df = DQ.encoding_corrupt(df, ["first_name", "last_name", "address"], r("encoding_rate"))
        df = DQ.inconsistent(df, "gender",
            {"Male": ["Female", "Non-Binary"], "Female": ["Male", "Non-Binary"], "Non-Binary": ["Male"]},
            r("logic_rate"))
        df = DQ.inconsistent(df, "smoking_status",
            {"Never": ["Current"], "Current": ["Never"], "Former": ["Current"]},
            r("logic_rate") * 0.5)
        df = DQ.wrong_type(df, "age", r("type_rate"))
        df = DQ.wrong_type(df, "mrn", r("type_rate"))
        df = DQ.duplicates(df, r("duplicate_rate"))
        return df

    def inject_patient_insurance(self, df):
        r = self._r
        df = DQ.missing(df, ["group_number", "deductible_met", "out_of_pocket_met"], r("missing_rate"))
        df = DQ.out_of_range(df, "copay_office", 0, 500, r("range_rate"))
        df = DQ.out_of_range(df, "deductible_individual", 0, 20000, r("range_rate"))
        df = DQ.negative(df, ["copay_office", "deductible_individual", "deductible_met",
                               "out_of_pocket_max", "out_of_pocket_met"], r("negative_rate"))
        df = DQ.future_date(df, "effective_date", r("date_rate"))
        df = DQ.date_sequence(df, "effective_date", "termination_date", r("date_seq_rate"))
        df = DQ.orphan_fk(df, "patient_id", r("orphan_rate"))
        df = DQ.orphan_fk(df, "payer_id", r("orphan_rate"))
        df = DQ.case_mess(df, ["plan_type", "insurance_priority"], r("case_rate"))
        df = DQ.whitespace(df, ["plan_name", "member_id"], r("whitespace_rate"))
        df = DQ.amount_mismatch(df, "deductible_met", "deductible_individual", r("amount_mismatch_rate"))
        df = DQ.duplicates(df, r("duplicate_rate"))
        return df

    def inject_appointments(self, df):
        r = self._r
        df = DQ.missing(df, ["visit_reason", "cancelled_reason", "referral_id"], r("missing_rate"))
        df = DQ.future_date(df, "appointment_date", r("date_rate"))
        df = DQ.past_date(df, "appointment_date", r("date_rate"), earliest_year=2015)
        df = DQ.impossible(df, "duration_minutes", r("range_rate"))
        df = DQ.out_of_range(df, "duration_minutes", 5, 480, r("range_rate"))
        df = DQ.inconsistent(df, "appointment_status",
            {"Completed": ["Cancelled", "No Show"], "Cancelled": ["Completed"],
             "No Show": ["Completed", "Scheduled"]},
            r("logic_rate"))
        df = DQ.special_chars(df, ["visit_reason"], r("special_rate"))
        df = DQ.whitespace(df, ["appointment_type", "appointment_status"], r("whitespace_rate"))
        df = DQ.case_mess(df, ["appointment_type", "appointment_status"], r("case_rate"))
        df = DQ.orphan_fk(df, "patient_id", r("orphan_rate"))
        df = DQ.orphan_fk(df, "provider_id", r("orphan_rate"))
        df = DQ.stale_status(df, "appointment_status", "appointment_date",
                             ["Scheduled", "Rescheduled"], r("stale_status_rate"))
        df = DQ.duplicates(df, r("duplicate_rate"))
        return df

    def inject_encounters(self, df):
        r = self._r
        df = DQ.missing(df, ["chief_complaint", "assessment", "plan", "hpi"], r("missing_rate"))
        df = DQ.future_date(df, "encounter_date", r("date_rate"))
        df = DQ.past_date(df, "encounter_date", r("date_rate"), earliest_year=2015)
        df = DQ.negative(df, ["drg_base_rate", "los_days"], r("negative_rate"))
        df = DQ.out_of_range(df, "los_days", 0, 365, r("range_rate"))
        df = DQ.date_sequence(df, "admit_date", "discharge_date", r("date_seq_rate"))
        df = DQ.special_chars(df, ["chief_complaint", "assessment", "plan"], r("special_rate"))
        df = DQ.truncate(df, ["chief_complaint", "hpi", "assessment", "plan"], r("truncate_rate"))
        df = DQ.whitespace(df, ["encounter_type", "place_of_service_code"], r("whitespace_rate"))
        df = DQ.case_mess(df, ["encounter_type", "discharge_disposition"], r("case_rate"))
        df = DQ.orphan_fk(df, "patient_id", r("orphan_rate"))
        df = DQ.orphan_fk(df, "appointment_id", r("orphan_rate"))
        df = DQ.encoding_corrupt(df, ["chief_complaint", "assessment"], r("encoding_rate"))
        df = DQ.duplicates(df, r("duplicate_rate"))
        return df

    def inject_vitals(self, df):
        r = self._r
        df = DQ.missing(df, ["recorded_by"], r("missing_rate"))
        df = DQ.impossible(df, "systolic_bp", r("range_rate"))
        df = DQ.impossible(df, "diastolic_bp", r("range_rate"))
        df = DQ.impossible(df, "heart_rate", r("range_rate"))
        df = DQ.impossible(df, "temperature_f", r("range_rate"))
        df = DQ.impossible(df, "oxygen_saturation", r("range_rate"))
        df = DQ.impossible(df, "pain_scale", r("range_rate"))
        df = DQ.impossible(df, "height_inches", r("range_rate"))
        df = DQ.impossible(df, "weight_lbs", r("range_rate"))
        df = DQ.out_of_range(df, "bmi", 10, 80, r("range_rate"))
        # Swap systolic/diastolic (common data entry error)
        df = DQ.swap_cols(df, "systolic_bp", "diastolic_bp", r("swap_rate"))
        # Swap height/weight
        df = DQ.swap_cols(df, "height_inches", "weight_lbs", r("swap_rate") * 0.5)
        df = DQ.wrong_type(df, "systolic_bp", r("type_rate"))
        df = DQ.wrong_type(df, "temperature_f", r("type_rate"))
        df = DQ.future_date(df, "recorded_at", r("date_rate"))
        df = DQ.orphan_fk(df, "encounter_id", r("orphan_rate"))
        df = DQ.duplicates(df, r("duplicate_rate"))
        return df

    def inject_diagnoses(self, df):
        r = self._r
        df = DQ.missing(df, ["poa_indicator", "hcc_code"], r("missing_rate"))
        df = DQ.invalid_format(df, "icd10_code", "icd", r("format_rate"))
        df = DQ.inconsistent(df, "diagnosis_type",
            {"Primary": ["Secondary", "Admitting"], "Secondary": ["Primary"],
             "Admitting": ["Primary", "Secondary"]},
            r("logic_rate"))
        df = DQ.special_chars(df, ["diagnosis_description"], r("special_rate"))
        df = DQ.truncate(df, ["diagnosis_description"], r("truncate_rate"))
        df = DQ.whitespace(df, ["icd10_code", "diagnosis_type", "poa_indicator"], r("whitespace_rate"))
        df = DQ.case_mess(df, ["diagnosis_type", "poa_indicator"], r("case_rate"))
        df = DQ.orphan_fk(df, "encounter_id", r("orphan_rate"))
        df = DQ.orphan_fk(df, "patient_id", r("orphan_rate"))
        df = DQ.duplicates(df, r("duplicate_rate"))
        return df

    def inject_procedures(self, df):
        r = self._r
        df = DQ.missing(df, ["modifier", "revenue_code", "ndc_code"], r("missing_rate"))
        df = DQ.invalid_format(df, "cpt_code", "cpt", r("format_rate"))
        df = DQ.negative(df, ["procedure_charge"], r("negative_rate"))
        df = DQ.out_of_range(df, "procedure_charge", 0, 100000, r("range_rate"))
        df = DQ.impossible(df, "units", r("range_rate"))
        df = DQ.out_of_range(df, "units", 1, 99, r("range_rate"))
        df = DQ.swap_cols(df, "cpt_code", "modifier", r("swap_rate"))
        df = DQ.truncate(df, ["procedure_description"], r("truncate_rate"))
        df = DQ.whitespace(df, ["cpt_code", "modifier", "revenue_code"], r("whitespace_rate"))
        df = DQ.case_mess(df, ["procedure_description"], r("case_rate"))
        df = DQ.wrong_type(df, "procedure_charge", r("type_rate"))
        df = DQ.wrong_type(df, "units", r("type_rate"))
        df = DQ.orphan_fk(df, "encounter_id", r("orphan_rate"))
        df = DQ.duplicates(df, r("duplicate_rate"))
        return df

    def inject_lab_results(self, df):
        r = self._r
        df = DQ.missing(df, ["result_value", "abnormal_flag", "loinc_code"], r("missing_rate"))
        df = DQ.out_of_range(df, "result_value", -999, 99999, r("range_rate"))
        df = DQ.impossible(df, "result_value", r("range_rate") * 0.5)
        df = DQ.future_date(df, "result_date", r("date_rate"))
        df = DQ.future_date(df, "order_date", r("date_rate"))
        df = DQ.date_sequence(df, "order_date", "result_date", r("date_seq_rate"))
        df = DQ.special_chars(df, ["test_name", "performing_lab"], r("special_rate"))
        df = DQ.whitespace(df, ["test_name", "lab_name", "abnormal_flag", "status"], r("whitespace_rate"))
        df = DQ.case_mess(df, ["abnormal_flag", "status", "lab_name"], r("case_rate"))
        df = DQ.inconsistent(df, "abnormal_flag",
            {"Normal": ["High", "Low", "Critical High"], "High": ["Normal", "Low"],
             "Low": ["Normal", "High"], "Critical High": ["Normal"]},
            r("logic_rate"))
        df = DQ.wrong_type(df, "result_value", r("type_rate"))
        df = DQ.orphan_fk(df, "encounter_id", r("orphan_rate"))
        df = DQ.orphan_fk(df, "patient_id", r("orphan_rate"))
        df = DQ.duplicates(df, r("duplicate_rate"))
        return df

    def inject_medications(self, df):
        r = self._r
        df = DQ.missing(df, ["pharmacy_name", "rx_number", "prior_auth_number"], r("missing_rate"))
        df = DQ.negative(df, ["quantity", "refills", "days_supply"], r("negative_rate"))
        df = DQ.impossible(df, "quantity", r("range_rate"))
        df = DQ.impossible(df, "refills", r("range_rate"))
        df = DQ.future_date(df, "fill_date", r("date_rate"))
        df = DQ.past_date(df, "prescribed_date", r("date_rate"), earliest_year=2010)
        df = DQ.date_sequence(df, "prescribed_date", "fill_date", r("date_seq_rate"))
        df = DQ.truncate(df, ["drug_name", "generic_name"], r("truncate_rate"))
        df = DQ.whitespace(df, ["drug_name", "generic_name", "drug_class", "route", "frequency"], r("whitespace_rate"))
        df = DQ.case_mess(df, ["drug_class", "route", "frequency", "status"], r("case_rate"))
        df = DQ.encoding_corrupt(df, ["drug_name", "generic_name"], r("encoding_rate"))
        df = DQ.inconsistent(df, "status",
            {"Active": ["Discontinued"], "Discontinued": ["Active"], "On Hold": ["Active"]},
            r("logic_rate"))
        df = DQ.orphan_fk(df, "encounter_id", r("orphan_rate"))
        df = DQ.orphan_fk(df, "patient_id", r("orphan_rate"))
        df = DQ.duplicates(df, r("duplicate_rate"))
        return df

    def inject_prior_authorizations(self, df):
        r = self._r
        df = DQ.missing(df, ["auth_number", "approved_units", "denial_reason"], r("missing_rate"))
        df = DQ.future_date(df, "request_date", r("date_rate"))
        df = DQ.date_sequence(df, "request_date", "decision_date", r("date_seq_rate"))
        df = DQ.date_sequence(df, "approved_from_date", "approved_to_date", r("date_seq_rate"))
        df = DQ.invalid_format(df, "cpt_code", "cpt", r("format_rate"))
        df = DQ.invalid_format(df, "icd10_code", "icd", r("format_rate"))
        df = DQ.inconsistent(df, "status",
            {"Approved": ["Denied", "Expired"], "Denied": ["Approved"], "Pending": ["Expired"]},
            r("logic_rate"))
        df = DQ.whitespace(df, ["service_type", "status", "urgency"], r("whitespace_rate"))
        df = DQ.case_mess(df, ["service_type", "status", "urgency"], r("case_rate"))
        df = DQ.orphan_fk(df, "patient_id", r("orphan_rate"))
        df = DQ.orphan_fk(df, "provider_id", r("orphan_rate"))
        df = DQ.duplicates(df, r("duplicate_rate"))
        return df

    def inject_claims(self, df):
        r = self._r
        df = DQ.missing(df, ["prior_auth_number", "secondary_payer_id",
                              "referring_provider_npi"], r("missing_rate"))
        df = DQ.negative(df, ["total_charge"], r("negative_rate"))
        df = DQ.out_of_range(df, "total_charge", 0, 500000, r("range_rate"))
        df = DQ.future_date(df, "submission_date", r("date_rate"))
        df = DQ.future_date(df, "adjudication_date", r("date_rate"))
        df = DQ.date_sequence(df, "submission_date", "adjudication_date", r("date_seq_rate"))
        df = DQ.invalid_format(df, "rendering_provider_npi", "npi", r("format_rate"))
        df = DQ.invalid_format(df, "billing_provider_npi", "npi", r("format_rate"))
        df = DQ.inconsistent(df, "claim_status",
            {"Paid": ["Denied", "Rejected"], "Denied": ["Paid"],
             "Void": ["Paid", "Pending"], "Rejected": ["Paid"]},
            r("logic_rate"))
        df = DQ.whitespace(df, ["claim_status", "claim_type", "filing_indicator"], r("whitespace_rate"))
        df = DQ.case_mess(df, ["claim_status", "claim_type", "filing_indicator"], r("case_rate"))
        df = DQ.wrong_type(df, "total_charge", r("type_rate"))
        df = DQ.orphan_fk(df, "patient_id", r("orphan_rate"))
        df = DQ.orphan_fk(df, "provider_id", r("orphan_rate"))
        df = DQ.stale_status(df, "claim_status", "submission_date",
                             ["Pending"], r("stale_status_rate"))
        df = DQ.duplicates(df, r("duplicate_rate"))
        return df

    def inject_claim_lines(self, df):
        r = self._r
        df = DQ.missing(df, ["modifier", "revenue_code", "remark_code"], r("missing_rate"))
        df = DQ.invalid_format(df, "cpt_code", "cpt", r("format_rate"))
        df = DQ.negative(df, ["billed_amount", "allowed_amount", "paid_amount",
                               "patient_responsibility"], r("negative_rate"))
        df = DQ.out_of_range(df, "billed_amount", 0, 100000, r("range_rate"))
        df = DQ.amount_mismatch(df, "paid_amount", "billed_amount", r("amount_mismatch_rate"))
        df = DQ.swap_cols(df, "billed_amount", "allowed_amount", r("swap_rate"))
        df = DQ.wrong_type(df, "billed_amount", r("type_rate"))
        df = DQ.wrong_type(df, "units", r("type_rate"))
        df = DQ.whitespace(df, ["cpt_code", "modifier", "line_status"], r("whitespace_rate"))
        df = DQ.case_mess(df, ["line_status", "adjustment_reason_code"], r("case_rate"))
        df = DQ.orphan_fk(df, "claim_id", r("orphan_rate"))
        df = DQ.duplicates(df, r("duplicate_rate"))
        return df

    def inject_remittances(self, df):
        r = self._r
        df = DQ.missing(df, ["check_number", "eft_trace_number"], r("missing_rate"))
        df = DQ.negative(df, ["total_payment_amount"], r("negative_rate"))
        df = DQ.future_date(df, "payment_date", r("date_rate"))
        df = DQ.out_of_range(df, "claim_count", 1, 10000, r("range_rate"))
        df = DQ.whitespace(df, ["payment_method", "payer_name"], r("whitespace_rate"))
        df = DQ.case_mess(df, ["payment_method"], r("case_rate"))
        df = DQ.wrong_type(df, "total_payment_amount", r("type_rate"))
        df = DQ.orphan_fk(df, "payer_id", r("orphan_rate"))
        df = DQ.duplicates(df, r("duplicate_rate"))
        return df

    def inject_payments(self, df):
        r = self._r
        df = DQ.missing(df, ["check_number", "eft_trace_number", "remittance_id"], r("missing_rate"))
        df = DQ.negative(df, ["paid_amount"], r("negative_rate"))
        df = DQ.out_of_range(df, "paid_amount", 0, 500000, r("range_rate"))
        df = DQ.future_date(df, "payment_date", r("date_rate"))
        df = DQ.future_date(df, "posted_date", r("date_rate"))
        df = DQ.date_sequence(df, "payment_date", "posted_date", r("date_seq_rate"))
        df = DQ.inconsistent(df, "payment_source",
            {"ERA": ["Check", "Patient"], "Check": ["ERA", "Wire"],
             "Patient": ["ERA"], "Wire": ["Check"]},
            r("logic_rate"))
        df = DQ.whitespace(df, ["payment_source", "patient_payment_type"], r("whitespace_rate"))
        df = DQ.case_mess(df, ["payment_source", "patient_payment_type"], r("case_rate"))
        df = DQ.wrong_type(df, "paid_amount", r("type_rate"))
        df = DQ.orphan_fk(df, "claim_id", r("orphan_rate"))
        df = DQ.duplicates(df, r("duplicate_rate"))
        return df

    def inject_denials(self, df):
        r = self._r
        df = DQ.missing(df, ["root_cause", "corrected_claim_submitted"], r("missing_rate"))
        df = DQ.future_date(df, "denial_date", r("date_rate"))
        df = DQ.date_sequence(df, "denial_date", "appeal_deadline", r("date_seq_rate"))
        df = DQ.inconsistent(df, "appeal_status",
            {"Overturned": ["Upheld"], "Upheld": ["Overturned"],
             "Not Appealed": ["Appealed", "Overturned"]},
            r("logic_rate"))
        df = DQ.whitespace(df, ["denial_code", "denial_category", "appeal_status"], r("whitespace_rate"))
        df = DQ.case_mess(df, ["denial_category", "appeal_status", "root_cause"], r("case_rate"))
        df = DQ.truncate(df, ["denial_reason", "root_cause"], r("truncate_rate"))
        df = DQ.orphan_fk(df, "claim_id", r("orphan_rate"))
        df = DQ.duplicates(df, r("duplicate_rate"))
        return df

    def inject_appeals(self, df):
        r = self._r
        df = DQ.missing(df, ["decision_date", "amount_recovered"], r("missing_rate"))
        df = DQ.future_date(df, "submitted_date", r("date_rate"))
        df = DQ.date_sequence(df, "submitted_date", "decision_date", r("date_seq_rate"))
        df = DQ.negative(df, ["amount_recovered"], r("negative_rate"))
        df = DQ.inconsistent(df, "decision",
            {"Overturned": ["Upheld"], "Upheld": ["Overturned"],
             "Pending": ["Overturned", "Upheld"]},
            r("logic_rate"))
        df = DQ.whitespace(df, ["appeal_type", "decision", "appeal_reason"], r("whitespace_rate"))
        df = DQ.case_mess(df, ["appeal_type", "decision"], r("case_rate"))
        df = DQ.truncate(df, ["appeal_reason"], r("truncate_rate"))
        df = DQ.orphan_fk(df, "claim_id", r("orphan_rate"))
        df = DQ.orphan_fk(df, "denial_id", r("orphan_rate"))
        df = DQ.duplicates(df, r("duplicate_rate"))
        return df

    def inject_eligibility_checks(self, df):
        r = self._r
        df = DQ.missing(df, ["deductible_remaining", "out_of_pocket_remaining"], r("missing_rate"))
        df = DQ.future_date(df, "check_date", r("date_rate"))
        df = DQ.negative(df, ["copay_verified", "deductible_verified",
                               "deductible_remaining", "out_of_pocket_remaining"], r("negative_rate"))
        df = DQ.out_of_range(df, "deductible_remaining", 0, 20000, r("range_rate"))
        df = DQ.inconsistent(df, "response_status",
            {"Active": ["Inactive", "Not Found"], "Inactive": ["Active"],
             "Not Found": ["Active"]},
            r("logic_rate"))
        df = DQ.whitespace(df, ["check_type", "response_status", "plan_name"], r("whitespace_rate"))
        df = DQ.case_mess(df, ["check_type", "response_status"], r("case_rate"))
        df = DQ.wrong_type(df, "deductible_verified", r("type_rate"))
        df = DQ.orphan_fk(df, "patient_id", r("orphan_rate"))
        df = DQ.orphan_fk(df, "payer_id", r("orphan_rate"))
        df = DQ.duplicates(df, r("duplicate_rate"))
        return df

    # ----------------------------------------------------------
    def run(self, table_name, df):
        """Dispatch to the right inject_* method."""
        method = getattr(self, "inject_" + table_name, None)
        if method is None:
            print("  No rules for", table_name, "- applying generic issues")
            df = DQ.missing(df, list(df.columns[:3]), self._r("missing_rate"))
            df = DQ.duplicates(df, self._r("duplicate_rate"))
            return df
        return method(df)

# ============================================================
# QUALITY REPORT
# ============================================================
def build_report(name, orig_df, issue_df):
    """Compare original vs injected DataFrame and return a summary dict."""
    report = {"table": name, "original_rows": len(orig_df), "issue_rows": len(issue_df)}
    report["rows_added_duplicates"] = len(issue_df) - len(orig_df)

    col_stats = {}
    for c in orig_df.columns:
        if c not in issue_df.columns:
            continue
        orig_null = int(orig_df[c].isna().sum())
        new_null  = int(issue_df[c].isna().sum())
        changed   = int((orig_df[c].astype(str) != issue_df[c].astype(str)).sum()
                        if len(orig_df) == len(issue_df) else 0)
        if new_null > orig_null or changed > 0:
            col_stats[c] = {
                "original_nulls": orig_null,
                "injected_nulls": new_null,
                "values_changed": changed,
            }
    report["column_issues"] = col_stats
    report["columns_affected"] = len(col_stats)
    return report


# ============================================================
# MAIN
# ============================================================
TABLES = [
    "clinics", "providers", "payers", "patients", "patient_insurance",
    "appointments", "encounters", "vitals", "diagnoses", "procedures",
    "lab_results", "medications", "prior_authorizations", "claims",
    "claim_lines", "remittances", "payments", "denials", "appeals",
    "eligibility_checks",
]


def main():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    in_dir  = CONFIG["input_dir"]
    out_dir = CONFIG["output_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    injector = Injector(CONFIG)

    print("=" * 70)
    print("  HEALTHCARE DATA QUALITY INJECTOR")
    print("=" * 70)
    print("  Input :", in_dir)
    print("  Output:", out_dir)
    print("  Issue types: 22 (missing, format, range, date, duplicate,")
    print("               type, special_chars, orphan_fk, logic, negative,")
    print("               swap, truncate, case, whitespace, encoding,")
    print("               amount_mismatch, date_sequence, stale_status, ...)")
    print("=" * 70)

    all_reports = []
    t0 = datetime.now()

    for table in TABLES:
        print("\n[" + table + "]")
        df = load_table(table, in_dir)
        if df is None:
            print("  -> skipped")
            continue

        orig_df = df.copy()
        # Convert to all-object dtype before injection so numeric methods
        # can freely assign strings, floats, negatives without dtype conflicts.
        df = df.astype(object)
        df_issues = injector.run(table, df)

        save_table(df_issues, table, out_dir)

        rpt = build_report(table, orig_df, df_issues)
        all_reports.append(rpt)
        print("  Issues injected: columns affected =", rpt["columns_affected"],
              "| duplicate rows added =", rpt["rows_added_duplicates"])

    # Save JSON report
    report_path = Path(out_dir) / "_injection_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "config": CONFIG,
            "tables": all_reports,
        }, f, indent=2, default=str)

    elapsed = (datetime.now() - t0).seconds
    print("\n" + "=" * 70)
    print("  INJECTION COMPLETE")
    print("=" * 70)
    total_orig  = sum(r["original_rows"] for r in all_reports)
    total_issue = sum(r["issue_rows"]    for r in all_reports)
    print("  Tables processed :", len(all_reports))
    print("  Total rows (orig):", total_orig)
    print("  Total rows (with issues):", total_issue)
    print("  Duplicate rows added:", total_issue - total_orig)
    print("  Report saved to  :", report_path)
    print("  Time elapsed     :", elapsed, "s")
    print("=" * 70)
    print()
    print("  Issue type summary per table:")
    for r in all_reports:
        print("   ", r["table"] + ":", r["original_rows"], "->",
              r["issue_rows"], "rows |", r["columns_affected"], "cols affected")
    print("=" * 70)


if __name__ == "__main__":
    main()
