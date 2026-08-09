"""Generate deterministic synthetic patient documents.

Run from the CarePilot repository root:
    python -m src.scripts.generate_synthetic_patients
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.utils import from_project_path

OUTPUT_ROOT = from_project_path("data/synthetic_patients")


@dataclass(frozen=True)
class PatientProfile:
    patient_id: str
    username: str
    display_name: str
    age: int
    sex: str
    gender: str
    home_city: str
    diagnosis: str
    stage: str
    treatment_plan: str
    oncologist: str
    insurer: str
    insurance_status: str
    referral_status: str
    referral_target: str
    key_lab_fact: str
    key_admin_fact: str
    medication_fact: str


PATIENTS = [
    PatientProfile("patient_1", "patient_1", "Synthetic Patient 1", 54, "female", "woman", "Haifa", "breast cancer", "stage II", "lumpectomy followed by paclitaxel and trastuzumab", "Dr. Leora Stein", "Clalit Shield", "approved", "submitted", "cardio-oncology", "absolute neutrophil count fell to 1.1 K/uL on 2026-02-17", "trastuzumab pre-approval is approved through 2026-08-31", "trastuzumab infusion every 21 days"),
    PatientProfile("patient_2", "patient_2", "Synthetic Patient 2", 67, "male", "man", "Tel Aviv", "colorectal cancer", "stage III", "FOLFOX chemotherapy after colectomy", "Dr. Amir Ravid", "Maccabi Care", "pending", "submitted", "nutrition clinic", "CEA decreased from 18.4 to 9.2 ng/mL", "oxaliplatin pre-approval is pending nurse review", "capecitabine tablets on days 1 through 14"),
    PatientProfile("patient_3", "patient_3", "Synthetic Patient 3", 61, "female", "woman", "Beer Sheva", "non-small cell lung cancer", "stage IV", "pembrolizumab with carboplatin and pemetrexed", "Dr. Noga Halevi", "Meuhedet Plus", "expired", "pending", "pulmonary rehabilitation", "TSH rose to 7.8 mIU/L during immunotherapy monitoring", "pembrolizumab authorization expired on 2026-03-15", "pembrolizumab infusion every 3 weeks"),
    PatientProfile("patient_4", "patient_4", "Synthetic Patient 4", 45, "female", "woman", "Jerusalem", "Hodgkin lymphoma", "stage IIB", "ABVD chemotherapy", "Dr. Eyal Shamir", "Leumit Gold", "approved", "missing", "fertility preservation", "hemoglobin improved to 11.2 g/dL", "fertility referral is missing from the chart", "doxorubicin and dacarbazine on ABVD cycle days"),
    PatientProfile("patient_5", "patient_5", "Synthetic Patient 5", 72, "male", "man", "Netanya", "prostate cancer", "stage IV", "androgen deprivation therapy plus docetaxel", "Dr. Yael Mor", "Clalit Shield", "approved", "submitted", "urology", "PSA dropped from 64 to 21 ng/mL", "docetaxel pre-approval is approved for six cycles", "leuprolide depot every 12 weeks"),
    PatientProfile("patient_6", "patient_6", "Synthetic Patient 6", 39, "female", "woman", "Ashdod", "ovarian cancer", "stage IIIC", "carboplatin and paclitaxel with planned niraparib maintenance", "Dr. Dana Keshet", "Maccabi Care", "pending", "pending", "genetic counseling", "CA-125 decreased from 410 to 188 U/mL", "BRCA testing referral is pending scheduling", "ondansetron before each chemotherapy cycle"),
    PatientProfile("patient_7", "patient_7", "Synthetic Patient 7", 58, "male", "man", "Nazareth", "pancreatic cancer", "stage II", "modified FOLFIRINOX before surgery review", "Dr. Ronen Gal", "Meuhedet Plus", "denied", "submitted", "hepatobiliary surgery", "bilirubin improved to 1.4 mg/dL after stent placement", "pegfilgrastim support was denied pending appeal", "irinotecan and fluorouracil by infusion pump"),
    PatientProfile("patient_8", "patient_8", "Synthetic Patient 8", 63, "female", "woman", "Acre", "diffuse large B-cell lymphoma", "stage III", "R-CHOP chemotherapy", "Dr. Mira Oren", "Leumit Gold", "approved", "submitted", "infectious disease", "LDH decreased from 620 to 390 U/L", "rituximab pre-approval is approved through cycle 6", "prednisone 100 mg on days 1 through 5"),
    PatientProfile("patient_9", "patient_9", "Synthetic Patient 9", 50, "male", "man", "Eilat", "melanoma", "stage III", "nivolumab adjuvant immunotherapy", "Dr. Shai Lior", "Clalit Shield", "pending", "missing", "dermatology surveillance", "ALT increased to 86 U/L during nivolumab monitoring", "dermatology surveillance referral is missing", "nivolumab infusion every 4 weeks"),
    PatientProfile("patient_10", "patient_10", "Synthetic Patient 10", 70, "female", "woman", "Tiberias", "multiple myeloma", "R-ISS stage II", "bortezomib, lenalidomide, and dexamethasone", "Dr. Tamar Sela", "Maccabi Care", "expired", "submitted", "nephrology", "M-spike decreased from 2.9 to 1.7 g/dL", "lenalidomide authorization expired on 2026-04-01", "lenalidomide 15 mg nightly on days 1 through 21"),
]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _header(profile: PatientProfile, title: str, category: str) -> str:
    return f"""Document: {title}
Category: {category}
Patient ID: {profile.patient_id}
Patient name: {profile.display_name}
Synthetic data notice: This file is fictional test data only. It contains no real patient information and no SSN.
"""


def _lab_docs(profile: PatientProfile) -> dict[str, str]:
    return {
        "labs_2026_02_17.txt": f"""{_header(profile, "Laboratory Results 2026-02-17", "lab/results")}
Diagnosis context: {profile.diagnosis}, {profile.stage}.
CBC: WBC 2.4 K/uL, ANC 1.1 K/uL, hemoglobin 10.8 g/dL, platelets 142 K/uL.
Chemistry: creatinine 0.9 mg/dL, AST 28 U/L, ALT 31 U/L.
Plain annotation: {profile.key_lab_fact}. Oncology team should review infection precautions and treatment timing.
""",
        "labs_2026_03_10.txt": f"""{_header(profile, "Laboratory Results 2026-03-10", "lab/results")}
Diagnosis context: {profile.diagnosis}, {profile.stage}.
CBC: WBC 3.1 K/uL, ANC 1.6 K/uL, hemoglobin 11.0 g/dL, platelets 166 K/uL.
Tumor marker or disease marker: {profile.key_lab_fact}.
Plain annotation: values are plausible synthetic monitoring data for retrieval testing.
""",
    }


def _appointment_docs(profile: PatientProfile) -> dict[str, str]:
    return {
        "appointment_summary_2026_02_18.txt": f"""{_header(profile, "Oncology Appointment Summary 2026-02-18", "appointment summary")}
Clinician: {profile.oncologist}.
Assessment: {profile.display_name} is being treated for {profile.diagnosis}, {profile.stage}.
Treatment plan: {profile.treatment_plan}.
Symptoms discussed: fatigue, appetite changes, and questions about appointment timing.
Next steps: continue current plan, repeat labs before the next cycle, and call urgently for fever or worsening symptoms.
""",
        "appointment_summary_2026_03_12.txt": f"""{_header(profile, "Oncology Appointment Summary 2026-03-12", "appointment summary")}
Clinician: {profile.oncologist}.
Assessment: interval status reviewed after recent treatment.
Plan update: keep {profile.treatment_plan}; confirm supportive medications are available at home.
Next steps: review insurance status, track referral status, and bring medication list to the next visit.
""",
    }


def _insurance_doc(profile: PatientProfile) -> dict[str, str]:
    return {
        "insurance_preauthorization.txt": f"""{_header(profile, "Insurance Pre-authorization Letter", "insurance/pre-approval")}
Insurer: {profile.insurer}.
Requested service: oncology treatment support for {profile.treatment_plan}.
Status: {profile.insurance_status}.
Ground truth: {profile.key_admin_fact}.
Patient action: keep this letter with oncology records and ask the clinic coordinator about any pending or expired authorization.
""",
    }


def _referral_doc(profile: PatientProfile) -> dict[str, str]:
    return {
        "referral_status.txt": f"""{_header(profile, "Referral Status", "referral")}
Referral target: {profile.referral_target}.
Referral status: {profile.referral_status}.
Reason: care coordination for {profile.diagnosis}, {profile.stage}.
Ground truth: referral to {profile.referral_target} is {profile.referral_status}.
Next step: clinic coordinator should verify the referral before the next oncology appointment.
""",
    }


def _medication_doc(profile: PatientProfile) -> dict[str, str]:
    return {
        "medication_schedule.txt": f"""{_header(profile, "Medication List and Schedule", "medication records")}
Primary regimen: {profile.treatment_plan}.
Key medication fact: {profile.medication_fact}.
Supportive medications: ondansetron 8 mg as needed for nausea, polyethylene glycol as needed for constipation, acetaminophen only after checking temperature.
Safety note: this is synthetic test data; medication details are plausible but not medical advice.
""",
    }


def _patient_docs(profile: PatientProfile) -> dict[str, str]:
    docs = {}
    docs.update(_lab_docs(profile))
    docs.update(_appointment_docs(profile))
    docs.update(_insurance_doc(profile))
    docs.update(_referral_doc(profile))
    docs.update(_medication_doc(profile))
    return docs


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "synthetic_data_notice": "All documents are fictional and generated for CarePilot testing only.",
        "patient_count": len(PATIENTS),
        "patients": {},
    }

    for profile in PATIENTS:
        patient_dir = OUTPUT_ROOT / profile.patient_id
        docs = _patient_docs(profile)
        files: list[str] = []
        for file_name, content in docs.items():
            _write(patient_dir / file_name, content)
            files.append(f"{profile.patient_id}/{file_name}")

        manifest["patients"][profile.patient_id] = {
            "username": profile.username,
            "profile": {
                "display_name": profile.display_name,
                "age": profile.age,
                "sex": profile.sex,
                "gender": profile.gender,
                "home_city": profile.home_city,
                "diagnosis": profile.diagnosis,
                "stage": profile.stage,
                "treatment_plan": profile.treatment_plan,
                "oncologist": profile.oncologist,
                "insurer": profile.insurer,
            },
            "files": files,
            "ground_truth_facts": {
                "lab": profile.key_lab_fact,
                "insurance": profile.key_admin_fact,
                "referral": f"referral to {profile.referral_target} is {profile.referral_status}",
                "medication": profile.medication_fact,
            },
        }

    _write(OUTPUT_ROOT / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=True))
    print(f"Generated {len(PATIENTS)} synthetic patients in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
