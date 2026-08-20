import json
from pathlib import Path

from src.scripts.build_clinical_index import _read_markdown_with_metadata
from src.scripts.generate_synthetic_patients import PATIENTS, _patient_docs


def test_requirements_include_pdf_document_support():
    requirements = Path("requirements.txt").read_text(encoding="utf-8").splitlines()

    assert "markitdown[pdf]==0.1.6" in requirements


def test_synthetic_patient_templates_cover_required_categories():
    required_files = {
        "labs_2026_02_17.txt",
        "labs_2026_03_10.txt",
        "appointment_summary_2026_02_18.txt",
        "appointment_summary_2026_03_12.txt",
        "insurance_preauthorization.txt",
        "referral_status.txt",
        "medication_schedule.txt",
    }

    assert len(PATIENTS) == 10
    for profile in PATIENTS:
        docs = _patient_docs(profile)
        assert set(docs) == required_files
        assert all(profile.patient_id in content for content in docs.values())
        assert all("Synthetic data notice" in content for content in docs.values())


def test_generated_manifest_matches_files_on_disk():
    manifest = json.loads(open("data/synthetic_patients/manifest.json", encoding="utf-8").read())

    assert manifest["patient_count"] == 10
    assert len(manifest["patients"]) == 10
    for patient_id, patient_data in manifest["patients"].items():
        assert patient_data["username"] == patient_id
        assert len(patient_data["files"]) == 7
        assert set(patient_data["ground_truth_facts"]) == {"lab", "insurance", "referral", "medication"}


def test_read_markdown_with_metadata(tmp_path):
    path = tmp_path / "source.md"
    path.write_text('---\n{"title":"T","topic_tags":["tag"]}\n---\nBody text\n', encoding="utf-8")

    metadata, body = _read_markdown_with_metadata(path)

    assert metadata == {"title": "T", "topic_tags": ["tag"]}
    assert body == "Body text"


def test_read_markdown_without_frontmatter(tmp_path):
    path = tmp_path / "source.md"
    path.write_text("Body only", encoding="utf-8")

    metadata, body = _read_markdown_with_metadata(path)

    assert metadata == {}
    assert body == "Body only"
