"""Scrape authoritative clinical reference pages into a chunk-ready corpus.

Run from the CarePilot repository root:
    python -m src.data_scraping.scrape_clinical_data
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date
from html import unescape
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = PROJECT_ROOT / "data" / "clinical_corpus"
RAW_DIR = CORPUS_ROOT / "raw"
CLEAN_DIR = CORPUS_ROOT / "cleaned"
MANIFEST_PATH = CORPUS_ROOT / "source_manifest.json"
README_PATH = CORPUS_ROOT / "README.md"
SOURCE_LOCATION_PLOT_PATH = CORPUS_ROOT / "source_locations_pie.png"

REQUEST_TIMEOUT_SECONDS = 30
REQUEST_DELAY_SECONDS = 0.4
MAX_DISCOVERED_PER_SEED = 120
MAX_HEALTHCARE_API_PAGES = 260
MIN_TEXT_CHARS = 400

HEADERS = {
    "User-Agent": (
        "CarePilot clinical corpus scraper/0.1 "
        "(course project; contact repository owner)"
    )
}

SKIP_EXACT_LINES = {
    "Email this page",
    "Print this page",
    "Print",
    "Español",
    "Sources Share",
    "Listen to pronunciation",
    "On this page",
    "View All",
    "You Are Here:",
    "Home",
}

SKIP_PREFIXES = (
    "To use the sharing features on this page",
    "You are leaving",
    "Continue[Button:",
    "Share sensitive information only on official",
    "An official website of the United States government",
)

HEALTHCARE_API_INDEX_URL = "https://www.healthcare.gov/api/index.json"
HEALTHCARE_RELEVANT_PATH_PREFIXES = (
    "/appeal-insurance-company-decision/",
    "/coverage-outside-open-enrollment/",
    "/get-answers/",
    "/glossary/",
    "/health-care-law-protections/",
    "/marketplace-appeals/",
    "/tips-and-troubleshooting/",
    "/verify-information/",
)
HEALTHCARE_RELEVANT_KEYWORDS = (
    "appeal",
    "application",
    "authorized representative",
    "benefit",
    "billing",
    "claim",
    "coinsurance",
    "copayment",
    "coverage",
    "deductible",
    "document",
    "eligibility",
    "enroll",
    "external review",
    "form",
    "health plan",
    "income",
    "insurance",
    "marketplace",
    "medicaid",
    "medicare",
    "network",
    "out-of-pocket",
    "payment",
    "preauthorization",
    "prior authorization",
    "referral",
    "special enrollment",
    "submit",
    "upload",
    "verification",
)


@dataclass(frozen=True)
class SourcePage:
    source_name: str
    url: str
    topic_tags: tuple[str, ...]
    license_note: str
    extractor: str = "html"
    min_text_chars: int = MIN_TEXT_CHARS
    jurisdiction: str = ""


@dataclass(frozen=True)
class DiscoverySeed:
    source_name: str
    seed_url: str
    include_patterns: tuple[str, ...]
    topic_tags: tuple[str, ...]
    license_note: str


LICENSE_NOTES = {
    "NCI": (
        "NCI states that, unless otherwise indicated, text in NCI products is "
        "free of copyright and may be reused with credit."
    ),
    "CDC": (
        "CDC states that most CDC/ATSDR website information is public domain "
        "and may be freely used or reproduced, except marked third-party items."
    ),
    "MedlinePlus": (
        "MedlinePlus health-topic summaries produced by NLM/NIH are public "
        "domain; this scraper avoids Medical Encyclopedia and drug monograph pages."
    ),
    "HealthCare.gov": (
        "HealthCare.gov provides official educational content through an open "
        "JSON content API intended for reuse by developers."
    ),
    "CMS": (
        "CMS is a U.S. federal agency; its public educational text is generally "
        "public-domain government work unless marked otherwise."
    ),
    "Medicare.gov": (
        "Medicare.gov is an official U.S. government Medicare site; its public "
        "educational text is generally public-domain government work unless marked otherwise."
    ),
    "European Union": (
        "Your Europe is an official European Union public information portal; "
        "EU reuse terms generally allow reuse with acknowledgement unless otherwise stated."
    ),
    "NHS": (
        "NHS and NHS England pages are official UK public-health sources; reuse "
        "is usually covered by Open Government Licence terms unless marked otherwise."
    ),
    "Israel Ministry of Health": (
        "Official Israel Ministry of Health service page. Israeli government "
        "copyright terms are more restrictive than U.S. public-domain sources; "
        "use for source-grounded reference, not redistribution."
    ),
    "Israel National Insurance": (
        "Official National Insurance Institute of Israel page. NII terms allow "
        "fair use with attribution but restrict copying/redistribution; use for "
        "source-grounded reference, not redistribution."
    ),
}

DISCOVERY_SEEDS: tuple[DiscoverySeed, ...] = (
    DiscoverySeed(
        source_name="NCI",
        seed_url="https://www.cancer.gov/types/common-cancers",
        include_patterns=(r"^https://www\.cancer\.gov/types/[^/?#]+/?$",),
        topic_tags=("cancer_types", "overview"),
        license_note=LICENSE_NOTES["NCI"],
    ),
    DiscoverySeed(
        source_name="NCI",
        seed_url="https://www.cancer.gov/about-cancer/treatment/types",
        include_patterns=(
            r"^https://www\.cancer\.gov/about-cancer/treatment/types/[^?#]+$",
        ),
        topic_tags=("treatment", "modalities"),
        license_note=LICENSE_NOTES["NCI"],
    ),
    DiscoverySeed(
        source_name="NCI",
        seed_url="https://www.cancer.gov/about-cancer/treatment/side-effects",
        include_patterns=(
            r"^https://www\.cancer\.gov/about-cancer/treatment/side-effects/[^?#]+$",
        ),
        topic_tags=("side_effects", "urgent_symptoms"),
        license_note=LICENSE_NOTES["NCI"],
    ),
    DiscoverySeed(
        source_name="CDC",
        seed_url="https://www.cdc.gov/cancer-survivors/index.html",
        include_patterns=(r"^https://www\.cdc\.gov/cancer-survivors/[^?#]+\.html$",),
        topic_tags=("survivorship", "care_process"),
        license_note=LICENSE_NOTES["CDC"],
    ),
    DiscoverySeed(
        source_name="HealthCare.gov",
        seed_url="https://www.healthcare.gov/get-answers/",
        include_patterns=(
            r"^https://www\.healthcare\.gov/(appeal-insurance-company-decision|coverage-outside-open-enrollment|get-answers|health-care-law-protections|marketplace-appeals|tips-and-troubleshooting|verify-information)/[^?#]*$",
        ),
        topic_tags=("insurance", "document_workflows", "care_process"),
        license_note=LICENSE_NOTES["HealthCare.gov"],
    ),
    DiscoverySeed(
        source_name="HealthCare.gov",
        seed_url="https://www.healthcare.gov/verify-information/",
        include_patterns=(
            r"^https://www\.healthcare\.gov/(coverage-outside-open-enrollment|tips-and-troubleshooting|verify-information)/[^?#]*$",
        ),
        topic_tags=("insurance", "document_workflows", "verification"),
        license_note=LICENSE_NOTES["HealthCare.gov"],
    ),
    DiscoverySeed(
        source_name="HealthCare.gov",
        seed_url="https://www.healthcare.gov/marketplace-appeals/",
        include_patterns=(
            r"^https://www\.healthcare\.gov/(appeal-insurance-company-decision|marketplace-appeals)/[^?#]*$",
        ),
        topic_tags=("insurance", "appeals", "document_workflows"),
        license_note=LICENSE_NOTES["HealthCare.gov"],
    ),
    DiscoverySeed(
        source_name="Medicare.gov",
        seed_url="https://www.medicare.gov/providers-services/claims-appeals-complaints/appeals",
        include_patterns=(
            r"^https://www\.medicare\.gov/(basics/forms-publications-mailings/forms|providers-services/claims-appeals-complaints)/[^?#]*$",
        ),
        topic_tags=("insurance", "medicare", "appeals", "document_workflows"),
        license_note=LICENSE_NOTES["Medicare.gov"],
    ),
    DiscoverySeed(
        source_name="CMS",
        seed_url="https://www.cms.gov/cciio/resources/fact-sheets-and-faqs/appeals06152012a",
        include_patterns=(
            r"^https://www\.cms\.gov/(cciio/resources|medicare/appeals-grievances|priorities/electronic-prior-authorization|data-research/monitoring-programs)/[^?#]*$",
        ),
        topic_tags=("insurance", "appeals", "prior_authorization", "document_workflows"),
        license_note=LICENSE_NOTES["CMS"],
    ),
    DiscoverySeed(
        source_name="European Union",
        seed_url="https://europa.eu/youreurope/citizens/health/planned-healthcare/index_en.htm",
        include_patterns=(
            r"^https://europa\.eu/youreurope/citizens/health/(planned-healthcare|unplanned-healthcare|prescription-medicine-abroad)/[^?#]*_en\.htm$",
        ),
        topic_tags=("europe", "eu", "insurance", "prior_authorization", "reimbursement", "document_workflows"),
        license_note=LICENSE_NOTES["European Union"],
    ),
    DiscoverySeed(
        source_name="NHS",
        seed_url="https://www.england.nhs.uk/london/london-clinical-networks/our-networks/cancer-earlier-diagnosis/urgent-cancer-referrals/",
        include_patterns=(
            r"^https://www\.england\.nhs\.uk/(digitaltechnology|london|long-read)/[^?#]*$",
        ),
        topic_tags=("europe", "uk", "nhs", "cancer_referrals", "document_workflows"),
        license_note=LICENSE_NOTES["NHS"],
    ),
)

CURATED_PAGES: tuple[SourcePage, ...] = (
    SourcePage(
        "NCI",
        "https://www.cancer.gov/about-cancer/understanding/what-is-cancer",
        ("overview", "terminology"),
        LICENSE_NOTES["NCI"],
    ),
    SourcePage(
        "NCI",
        "https://www.cancer.gov/about-cancer/diagnosis-staging/diagnosis",
        ("diagnosis_process", "testing"),
        LICENSE_NOTES["NCI"],
    ),
    SourcePage(
        "NCI",
        "https://www.cancer.gov/about-cancer/diagnosis-staging/staging",
        ("diagnosis_process", "terminology"),
        LICENSE_NOTES["NCI"],
    ),
    SourcePage(
        "NCI",
        "https://www.cancer.gov/about-cancer/diagnosis-staging/prognosis",
        ("terminology", "care_process"),
        LICENSE_NOTES["NCI"],
    ),
    SourcePage(
        "NCI",
        "https://www.cancer.gov/about-cancer/treatment/questions",
        ("appointments", "care_process"),
        LICENSE_NOTES["NCI"],
    ),
    SourcePage(
        "NCI",
        "https://www.cancer.gov/about-cancer/managing-care",
        ("appointments", "care_process"),
        LICENSE_NOTES["NCI"],
    ),
    SourcePage(
        "NCI",
        "https://www.cancer.gov/about-cancer/coping",
        ("coping", "survivorship"),
        LICENSE_NOTES["NCI"],
    ),
    SourcePage(
        "NCI",
        "https://www.cancer.gov/about-cancer/advanced-cancer",
        ("advanced_cancer", "care_process"),
        LICENSE_NOTES["NCI"],
    ),
    SourcePage(
        "NCI",
        "https://www.cancer.gov/publications/dictionaries/cancer-terms/def/neutropenia",
        ("terminology", "side_effects"),
        LICENSE_NOTES["NCI"],
    ),
    SourcePage(
        "NCI",
        "https://www.cancer.gov/publications/dictionaries/cancer-terms/def/chemotherapy",
        ("terminology", "treatment"),
        LICENSE_NOTES["NCI"],
    ),
    SourcePage(
        "NCI",
        "https://www.cancer.gov/publications/dictionaries/cancer-terms/def/immunotherapy",
        ("terminology", "treatment"),
        LICENSE_NOTES["NCI"],
    ),
    SourcePage(
        "NCI",
        "https://www.cancer.gov/publications/dictionaries/cancer-terms/def/metastasis",
        ("terminology", "diagnosis_process"),
        LICENSE_NOTES["NCI"],
    ),
    SourcePage(
        "NCI",
        "https://www.cancer.gov/publications/dictionaries/cancer-terms/def/biopsy",
        ("terminology", "testing"),
        LICENSE_NOTES["NCI"],
    ),
    SourcePage(
        "NCI",
        "https://www.cancer.gov/publications/dictionaries/cancer-terms/def/remission",
        ("terminology", "survivorship"),
        LICENSE_NOTES["NCI"],
    ),
    SourcePage(
        "MedlinePlus",
        "https://medlineplus.gov/cancer.html",
        ("overview", "diagnosis_process", "treatment"),
        LICENSE_NOTES["MedlinePlus"],
    ),
    SourcePage(
        "MedlinePlus",
        "https://medlineplus.gov/cancerchemotherapy.html",
        ("treatment", "side_effects"),
        LICENSE_NOTES["MedlinePlus"],
    ),
    SourcePage(
        "MedlinePlus",
        "https://medlineplus.gov/radiationtherapy.html",
        ("treatment", "side_effects"),
        LICENSE_NOTES["MedlinePlus"],
    ),
    SourcePage(
        "MedlinePlus",
        "https://medlineplus.gov/cancerlivingwithcancer.html",
        ("survivorship", "coping"),
        LICENSE_NOTES["MedlinePlus"],
    ),
    SourcePage(
        "MedlinePlus",
        "https://medlineplus.gov/palliativecare.html",
        ("palliative_care", "care_process"),
        LICENSE_NOTES["MedlinePlus"],
    ),
    SourcePage(
        "CDC",
        "https://www.cdc.gov/cancer-survivors/patients/index.html",
        ("care_process", "treatment"),
        LICENSE_NOTES["CDC"],
    ),
    SourcePage(
        "CDC",
        "https://www.cdc.gov/cancer-survivors/patients/treatments.html",
        ("treatment", "care_process"),
        LICENSE_NOTES["CDC"],
    ),
    SourcePage(
        "CDC",
        "https://www.cdc.gov/cancer-survivors/patients/side-effects-of-treatment.html",
        ("side_effects", "urgent_symptoms"),
        LICENSE_NOTES["CDC"],
    ),
    SourcePage(
        "CDC",
        "https://www.cdc.gov/cancer-survivors/patients/paying-for-cancer-treatment.html",
        ("insurance", "care_process"),
        LICENSE_NOTES["CDC"],
    ),
    SourcePage(
        "CDC",
        "https://www.cdc.gov/cancer-survivors/life-after-cancer/survivorship-care-plans.html",
        ("survivorship", "medical_records", "appointments"),
        LICENSE_NOTES["CDC"],
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/verify-information/",
        ("insurance", "document_workflows", "verification"),
        LICENSE_NOTES["HealthCare.gov"],
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/verify-information/documents-and-deadlines/",
        ("insurance", "document_workflows", "verification", "deadlines"),
        LICENSE_NOTES["HealthCare.gov"],
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/tips-and-troubleshooting/uploading-documents/",
        ("insurance", "document_workflows", "uploading_documents"),
        LICENSE_NOTES["HealthCare.gov"],
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/coverage-outside-open-enrollment/confirm-special-enrollment-period/",
        ("insurance", "document_workflows", "special_enrollment"),
        LICENSE_NOTES["HealthCare.gov"],
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/marketplace-appeals/",
        ("insurance", "appeals", "document_workflows"),
        LICENSE_NOTES["HealthCare.gov"],
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/marketplace-appeals/ways-to-appeal/index.html",
        ("insurance", "appeals", "document_workflows"),
        LICENSE_NOTES["HealthCare.gov"],
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/marketplace-appeals/after-you-file/",
        ("insurance", "appeals", "next_steps"),
        LICENSE_NOTES["HealthCare.gov"],
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/marketplace-appeals/getting-help/",
        ("insurance", "appeals", "authorized_representative"),
        LICENSE_NOTES["HealthCare.gov"],
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/marketplace-appeals/expedited-appeal/",
        ("insurance", "appeals", "urgent_workflows"),
        LICENSE_NOTES["HealthCare.gov"],
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/appeal-insurance-company-decision/",
        ("insurance", "appeals", "claims"),
        LICENSE_NOTES["HealthCare.gov"],
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/appeal-insurance-company-decision/internal-appeals/",
        ("insurance", "appeals", "document_workflows", "claims"),
        LICENSE_NOTES["HealthCare.gov"],
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/appeal-insurance-company-decision/external-review/",
        ("insurance", "appeals", "external_review", "next_steps"),
        LICENSE_NOTES["HealthCare.gov"],
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/health-care-law-protections/summary-of-benefits-and-coverage/",
        ("insurance", "coverage_documents", "benefits"),
        LICENSE_NOTES["HealthCare.gov"],
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/glossary/referral/",
        ("insurance", "referrals"),
        LICENSE_NOTES["HealthCare.gov"],
        extractor="healthcare_json",
        min_text_chars=80,
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/glossary/preauthorization/",
        ("insurance", "prior_authorization"),
        LICENSE_NOTES["HealthCare.gov"],
        extractor="healthcare_json",
        min_text_chars=80,
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/glossary/prior-authorization/",
        ("insurance", "prior_authorization"),
        LICENSE_NOTES["HealthCare.gov"],
        extractor="healthcare_json",
        min_text_chars=80,
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/glossary/appeal/",
        ("insurance", "appeals"),
        LICENSE_NOTES["HealthCare.gov"],
        extractor="healthcare_json",
        min_text_chars=80,
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/glossary/claim/",
        ("insurance", "billing"),
        LICENSE_NOTES["HealthCare.gov"],
        extractor="healthcare_json",
        min_text_chars=80,
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/glossary/external-review/",
        ("insurance", "appeals", "external_review"),
        LICENSE_NOTES["HealthCare.gov"],
        extractor="healthcare_json",
        min_text_chars=80,
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/glossary/deductible/",
        ("insurance", "billing"),
        LICENSE_NOTES["HealthCare.gov"],
        extractor="healthcare_json",
        min_text_chars=80,
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/glossary/copayment/",
        ("insurance", "billing"),
        LICENSE_NOTES["HealthCare.gov"],
        extractor="healthcare_json",
        min_text_chars=80,
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/glossary/coinsurance/",
        ("insurance", "billing"),
        LICENSE_NOTES["HealthCare.gov"],
        extractor="healthcare_json",
        min_text_chars=80,
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/glossary/network/",
        ("insurance", "care_process"),
        LICENSE_NOTES["HealthCare.gov"],
        extractor="healthcare_json",
        min_text_chars=80,
    ),
    SourcePage(
        "HealthCare.gov",
        "https://www.healthcare.gov/glossary/specialist/",
        ("insurance", "referrals"),
        LICENSE_NOTES["HealthCare.gov"],
        extractor="healthcare_json",
        min_text_chars=80,
    ),
    SourcePage(
        "CMS",
        "https://www.cms.gov/cciio/resources/fact-sheets-and-faqs/appeals06152012a",
        ("insurance", "appeals", "prior_authorization", "claims"),
        LICENSE_NOTES["CMS"],
    ),
    SourcePage(
        "CMS",
        "https://www.cms.gov/medicare/appeals-grievances/managed-care/reconsideration-advantage-health-plan-part-c",
        ("insurance", "medicare", "appeals", "next_steps"),
        LICENSE_NOTES["CMS"],
    ),
    SourcePage(
        "CMS",
        "https://www.cms.gov/data-research/monitoring-programs/medicare-fee-service-compliance-programs/prior-authorization-and-pre-claim-review-initiatives/prior-authorization-process-certain-durable-medical-equipment-prosthetics-orthotics-and-supplies",
        ("insurance", "medicare", "prior_authorization", "document_workflows"),
        LICENSE_NOTES["CMS"],
    ),
    SourcePage(
        "Medicare.gov",
        "https://www.medicare.gov/providers-services/claims-appeals-complaints/appeals",
        ("insurance", "medicare", "appeals", "next_steps"),
        LICENSE_NOTES["Medicare.gov"],
    ),
    SourcePage(
        "Medicare.gov",
        "https://www.medicare.gov/basics/forms-publications-mailings/forms/appeals",
        ("insurance", "medicare", "appeal_forms", "document_workflows"),
        LICENSE_NOTES["Medicare.gov"],
    ),
    SourcePage(
        "European Union",
        "https://europa.eu/youreurope/citizens/health/planned-healthcare/index_en.htm",
        ("europe", "eu", "insurance", "planned_treatment", "document_workflows"),
        LICENSE_NOTES["European Union"],
        jurisdiction="eu",
    ),
    SourcePage(
        "European Union",
        "https://europa.eu/youreurope/citizens/health/planned-healthcare/right-to-treatment/index_en.htm",
        ("europe", "eu", "insurance", "planned_treatment", "prior_authorization", "reimbursement", "document_workflows"),
        LICENSE_NOTES["European Union"],
        jurisdiction="eu",
    ),
    SourcePage(
        "European Union",
        "https://europa.eu/youreurope/citizens/health/planned-healthcare/information-points/index_en.htm",
        ("europe", "eu", "insurance", "national_contact_points", "document_workflows"),
        LICENSE_NOTES["European Union"],
        jurisdiction="eu",
    ),
    SourcePage(
        "European Union",
        "https://europa.eu/youreurope/citizens/health/unplanned-healthcare/temporary-stays/index_en.htm",
        ("europe", "eu", "ehic", "insurance", "urgent_care"),
        LICENSE_NOTES["European Union"],
        jurisdiction="eu",
    ),
    SourcePage(
        "European Union",
        "https://europa.eu/youreurope/citizens/health/unplanned-healthcare/payments-reimbursements/index_en.htm",
        ("europe", "eu", "insurance", "claims", "reimbursement", "document_workflows"),
        LICENSE_NOTES["European Union"],
        jurisdiction="eu",
    ),
    SourcePage(
        "European Union",
        "https://europa.eu/youreurope/citizens/health/prescription-medicine-abroad/prescriptions/index_en.htm",
        ("europe", "eu", "prescriptions", "document_workflows"),
        LICENSE_NOTES["European Union"],
        jurisdiction="eu",
    ),
    SourcePage(
        "European Union",
        "https://europa.eu/youreurope/citizens/health/prescription-medicine-abroad/expenses-reimbursements/index_en.htm",
        ("europe", "eu", "prescriptions", "claims", "reimbursement"),
        LICENSE_NOTES["European Union"],
        jurisdiction="eu",
    ),
    SourcePage(
        "NHS",
        "https://www.nhs.uk/nhs-services/hospitals/referrals-for-specialist-care/",
        ("europe", "uk", "nhs", "referrals", "appointments"),
        LICENSE_NOTES["NHS"],
        jurisdiction="uk",
    ),
    SourcePage(
        "NHS",
        "https://www.nhs.uk/nhs-services/hospitals/book-an-appointment/",
        ("europe", "uk", "nhs", "referrals", "appointments", "document_workflows"),
        LICENSE_NOTES["NHS"],
        jurisdiction="uk",
    ),
    SourcePage(
        "NHS",
        "https://www.england.nhs.uk/long-read/e-referrals/",
        ("europe", "uk", "nhs", "referrals", "cancer_referrals", "document_workflows"),
        LICENSE_NOTES["NHS"],
        jurisdiction="uk",
    ),
    SourcePage(
        "NHS",
        "https://www.england.nhs.uk/london/london-clinical-networks/our-networks/cancer-earlier-diagnosis/urgent-cancer-referrals/",
        ("europe", "uk", "nhs", "cancer_referrals", "referral_forms", "document_workflows"),
        LICENSE_NOTES["NHS"],
        jurisdiction="uk",
    ),
    SourcePage(
        "NHS",
        "https://www.england.nhs.uk/long-read/national-cancer-waiting-times-monitoring-dataset-guidance/",
        ("europe", "uk", "nhs", "cancer_referrals", "next_steps"),
        LICENSE_NOTES["NHS"],
        jurisdiction="uk",
    ),
    SourcePage(
        "Israel Ministry of Health",
        "https://www.gov.il/en/service/national-health-insurance-law-complaint-submission",
        ("israel", "insurance", "hmo", "complaints", "document_workflows"),
        LICENSE_NOTES["Israel Ministry of Health"],
    ),
    SourcePage(
        "Israel Ministry of Health",
        "https://www.gov.il/en/service/appeal-an-hmo-decision",
        ("israel", "insurance", "hmo", "appeals", "document_workflows"),
        LICENSE_NOTES["Israel Ministry of Health"],
    ),
    SourcePage(
        "Israel Ministry of Health",
        "https://www.gov.il/en/service/dying-patient-request",
        ("israel", "medical_forms", "advance_directives", "document_workflows"),
        LICENSE_NOTES["Israel Ministry of Health"],
    ),
    SourcePage(
        "Israel National Insurance",
        "https://www.btl.gov.il/English%20Homepage/Benefits/Disability%20Insurance/Pages/HolimONkologim.aspx",
        ("israel", "oncology", "benefits", "disability", "document_workflows"),
        LICENSE_NOTES["Israel National Insurance"],
    ),
    SourcePage(
        "Israel National Insurance",
        "https://www.btl.gov.il/English%20Homepage/Insurance/Health%20Insurance/Pages/HealthInsuranceLaw.aspx",
        ("israel", "insurance", "health_basket", "hmo"),
        LICENSE_NOTES["Israel National Insurance"],
    ),
)


def main() -> None:
    prepare_directories()
    pages = collect_source_pages()
    results = []

    for index, page in enumerate(pages, start=1):
        try:
            print(f"[{index}/{len(pages)}] scraping {page.url}")
            document = scrape_page(page)
            if len(document["text"]) < page.min_text_chars:
                print(f"  skipped: only {len(document['text'])} chars after cleaning")
                continue
            output_path = write_clean_document(page, document)
            results.append({**document["metadata"], "output_path": str(output_path)})
        except requests.HTTPError as exc:
            print(f"  skipped: HTTP {exc.response.status_code}")
        except requests.RequestException as exc:
            print(f"  skipped: request failed: {exc}")
        except ValueError as exc:
            print(f"  skipped: {exc}")
        time.sleep(REQUEST_DELAY_SECONDS)

    write_manifest(results)
    write_readme(results)
    write_source_location_plot(results)
    print(f"done: wrote {len(results)} cleaned documents to {CLEAN_DIR}")


def prepare_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    (CORPUS_ROOT / ".gitkeep").touch(exist_ok=True)


def collect_source_pages() -> list[SourcePage]:
    pages: dict[str, SourcePage] = {page.url: page for page in CURATED_PAGES}
    pages.update({page.url: page for page in discover_healthcare_api_pages()})

    for seed in DISCOVERY_SEEDS:
        try:
            html = fetch_text(seed.seed_url)
        except requests.RequestException as exc:
            print(f"discovery skipped for {seed.seed_url}: {exc}")
            continue

        soup = BeautifulSoup(html, "html.parser")
        discovered = []
        for link in soup.find_all("a", href=True):
            absolute_url = normalize_url(urljoin(seed.seed_url, link["href"]))
            if not any(re.match(pattern, absolute_url) for pattern in seed.include_patterns):
                continue
            if absolute_url == normalize_url(seed.seed_url):
                continue
            discovered.append(absolute_url)

        for url in sorted(set(discovered))[:MAX_DISCOVERED_PER_SEED]:
            extractor = "healthcare_json" if seed.source_name == "HealthCare.gov" else "html"
            min_text_chars = 80 if seed.source_name == "HealthCare.gov" and "/glossary/" in url else MIN_TEXT_CHARS
            pages.setdefault(
                url,
                SourcePage(
                    source_name=seed.source_name,
                    url=url,
                    topic_tags=seed.topic_tags,
                    license_note=seed.license_note,
                    extractor=extractor,
                    min_text_chars=min_text_chars,
                ),
            )

    return sorted(pages.values(), key=lambda item: (item.source_name, item.url))


def discover_healthcare_api_pages() -> list[SourcePage]:
    try:
        raw = fetch_text(HEALTHCARE_API_INDEX_URL)
    except requests.RequestException as exc:
        print(f"HealthCare.gov API discovery skipped: {exc}")
        return []

    pages = []
    for item in json.loads(raw):
        url = normalize_url(urljoin("https://www.healthcare.gov", item.get("url", "")))
        if not is_relevant_healthcare_url(url, item):
            continue
        pages.append(
                SourcePage(
                    source_name="HealthCare.gov",
                    url=url,
                    topic_tags=healthcare_tags_for(item),
                    license_note=LICENSE_NOTES["HealthCare.gov"],
                    extractor="healthcare_json",
                    min_text_chars=80 if "/glossary/" in url else MIN_TEXT_CHARS,
                    jurisdiction="us",
                )
            )

    unique_pages = {page.url: page for page in pages}
    return sorted(unique_pages.values(), key=lambda page: page.url)[:MAX_HEALTHCARE_API_PAGES]


def is_relevant_healthcare_url(url: str, item: dict[str, object]) -> bool:
    parsed = urlparse(url)
    if parsed.netloc != "www.healthcare.gov":
        return False
    if not any(parsed.path.startswith(prefix) for prefix in HEALTHCARE_RELEVANT_PATH_PREFIXES):
        return False
    text = " ".join(
        str(value)
        for key, value in item.items()
        if key in {"title", "bite", "categories", "tags", "topics"}
    ).lower()
    return any(keyword in text for keyword in HEALTHCARE_RELEVANT_KEYWORDS)


def healthcare_tags_for(item: dict[str, object]) -> tuple[str, ...]:
    raw_text = " ".join(
        str(value)
        for key, value in item.items()
        if key in {"title", "bite", "categories", "tags", "topics", "url"}
    ).lower()
    tags = ["insurance"]
    if any(word in raw_text for word in ("document", "upload", "submit", "verification")):
        tags.append("document_workflows")
    if any(word in raw_text for word in ("appeal", "external review")):
        tags.append("appeals")
    if any(word in raw_text for word in ("prior authorization", "preauthorization", "referral")):
        tags.append("prior_authorization")
    if any(word in raw_text for word in ("claim", "payment", "billing", "deductible", "copay", "coinsurance")):
        tags.append("billing")
    if "special enrollment" in raw_text:
        tags.append("special_enrollment")
    if any(word in raw_text for word in ("medicaid", "medicare")):
        tags.append("public_coverage")
    return tuple(dict.fromkeys(tags))


def scrape_page(page: SourcePage) -> dict[str, object]:
    if page.extractor == "healthcare_json":
        title, text, raw = scrape_healthcare_json(page.url)
        raw_suffix = ".json"
    else:
        raw = fetch_text(page.url)
        title, text = scrape_html(page.url, raw)
        raw_suffix = ".html"

    raw_path = RAW_DIR / f"{stable_slug(page.url)}{raw_suffix}"
    raw_path.write_text(raw, encoding="utf-8")

    metadata = {
        "source_url": page.url,
        "source_name": page.source_name,
        "title": title,
        "retrieved_date": date.today().isoformat(),
        "topic_tags": list(page.topic_tags),
        "jurisdiction": infer_jurisdiction(page),
        "license_note": page.license_note,
        "raw_path": str(raw_path.relative_to(PROJECT_ROOT)),
    }
    return {"metadata": metadata, "text": text}


def scrape_healthcare_json(url: str) -> tuple[str, str, str]:
    raw = fetch_text(first_working_healthcare_json_url(url))
    payload = json.loads(raw)
    title = normalize_whitespace(payload.get("title", "")) or title_from_url(url)
    content = payload.get("content", "")
    text = clean_html_fragment(url, content)
    return title, text, raw


def infer_jurisdiction(page: SourcePage) -> str:
    if page.jurisdiction:
        return page.jurisdiction
    if page.source_name in {"HealthCare.gov", "CMS", "Medicare.gov"}:
        return "us"
    if page.source_name == "European Union":
        return "eu"
    if page.source_name == "NHS":
        return "uk"
    if page.source_name in {"Israel Ministry of Health", "Israel National Insurance"}:
        return "israel"
    return "general"


def first_working_healthcare_json_url(url: str) -> str:
    candidates = []
    normalized = url.rstrip("/")
    candidates.append(f"{normalized}.json")
    if normalized.endswith("/index.html"):
        candidates.append(normalized.removesuffix(".html") + ".json")
        candidates.append(normalized.removesuffix("/index.html") + ".json")
    for candidate in dict.fromkeys(candidates):
        try:
            response = requests.head(candidate, headers=HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code < 400:
                return candidate
        except requests.RequestException:
            continue
    return candidates[0]


def scrape_html(url: str, html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = extract_title(url, soup)
    text = clean_soup(url, soup)
    return title, text


def fetch_text(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    if not response.encoding:
        response.encoding = "utf-8"
    return response.text


def clean_html_fragment(base_url: str, html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return clean_soup(base_url, soup)


def clean_soup(base_url: str, soup: BeautifulSoup) -> str:
    for selector in (
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "form",
        "button",
        "svg",
        "img",
        "picture",
        "video",
        "audio",
        "iframe",
        "noscript",
        "[role='navigation']",
        "[aria-hidden='true']",
        ".site-alert",
        ".usa-banner",
        ".breadcrumb",
        ".breadcrumbs",
        ".share",
        ".social-share",
    ):
        for element in soup.select(selector):
            element.decompose()

    container = pick_content_container(soup)
    lines = list(walk_content(container, base_url))
    lines = remove_boilerplate(lines)
    return "\n".join(lines).strip() + "\n"


def pick_content_container(soup: BeautifulSoup) -> Tag:
    selectors = (
        "main",
        "article",
        "[role='main']",
        "#main-content",
        ".main-content",
        ".contentzone",
        "#topic-summary",
        "body",
    )
    for selector in selectors:
        element = soup.select_one(selector)
        if isinstance(element, Tag):
            return element
    if soup.body:
        return soup.body
    if soup.get_text(" ", strip=True):
        return soup
    raise ValueError("no readable content container found")


def walk_content(node: Tag | NavigableString, base_url: str) -> Iterable[str]:
    if isinstance(node, NavigableString):
        return
    if not isinstance(node, Tag):
        return

    name = node.name.lower()
    if name in {"h1", "h2", "h3", "h4"}:
        level = {"h1": "#", "h2": "##", "h3": "###", "h4": "####"}[name]
        text = normalize_whitespace(node.get_text(" ", strip=True))
        if text:
            yield f"{level} {text}"
        return

    if name == "p":
        text = normalize_whitespace(node.get_text(" ", strip=True))
        if text:
            yield text
        return

    if name == "li":
        text = normalize_whitespace(node.get_text(" ", strip=True))
        if text:
            yield f"- {text}"
        return

    if name == "table":
        for line in table_to_lines(node):
            yield line
        return

    for child in node.children:
        yield from walk_content(child, base_url)


def table_to_lines(table: Tag) -> Iterable[str]:
    for row in table.find_all("tr"):
        cells = [
            normalize_whitespace(cell.get_text(" ", strip=True))
            for cell in row.find_all(["th", "td"])
        ]
        cells = [cell for cell in cells if cell]
        if cells:
            yield " | ".join(cells)


def remove_boilerplate(lines: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    previous = ""
    for raw_line in lines:
        line = normalize_whitespace(raw_line)
        if not line:
            continue
        bare_line = line.lstrip("#- ").strip()
        if bare_line in SKIP_EXACT_LINES:
            continue
        if any(bare_line.startswith(prefix) for prefix in SKIP_PREFIXES):
            continue
        if line == previous:
            continue
        cleaned.append(line)
        previous = line
    return cleaned


def extract_title(url: str, soup: BeautifulSoup) -> str:
    for selector in ("h1", "meta[property='og:title']", "title"):
        element = soup.select_one(selector)
        if not element:
            continue
        if element.name == "meta":
            title = element.get("content", "")
        else:
            title = element.get_text(" ", strip=True)
        title = normalize_whitespace(title)
        if title:
            return title
    return title_from_url(url)


def title_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return urlparse(url).netloc
    return path.rsplit("/", 1)[-1].replace("-", " ").title()


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    cleaned = parsed._replace(query="", fragment="")
    return cleaned.geturl().rstrip("/")


def normalize_whitespace(text: str) -> str:
    text = unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def stable_slug(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "index"
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", f"{parsed.netloc}-{path}").strip("-").lower()
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:90]}-{digest}"


def write_clean_document(page: SourcePage, document: dict[str, object]) -> Path:
    metadata = document["metadata"]
    text = str(document["text"])
    path = CLEAN_DIR / f"{stable_slug(page.url)}.md"
    front_matter = json.dumps(metadata, indent=2, ensure_ascii=False)
    path.write_text(f"---\n{front_matter}\n---\n\n{text}", encoding="utf-8")
    return path


def write_manifest(results: list[dict[str, object]]) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "retrieved_date": date.today().isoformat(),
                "document_count": len(results),
                "documents": results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_readme(results: list[dict[str, object]]) -> None:
    source_lines = [
        "# Clinical Corpus",
        "",
        "Generated by `python -m src.data_scraping.scrape_clinical_data` from the CarePilot repository root.",
        "",
        "Generated files under `raw/`, `cleaned/`, and `source_manifest.json` are rebuildable and ignored by git.",
        "",
        "## Source Families",
        "",
        "- NCI / Cancer.gov: authoritative National Cancer Institute oncology pages; NCI text is reusable unless otherwise indicated, with credit.",
        "- CDC Cancer Survivors: U.S. public-health survivorship guidance; CDC states most website information is public domain unless marked otherwise.",
        "- MedlinePlus health-topic summaries: NLM/NIH public-domain summaries; Medical Encyclopedia and drug monograph pages are deliberately excluded.",
        "- HealthCare.gov glossary/API: official CMS/HHS insurance terminology exposed through an open developer content API.",
        "- CMS and Medicare.gov: official federal pages covering claims, appeals, prior authorization, and Medicare forms.",
        "- European Union / Your Europe: official EU pages for planned treatment abroad, EHIC, reimbursement, prescriptions, and National Contact Points.",
        "- NHS / NHS England / NHS Digital: official UK pages for e-referrals, appointment booking, referral letters, urgent suspected cancer referral forms, and waiting-time pathways.",
        "- Israel Ministry of Health and National Insurance: official English Israeli pages for HMO complaints, HMO appeal documents, oncology benefits, and health-insurance basics; copyright is more restrictive than U.S. public-domain sources.",
        "",
        "## Source Documents",
        "",
    ]

    if not results:
        source_lines.append("- No documents scraped yet.")
    else:
        for item in sorted(results, key=lambda row: str(row["source_url"])):
            tags = ", ".join(str(tag) for tag in item.get("topic_tags", []))
            source_lines.append(
                f"- {item['source_name']}: {item['title']} — {item['source_url']} — tags: {tags}"
            )

    README_PATH.write_text("\n".join(source_lines) + "\n", encoding="utf-8")


def write_source_location_plot(results: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("source-location plot skipped: matplotlib is not installed")
        return

    counts = Counter(str(item.get("jurisdiction", "unknown")) for item in results)
    if not counts:
        print("source-location plot skipped: no scraped documents")
        return

    labels = [f"{name} ({count})" for name, count in sorted(counts.items())]
    sizes = [count for _, count in sorted(counts.items())]

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    axis.set_title("Clinical Corpus Source Locations")
    axis.axis("equal")
    figure.tight_layout()
    figure.savefig(SOURCE_LOCATION_PLOT_PATH, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
