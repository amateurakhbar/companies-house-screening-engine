"""25 — Deep-dive recategorization of the healthcare platform shortlist.

Hand-built from reading each firm's actual website scrape (data/cache/scrapes/),
not the bulk vertical tags. Distinguishes genuine vertical specialists from
generalist MSPs and flags firms that aren't really IT / aren't really healthcare.

Adds four columns to msp_platform_ranked_healthcare.csv:
  recat_business_type  - what they actually do (core offering)
  recat_specialisation - true sector/tech focus (or 'multi_sector')
  recat_healthcare_fit - core | significant | partial | incidental | none
  recat_note           - one-line, evidence-based justification

Writes output/healthcare_platform_recategorized.csv (does not mutate inputs).
"""
from __future__ import annotations
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
INP = ROOT / "output" / "msp_platform_ranked_healthcare.csv"
OUT = ROOT / "output" / "healthcare_platform_recategorized.csv"

# CompanyNumber -> (business_type, specialisation, healthcare_fit, note)
CAT = {
    "13158806": ("specialist_it_hardware", "dental", "core",
                 "Dental digital-imaging IT & maintenance (CQC, manufacturer-certified)."),
    "SC167135": ("niche_managed_service", "tech_specialist_hpe_nonstop", "incidental",
                 "HPE NonStop systems lifecycle specialist; sector-agnostic, healthcare just a client type."),
    "08919564": ("msp_generalist", "regulated_smes", "partial",
                 "Generalist MSP (support/dev/cyber) for regulated SMEs; some care-sector clients."),
    "05112055": ("msp_vertical", "public_sector_nhs", "significant",
                 "Public-sector/NHS MSP; HSCN-connected, NHS DSP Toolkit published."),
    "01859500": ("specialist_it_hardware", "healthcare_nursecall", "core",
                 "NurseCall systems design/install/maintain for NHS, hospitals, care homes."),
    "05785120": ("msp_vertical", "dental", "core",
                 "Dental-only MSP ('we do IT for dental practices'); built own monitoring + VoIP."),
    "05006536": ("msp_vertical", "hospitality", "none",
                 "Hospitality & leisure MSP (hotels/resorts), global; NOT healthcare (mistagged)."),
    "06957582": ("iot_monitoring_product", "leisure_agri_utilities", "none",
                 "Remote asset monitoring/control (IoT) for leisure/agri/energy/water; not healthcare, not classic IT."),
    "05145648": ("av_integration", "education", "incidental",
                 "AV integrator for education/business/blue-light; not an MSP."),
    "07398559": ("msp_generalist", "microsoft_multi_sector", "partial",
                 "Microsoft-focused MSP; some primary-care clients (Bromley GPA)."),
    "09819394": ("msp_vertical", "mental_health_care", "core",
                 "Mental-healthcare & care-home IT specialist (CAMHS, secure MH, learning disability)."),
    "09807084": ("msp_vertical", "finance_us", "none",
                 "US (Florida) IT firm for community banks/credit unions; not healthcare, not UK."),
    "05537445": ("ict_infrastructure", "multi_sector", "partial",
                 "Generalist ICT infrastructure/cabling/AV; NHS one of several client sectors (the VIETEC case)."),
    "08671960": ("msp_vertical", "care_homes", "core",
                 "Residential care-home IT specialist."),
    "07788868": ("msp_vertical", "education_public_nhs", "significant",
                 "Education/public-sector/NHS MSP (schools, MATs, NHS departments)."),
    "10590426": ("msp_vertical", "legal", "none",
                 "Legal-focused MSP despite 'hospitality' name (law-firm IT, HIPAA/GDPR); not healthcare."),
    "12030414": ("msp_vertical", "health_social_care", "core",
                 "Health & social-care IT 'service assurance' platform; NHS DSPT/ISO27001 aligned."),
    "06619540": ("msp_generalist", "regulated_industries", "incidental",
                 "Generalist MSP for regulated industries (Northern Powerhouse); healthcare just listed."),
    "SC334935": ("msp_generalist", "sme_multi_sector", "incidental",
                 "Generalist SME MSP/consultancy (Edinburgh, B-Corp); healthcare one of many sectors."),
    "03609853": ("specialist_it_hardware", "critical_power", "incidental",
                 "Emergency backup-power specialist (UPS/generators); not IT/MSP, healthcare incidental."),
    "08352622": ("msp_vertical", "dental", "core",
                 "Dental-only MSP, nationwide."),
    "04830832": ("field_services", "retail_hospitality_estates", "incidental",
                 "EPOS & IT install/support across retail/hospitality estates; healthcare a minor brand set."),
    "08735028": ("msp_vertical", "dental", "core",
                 "Dental IT specialist (private & NHS practices), 30 yrs."),
    "07161527": ("ict_infrastructure", "multi_sector", "partial",
                 "Generalist ICT infrastructure/cabling/AV/CCTV (NE England); health one of several sectors."),
    "03955112": ("av_iptv_integration", "hospitality_av", "incidental",
                 "Satellite/IPTV/AV/digital-signage integrator; hospitals a client, not an MSP."),
    "07646622": ("msp_generalist", "multi_sector", "incidental",
                 "Budget generalist MSP claiming multi-vertical; US datacentre; healthcare incidental."),
    "06212667": ("non_it_facilities", "ventilation_hygiene", "significant",
                 "Ventilation/ductwork cleaning & chlorination (TR19) for NHS estates — FACILITIES, not IT. Mis-scoped."),
    "07693142": ("msp_generalist", "education_multi_sector", "incidental",
                 "Generalist MSP + web/GDPR for schools/SMEs/charities/retail/health/public sector."),
    "06746875": ("msp_generalist", "leisure_multi_sector", "incidental",
                 "Generalist managed-IT/cloud/security; flagship casino/leisure client; not healthcare."),
    "07251970": ("msp_generalist", "sme_multi_sector", "incidental",
                 "Generalist SME MSP; healthcare only in founders' background."),
    "06302931": ("msp_generalist", "regulated_us", "partial",
                 "US-leaning IT/cyber MSP (Pro Cloud SaaS) for regulated industries; some senior-living."),
    "10777312": ("specialist_consultancy", "tech_specialist_euc_vdi", "incidental",
                 "End-user-computing/VDI (Citrix/VMware) specialist; multi-sector, healthcare incidental."),
    "10289770": ("msp_generalist", "multi_sector", "incidental",
                 "Broad multi-sector generalist MSP (logistics/manufacturing/retail/education/public); healthcare incidental."),
    "06592263": ("msp_vertical", "professional_services", "incidental",
                 "Professional-services MSP (law firms, accountants, advisors); HIPAA/NIST noted but not healthcare-focused."),
    "08971725": ("cloud_hosting", "sovereign_cloud", "incidental",
                 "UK-sovereign cloud/IaaS (liquid-immersion cooling), 1,600+ customers; not an MSP, not healthcare-specific."),
    "09742730": ("msp_vertical", "hospitality", "none",
                 "Hospitality-only MSP (bars/restaurants/hotels); mistagged healthcare."),
    "09424762": ("msp_generalist", "sme_multi_sector", "incidental",
                 "Regional (NW England) generalist SME MSP."),
    "08825944": ("msp_vertical", "charities_nonprofit", "none",
                 "Charity/non-profit IT specialist; not healthcare."),
    "12231678": ("software_product", "hospitality", "incidental",
                 "Hospitality guest-tech product vendor (IPTV/PMS/cloud), global; mistagged healthcare."),
    "09536071": ("msp_vertical", "finance_legal", "incidental",
                 "Finance/legal MSP (insurance, accountancy, legal, banking); not healthcare."),
    "12212084": ("msp_vertical", "education", "incidental",
                 "Education-specialist MSP (schools/MATs); occasional NHS CSU project."),
    "13184326": ("specialist_it_hardware", "dental", "core",
                 "Dental equipment + IT supplier (Dentsply Sirona / Cerec specialist)."),
    "11859968": ("msp_vertical", "critical_infra_ot", "none",
                 "Critical-infrastructure IT/OT MSP (police/energy/finance); not healthcare."),
    "06775290": ("msp_generalist", "sme_multi_sector", "incidental",
                 "London generalist SME MSP (travel/retail/finance/professional)."),
    "03270278": ("telecoms_voip", "multi_sector", "incidental",
                 "Telecoms/voice & data platform specialist; multi-sector (health one of many)."),
    "06155295": ("msp_vertical", "primary_care_gp", "core",
                 "Primary-care/GP-practice IT solutions & staff training specialist."),
    "07883547": ("msp_generalist", "small_business_professional", "partial",
                 "Small-business generalist MSP (solicitors, dental, veterinary, funeral, accounting, farms)."),
}

FIT_ORDER = {"core": 0, "significant": 1, "partial": 2, "incidental": 3, "none": 4}


def main():
    df = pd.read_csv(INP, dtype={"CompanyNumber": str})
    miss = [c for c in df["CompanyNumber"] if c not in CAT]
    if miss:
        print("WARN: no categorization for:", miss)
    df["recat_business_type"] = df["CompanyNumber"].map(lambda c: CAT.get(c, ("", "", "", ""))[0])
    df["recat_specialisation"] = df["CompanyNumber"].map(lambda c: CAT.get(c, ("", "", "", ""))[1])
    df["recat_healthcare_fit"] = df["CompanyNumber"].map(lambda c: CAT.get(c, ("", "", "", ""))[2])
    df["recat_note"] = df["CompanyNumber"].map(lambda c: CAT.get(c, ("", "", "", ""))[3])
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(df)} firms, +4 recat columns\n")

    print("=== healthcare_fit distribution ===")
    print(df["recat_healthcare_fit"].value_counts().reindex(FIT_ORDER.keys()).to_string())
    print("\n=== business_type distribution ===")
    print(df["recat_business_type"].value_counts().to_string())
    print("\n=== GENUINE healthcare specialists (fit=core) ===")
    core = df[df["recat_healthcare_fit"] == "core"]
    for r in core.itertuples():
        print(f"  {r.CompanyName[:34]:34} {r.recat_specialisation:20} {r.recat_note}")
    print("\n=== NOT healthcare at all (fit=none) — remove from a healthcare cut ===")
    for r in df[df["recat_healthcare_fit"] == "none"].itertuples():
        print(f"  {r.CompanyName[:34]:34} {r.recat_specialisation:20} {r.recat_note}")
    print("\n=== NOT EVEN IT (review scope) ===")
    for r in df[df["recat_business_type"].isin(["non_it_facilities", "iot_monitoring_product"])].itertuples():
        print(f"  {r.CompanyName[:34]:34} {r.recat_business_type:22} {r.recat_note}")


if __name__ == "__main__":
    main()
