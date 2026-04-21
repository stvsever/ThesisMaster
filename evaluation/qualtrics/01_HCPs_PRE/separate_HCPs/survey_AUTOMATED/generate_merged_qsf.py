#!/usr/bin/env python3
"""
generate_merged_qsf.py
Generates ONE merged QSF containing all 10 PHOENIX PRE cases.
Qualtrics RandomizationV2 (SubSet=1, EvenPresentation=true) randomly assigns
each respondent exactly one case. Import once, share a single link with all HCPs.

Output: generated/qsf_files/PHOENIX_PRE_MERGED_ALL10.qsf

Usage:
    python3 generate_merged_qsf.py
"""
from __future__ import annotations

import json
from pathlib import Path

from generate_qsf import (
    # data + constants
    CaseSurvey, SCALE_7PT, HCP_DIRS,
    # ID
    gen_id,
    # parsing
    parse_case_survey, extract_between, extract_first, parse_network, network_to_b64,
    # HTML
    p, b, i, hr, h2, ul, esc, plain,
    blue_box, gray_box, green_box, amber_box, purple_box, fig_html, meta_table,
    # page builders (shared / generic)
    page_instr_p1, page_instr_p2,
    page_d1_instr, page_d1_example, page_d1_case,
    page_d2_instr, page_d2_example, page_d2_case,
    page_d3_instr, page_d3_example, page_d3_case,
    page_d4_instr, page_d4_example, page_d4_case,
    page_d5_instr, page_d5_example, page_d5_case,
    page_d6_context, q_d6_a, q_d6_b, q_d6_c,
    # QSFBuilder base
    QSFBuilder,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "generated" / "qsf_files"


# ─────────────────────────────────────────────────────────────────────────────
# GENERIC PAGES (no HCP-specific code)
# ─────────────────────────────────────────────────────────────────────────────

def page_consent_generic() -> str:
    return (
        h2("Geïnformeerde toestemming")
        + blue_box("",
            p("U staat op het punt de PHOENIX evaluatiesurvey te starten.")
            + p("Lees onderstaande informatie zorgvuldig door voordat u verdergaat.")
        )
        + gray_box("Doel van het onderzoek",
            p("Deze survey verzamelt onafhankelijke expertantwoorden van zorgprofessionals "
              "op dezelfde klinische redeneerstappen die het PHOENIX-systeem uitvoert. "
              "Uw antwoorden vormen het menselijk referentiecorpus voor een "
              "<strong>dubbelblinde vergelijking</strong> met systeemoutput in het kader van "
              "een masterproef aan de Universiteit Gent (promotoren: Prof.&nbsp;Dr.&nbsp;Geert "
              "Crombez; Dr.&nbsp;Annick De Paepe).")
        )
        + gray_box("Wat deelname inhoudt",
            ul([
                "U vult eerst een korte <strong>intake</strong> in (3 vragen: achtergrond, "
                "houding tegenover agentic AI, en kennis van netwerkanalyse).",
                "Vervolgens krijgt u een willekeurig toegewezen <strong>casus</strong> met "
                "6 inhoudelijke delen (symptoomidentificatie, behandelingsopties, prioritering, "
                "EMA-itemselectie, coachingsboodschap, reflectie).",
                "De survey duurt <strong>ongeveer 25–35 minuten</strong> in totaal.",
                "We vragen uitsluitend uw <strong>eigen klinisch oordeel</strong> — "
                "<em>geen generatieve AI, collegaoverleg of andere externe hulpmiddelen</em>.",
            ])
        )
        + green_box("Vrijwilligheid en vertrouwelijkheid",
            ul([
                "Deelname is <strong>volledig vrijwillig</strong>.",
                "U kunt op <strong>elk moment stoppen</strong> door het venster te sluiten; "
                "uw gedeeltelijke antwoorden worden dan niet verwerkt.",
                "Uw antwoorden worden na ontvangst <strong>geanonimiseerd</strong> en "
                "uitsluitend voor deze masterproefanalyse gebruikt.",
                "De resultaten worden gerapporteerd op groepsniveau; individuele antwoorden "
                "zijn niet herleidbaar naar uw persoon.",
            ])
        )
        + amber_box("Contact",
            meta_table([
                ("Onderzoeker", "Stijn Van Severen"),
                ("E-mail", "stijn.vanseveren@ugent.be"),
                ("Instelling", "Universiteit Gent, Fac. Psychologie en Pedagogische Wetenschappen"),
            ])
        )
        + p("<em>Kies hieronder of u op basis van bovenstaande informatie wenst deel te nemen. "
            "Indien u <strong>niet</strong> wenst deel te nemen, kiest u optie 2 — "
            "de survey sluit dan automatisch af.</em>")
    )


def page_title_case(case: CaseSurvey) -> str:
    return (
        f'<div style="background:#1E3A5F;border-radius:8px;padding:22px 28px;'
        f'margin-bottom:18px;text-align:center;">'
        f'<div style="font-size:24px;font-weight:bold;color:white;margin-bottom:8px;">'
        f'PHOENIX evaluatiestudie</div>'
        f'<div style="font-size:16px;color:#BFDBFE;margin-bottom:6px;">'
        f'Fase 1 &#8212; onafhankelijke expertgeneratie</div>'
        f'<div style="font-size:13px;color:white;font-style:italic;">'
        f'Instrument voor Zorgprofessionals</div></div>'
        + f'<div style="background:#DBEAFE;border:2px solid #1D4ED8;border-radius:6px;'
        f'padding:14px 20px;text-align:center;margin-bottom:14px;">'
        f'<div style="font-size:16px;font-weight:bold;color:#1E3A5F;">'
        f'Uw toegewezen casus:&nbsp;'
        f'<strong style="font-family:monospace;color:#1D4ED8;">{case.case_code}</strong></div>'
        f'<div style="font-size:13px;color:#475569;margin-top:6px;">'
        f'{esc(case.profile)}&nbsp;&bull;&nbsp;Duur: {esc(case.duration)}</div></div>'
        + gray_box("Studiegegevens", meta_table([
            ("Studie", "Evaluatie van de klinische kwaliteit van een ontologiegebaseerd "
                       "multi-agentsysteem voor gepersonaliseerde digitale geestelijke "
                       "gezondheidszorg (PHOENIX)"),
            ("Instelling", "Universiteit Gent — Faculteit Psychologie en Pedagogische Wetenschappen"),
            ("Onderzoeker", "Stijn Van Severen (masterproefstudent)"),
            ("Promotoren", "Prof. Dr. Geert Crombez; Dr. Annick De Paepe"),
            ("Contact", "stijn.vanseveren@ugent.be"),
            ("Geschatte duur", "Ongeveer 25–35 minuten (inclusief instructies en Deel 6)"),
        ]))
        + gray_box("Werkwijze",
            p("Werk <strong>sequentieel</strong> van Deel 1 naar Deel 6. "
              "Gebruik <u>geen</u> generatieve AI, richtlijnen of collegaoverleg. "
              "Uw antwoorden weerspiegelen uw eigen klinische oordeelsvorming.")
        )
        + amber_box("Vertrouwelijkheid",
            "Uw antwoorden worden geanonimiseerd en uitsluitend voor deze "
            "masterproefstudie gebruikt. Deelname is <strong>volledig vrijwillig</strong>."
        )
    )


def page_closing_generic() -> str:
    return (
        h2("Afronding en terugbezorging")
        + p("Dank u voor het invullen van alle zes delen voor uw toegewezen casus "
            "en de afsluitende reflectievragen.")
        + gray_box("Checklist voor terugbezorging",
            ul([
                "&#9744;&nbsp; Ik heb in Deel 1 symptoomlabels ingevuld.",
                "&#9744;&nbsp; Ik heb in Deel 2 behandelingsoptielabels ingevuld.",
                "&#9744;&nbsp; Ik heb in Deel 3 alle 5 behandelingsopties gerangschikt.",
                "&#9744;&nbsp; Ik heb in Deel 4 exact 6 EMA-items geselecteerd.",
                "&#9744;&nbsp; Ik heb in Deel 5 een mobiele coachingsboodschap geschreven.",
                "&#9744;&nbsp; Ik heb in Deel 6 de drie reflectievragen beantwoord.",
                "&#9744;&nbsp; Mijn antwoorden weerspiegelen mijn eigen klinische "
                "oordeelsvorming zonder gebruik van generatieve AI of andere externe hulp.",
                "&#9744;&nbsp; Ik begrijp dat mijn antwoorden geanonimiseerd worden "
                "voor analyse.",
            ])
        )
        + blue_box("Uw antwoorden zijn opgeslagen",
            p("Na het indienen van deze survey worden uw antwoorden automatisch opgeslagen. "
              "U hoeft niets terug te mailen.")
            + p("Na ontvangst worden uw antwoorden geanonimiseerd en opgenomen in het "
                "expertreferentiecorpus voor de latere blinde evaluatie van PHOENIX.")
            + p(b("Hartelijk dank voor uw waardevolle bijdrage aan dit onderzoek."))
            + p("Vragen of opmerkingen? Neem contact op via: "
                "<strong>stijn.vanseveren@ugent.be</strong>")
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# MERGED QSF BUILDER  (RandomizationV2 survey flow)
# ─────────────────────────────────────────────────────────────────────────────

class MergedQSFBuilder(QSFBuilder):
    """Extends QSFBuilder with RandomizationV2 flow support."""

    def __init__(self, survey_id: str, survey_name: str):
        super().__init__(survey_id, survey_name)
        self._phase: str = "before"
        self._shared_before: list[str] = []
        self._case_groups: list[tuple[str, list[str]]] = []
        self._cur_case_blks: list[str] | None = None
        self._shared_after: list[str] = []

    def start_case(self, case_code: str) -> "MergedQSFBuilder":
        self._phase = "case"
        self._cur_case_blks = []
        self._case_groups.append((case_code, self._cur_case_blks))
        return self

    def end_cases(self) -> "MergedQSFBuilder":
        self._phase = "after"
        self._cur_case_blks = None
        return self

    def block(self, name: str) -> "MergedQSFBuilder":
        super().block(name)
        blk_id = self._cur["id"]
        if self._phase == "before":
            self._shared_before.append(blk_id)
        elif self._phase == "case" and self._cur_case_blks is not None:
            self._cur_case_blks.append(blk_id)
        else:
            self._shared_after.append(blk_id)
        return self

    def build(self) -> dict:
        rs_id = gen_id("RS_", self.survey_id)

        block_payload = {
            str(idx): {
                "Type": "Default", "SubType": "", "Description": blk["name"],
                "ID": blk["id"], "BlockElements": blk["elements"],
                "Options": {"BlockLocking": "false", "RandomizeQuestions": "false",
                            "BlockVisibility": "Expanded"},
            }
            for idx, blk in enumerate(self.blocks)
        }

        # ── Build flow ────────────────────────────────────────────────────────
        fl = [0]
        def nfl() -> str:
            fl[0] += 1
            return f"FL_{fl[0]}"

        def ed_node(field: str, value: str) -> dict:
            return {
                "Type": "EmbeddedData",
                "FlowID": nfl(),
                "EmbeddedData": [{
                    "Description": field,
                    "Type": "Custom",
                    "Field": field,
                    "VariableType": "String",
                    "Value": value,
                    "AnalyzeText": False,
                    "DataVisibility": [],
                }],
            }

        flow: list[dict] = []

        # Declare CaseAssigned as embeddeddata field (also reads URL param
        # e.g. ?CaseAssigned=C01 for manual guaranteed assignment)
        flow.append(ed_node("CaseAssigned", ""))

        for bid in self._shared_before:
            flow.append({"Type": "Block", "ID": bid, "FlowID": nfl()})

        # RandomizationV2: EvenPresentation ensures round-robin (each of 10
        # cases is shown once before any is repeated — with 10 HCPs this
        # guarantees a distinct case per HCP)
        rand_groups: list[dict] = []
        for code, blk_ids in self._case_groups:
            grp_fl_id = nfl()
            # EmbeddedData node inside each group records which case was assigned
            grp_flow: list[dict] = [ed_node("CaseAssigned", code)]
            grp_flow += [{"Type": "Block", "ID": bid, "FlowID": nfl()}
                         for bid in blk_ids]
            rand_groups.append({
                "Type": "Group",
                "FlowID": grp_fl_id,
                "Description": f"Casus {code}",
                "Flow": grp_flow,
            })

        flow.append({
            "Type": "RandomizationV2",
            "FlowID": nfl(),
            "EvenPresentation": True,
            "SubSet": 1,
            "Flow": rand_groups,
        })

        for bid in self._shared_after:
            flow.append({"Type": "Block", "ID": bid, "FlowID": nfl()})

        flow.append({"Type": "EndSurvey", "FlowID": nfl()})

        elements: list[dict] = [
            {"SurveyID": self.survey_id, "Element": "BL",
             "PrimaryAttribute": "Survey Blocks",
             "SecondaryAttribute": None, "TertiaryAttribute": None,
             "Payload": block_payload},
            {"SurveyID": self.survey_id, "Element": "FL",
             "PrimaryAttribute": "Survey Flow",
             "SecondaryAttribute": None, "TertiaryAttribute": None,
             "Payload": {"Type": "Root", "FlowID": "FL_0",
                         "Flow": flow, "Properties": {"Count": len(flow)}}},
            {"SurveyID": self.survey_id, "Element": "SO",
             "PrimaryAttribute": "Survey Options",
             "SecondaryAttribute": "Default Question Block",
             "TertiaryAttribute": None,
             "Payload": {
                 "BackButton": "false", "SaveAndContinue": "true",
                 "SurveyProtection": "PublicSurvey",
                 "BallotBoxStuffingPrevention": "false",
                 "NoIndex": "No", "SecureResponseFiles": "true",
                 "SurveyExpiration": "None",
                 "SurveyTermination": "DefaultMessage",
                 "Header": "", "Footer": "",
                 "ProgressBarDisplay": "WithText", "PartialData": "+7 days",
                 "ValidationMessage": "", "InactiveSurvey": "DefaultMessage",
                 "AvailableLanguages": {"NL": "Dutch"}, "Language": "NL",
                 "CustomStyles": "", "HeaderMid": "", "FooterMid": ""}},
            {"SurveyID": self.survey_id, "Element": "PROJ",
             "PrimaryAttribute": "Survey Project",
             "SecondaryAttribute": None, "TertiaryAttribute": None,
             "Payload": {"ProjectCategory": "CORE", "SchemaVersion": "1.1.0"}},
            {"SurveyID": self.survey_id, "Element": "RS",
             "PrimaryAttribute": rs_id,
             "SecondaryAttribute": "Default Response Set",
             "TertiaryAttribute": None,
             "Payload": {"ID": rs_id, "Name": "Default Response Set",
                         "IsDefault": True,
                         "CreationDate": "2026-04-21 10:00:00",
                         "LastModifiedDate": "2026-04-21 10:00:00"}},
        ]
        elements.extend(self.questions)

        return {
            "SurveyEntry": {
                "SurveyID": self.survey_id, "SurveyName": self.survey_name,
                "SurveyDescription": None,
                "SurveyOwnerID": "UR_00000000000000000",
                "SurveyBrandID": "ugent", "DivisionID": None,
                "SurveyLanguage": "NL",
                "SurveyActiveResponseSet": rs_id,
                "SurveyStatus": "Inactive",
                "SurveyStartDate": "0000-00-00 00:00:00",
                "SurveyExpirationDate": "0000-00-00 00:00:00",
                "SurveyCreationDate": "2026-04-21 10:00:00",
                "CreatorID": "UR_00000000000000000",
                "LastModified": "2026-04-21 10:00:00",
                "LastAccessed": "0000-00-00 00:00:00",
                "LastActivated": "0000-00-00 00:00:00",
                "Deleted": None,
            },
            "SurveyElements": elements,
        }


# ─────────────────────────────────────────────────────────────────────────────
# FORCE-RESPONSE POST-PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

def apply_force_response(qsf: dict) -> dict:
    """Set ForceResponse=ON on every interactive question.
    Exceptions:
      - DB  (descriptive text blocks — no response field)
      - FORM (te_form symptom/treatment labels — variable number of fields;
              HCPs are instructed to leave unused slots blank)
    """
    for el in qsf["SurveyElements"]:
        if el["Element"] != "SQ":
            continue
        pl = el["Payload"]
        if pl["Type"] == "DB":
            continue
        if pl.get("Selector") == "FORM":
            continue
        v = pl.setdefault("Validation", {}).setdefault("Settings", {})
        v["ForceResponse"] = "ON"
        v["ForceResponseType"] = "ON"
    return qsf


# ─────────────────────────────────────────────────────────────────────────────
# SURVEY BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_merged_survey(cases: list[CaseSurvey], example_b64: str) -> dict:
    survey_id = gen_id("SV_", "PHOENIX_PRE_MERGED_ALL10")
    q = MergedQSFBuilder(survey_id,
                         "PHOENIX Evaluatiestudie — PRE fase (alle casussen, willekeurige toewijzing)")

    # ── Block 1: Toestemming ──────────────────────────────────────────────────
    q.block("Toestemming")
    q.db(page_consent_generic(), tag="TOESTEMMING_INFO")
    q.mc_single(
        p(b("Toestemmingsverklaring") + "<br>"
          "Ik heb de bovenstaande informatie gelezen en ga akkoord met deelname "
          "aan dit onderzoek."),
        choices=["Ja, ik stem toe en wil deelnemen.",
                 "Nee, ik stem niet toe en wil niet deelnemen."],
        tag="CONSENT",
        skip_end_if=2,
    )

    # ── Block 2: Instructiepagina ─────────────────────────────────────────────
    q.block("Instructiepagina")
    q.db(page_instr_p1(), tag="INSTR_P1")
    q.page_break()
    q.db(page_instr_p2(), tag="INSTR_P2")

    # ── Block 3: Intake ───────────────────────────────────────────────────────
    q.block("Intake")
    q.db(
        h2("Intake — drie oriënterende vragen")
        + gray_box("Toelichting",
            p("Voorafgaand aan de casusopgave stellen we u drie korte oriënterende vragen. "
              "Deze dienen uitsluitend om de professionele context van uw antwoorden te "
              "kunnen duiden bij de latere analyse — ze zijn niet evaluatief.")
            + ul([
                b("1. Algemene achtergrond en klinische ervaring:") + " om de professionele "
                + "context van uw antwoorden te kunnen situeren.",
                b("2. Houding tegenover agentic AI in de klinische context:") + " om "
                + "eventuele verwachtingsbias bij de latere blinde scoring in kaart te brengen.",
                b("3. Kennis van netwerk-analytisch denken:") + " omdat de casusopgaven "
                + "vertrekken vanuit een netwerk-analytisch redeneermodel.",
            ])
        ),
        tag="INTAKE_INTRO",
    )
    q.te_essay(
        p(b("Vraag 1 — Professionele achtergrond en klinische ervaring") + "<br><br>"
          "Beschrijf kort uw professionele achtergrond. Geef daarbij aan: "
          "(1) uw discipline en beroepstitel, "
          "(2) uw huidige functie of werksetting, "
          "(3) het aantal jaren professionele ervaring, en "
          "(4) de mate van direct contact met personen met psychische klachten."),
        tag="INTAKE_ACHTERGROND",
        force=False,
    )
    q.page_break()
    q.matrix(
        p(b("Vraag 2 — Houding tegenover agentic AI in de klinische context") + "<br><br>"
          "Geef aan in welke mate u akkoord gaat met onderstaande uitspraken. "
          "Gebruik de 7-punt schaal van <em>Helemaal oneens</em> tot "
          "<em>Helemaal eens</em>. Er zijn geen goede of foute antwoorden."),
        statements=[
            "Ik zie agentic AI als een potentieel zinvolle ondersteuning voor klinische "
            "besluitvorming in mentale gezondheidszorg.",
            "Ik zou agentic AI enkel vertrouwen wanneer de onderliggende redeneerstappen "
            "expliciet en controleerbaar worden getoond.",
            "Ik verwacht dat agentic AI binnen afzienbare tijd bruikbaar kan zijn voor "
            "gepersonaliseerde digitale geestelijke gezondheidszorg.",
        ],
        scale=SCALE_7PT,
        tag="INTAKE_AGENTIC_AI",
    )
    q.page_break()
    q.matrix(
        p(b("Vraag 3 — Kennis van en houding tegenover netwerk-analytisch denken") + "<br><br>"
          "Geef aan in welke mate u akkoord gaat met onderstaande uitspraken. "
          "Gebruik de 7-punt schaal van <em>Helemaal oneens</em> tot "
          "<em>Helemaal eens</em>."),
        statements=[
            "Ik ben voldoende vertrouwd met het netwerk-analytisch denken om de logica "
            "van de huidige surveyopgaven goed te kunnen volgen.",
            "Het in kaart brengen van dynamische relaties tussen klachten en modificeerbaar "
            "gedrag kan klinisch relevante behandelprioriteiten zichtbaar maken.",
            "Ik vind een netwerk-analytische benadering methodologisch bruikbaar als basis "
            "voor gepersonaliseerde EMA-monitoring en digitale interventies.",
        ],
        scale=SCALE_7PT,
        tag="INTAKE_NETWORK_ANALYSIS",
    )
    q.page_break()

    # ── Shared D3 instructies & voorbeeld (rendered once, not 10×) ────────────
    q.block("Deel 3 — Instructies & Voorbeeld")
    q.db(page_d3_instr(), tag="D3_INSTR_SHARED")
    q.page_break()
    q.db(page_d3_example(example_b64), tag="D3_VOORBEELD_SHARED")
    q.page_break()

    # ── Per-case blocks (10 cases × 6 blocks each = 60 blocks) ───────────────
    for case in cases:
        case_b64 = network_to_b64(case.network, dpi=96)
        cc = case.case_code
        q.start_case(cc)

        # ── Titelpagina ───────────────────────────────────────────────────────
        q.block(f"Titelpagina — {cc}")
        q.db(page_title_case(case), tag=f"{cc}_TITEL")

        # ── Deel 1 ────────────────────────────────────────────────────────────
        q.block(f"Deel 1 — {cc}")
        q.db(page_d1_instr(), tag=f"{cc}_D1_INSTR")
        q.page_break()
        q.db(page_d1_example(), tag=f"{cc}_D1_VOORBEELD")
        q.page_break()
        q.db(page_d1_case(case), tag=f"{cc}_D1_CASUS")
        q.te_form(
            p(b(f"Uw antwoord — Deel 1 ({cc})") + "<br>"
              "Noteer <strong>2–6 symptoomlabels</strong> (2–5 woorden per label; "
              "laat ongebruikte velden leeg)."),
            fields=[f"Symptoom {n}" for n in range(1, 7)],
            tag=f"{cc}_D1_ANTWOORD",
        )

        # ── Deel 2 ────────────────────────────────────────────────────────────
        q.block(f"Deel 2 — {cc}")
        q.db(page_d2_instr(), tag=f"{cc}_D2_INSTR")
        q.page_break()
        q.db(page_d2_example(), tag=f"{cc}_D2_VOORBEELD")
        q.page_break()
        q.db(page_d2_case(case), tag=f"{cc}_D2_CASUS")
        q.te_form(
            p(b(f"Uw antwoord — Deel 2 ({cc})") + "<br>"
              "Noteer <strong>3–5 behandelingsoptielabels</strong> (2–6 woorden per label; "
              "laat ongebruikte velden leeg)."),
            fields=[f"Behandelingsoptie {n}" for n in range(1, 6)],
            tag=f"{cc}_D2_ANTWOORD",
        )

        # ── Deel 3 ────────────────────────────────────────────────────────────
        q.block(f"Deel 3 — {cc}")
        q.db(page_d3_case(case, case_b64), tag=f"{cc}_D3_CASUS")
        q.rank_order(
            p(b(f"Uw antwoord — Deel 3 ({cc})") + "<br>"
              + "Rangschik alle 5 behandelingsopties van hoogste (positie 1) naar laagste "
              + "(positie 5) behandelprioriteit.<br>"
              + i("Sleep de opties in de gewenste volgorde, "
                  "of typ rangorde 1–5 in de velden.")),
            choices=case.part3_options,
            tag=f"{cc}_D3_ANTWOORD",
        )

        # ── Deel 4 ────────────────────────────────────────────────────────────
        q.block(f"Deel 4 — {cc}")
        q.db(page_d4_instr(), tag=f"{cc}_D4_INSTR")
        q.page_break()
        q.db(page_d4_example(), tag=f"{cc}_D4_VOORBEELD")
        q.page_break()
        q.db(page_d4_case(case), tag=f"{cc}_D4_CASUS")
        q.mc_multi(
            p(b(f"Uw antwoord — Deel 4 ({cc})") + "<br>"
              "Selecteer <strong>exact 6</strong> EMA-items (2 per behandeldoel). "
              "<em>Geen symptomen</em> — alle items zijn modificeerbare gedragingen. "
              "Selecteer 2 sub-behandelingsopties per behandeldoel (6 in totaal)."),
            choices=[f"{n}.&nbsp; {item}"
                     for n, item in enumerate(case.part4_items, start=1)],
            tag=f"{cc}_D4_ANTWOORD",
            min_c=6, max_c=6,
        )

        # ── Deel 5 ────────────────────────────────────────────────────────────
        q.block(f"Deel 5 — {cc}")
        q.db(page_d5_instr(), tag=f"{cc}_D5_INSTR")
        q.page_break()
        q.db(page_d5_example(), tag=f"{cc}_D5_VOORBEELD")
        q.page_break()
        q.db(page_d5_case(case), tag=f"{cc}_D5_CASUS")
        q.te_essay(
            p(b(f"Uw antwoord — Deel 5 ({cc})") + "<br>"
              "Schrijf hieronder de mobiele coachingsboodschap voor deze casus. "
              "Formuleer alsof de tekst morgen rechtstreeks op de smartphone van de "
              "persoon verschijnt (2–4 zinnen, warm, direct, concreet)."),
            tag=f"{cc}_D5_ANTWOORD",
        )

        # ── Deel 6 ────────────────────────────────────────────────────────────
        q.block(f"Deel 6 — {cc}")
        q.db(page_d6_context(), tag=f"{cc}_D6_CONTEXT")
        q.page_break()
        q.te_essay(q_d6_a(case), tag=f"{cc}_D6_A", force=False)
        q.page_break()
        q.te_essay(q_d6_b(case), tag=f"{cc}_D6_B", force=False)
        q.page_break()
        q.te_essay(q_d6_c(case), tag=f"{cc}_D6_C", force=False)

    # ── Shared closing block ──────────────────────────────────────────────────
    q.end_cases()
    q.block("Afronding")
    q.db(page_closing_generic(), tag="AFRONDING")

    return apply_force_response(q.build())


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading shared template (HCP_01/main.tex) ...")
    template_tex = (HCP_DIRS[0] / "main.tex").read_text(encoding="utf-8")

    p3_shared = extract_between(
        template_tex,
        r"\\section\{Deel 3:.*?\}",
        r"\\subsection\*\{Casus 1:",
    )
    example_tikz = extract_first(
        r"\\begin\{voorbeeldbox\}\{Deel 3:.*?\\begin\{tikzpicture\}"
        r"(.*?)\\end\{tikzpicture\}.*?\\end\{voorbeeldbox\}",
        p3_shared,
    )
    example_network = parse_network(example_tikz, "Behandelingsopties", "Symptomen",
                                    "shared_example")
    print("Rendering shared example network figure ...")
    example_b64 = network_to_b64(example_network, dpi=96)

    print(f"\nParsing {len(HCP_DIRS)} case LaTeX files ...")
    cases: list[CaseSurvey] = []
    for hcp_dir in HCP_DIRS:
        tex = (hcp_dir / "main.tex").read_text(encoding="utf-8")
        case = parse_case_survey(tex, example_network)
        cases.append(case)
        print(f"  ✓ {case.case_code} — {case.profile}")

    print("\nBuilding merged QSF (all 10 cases, RandomizationV2) ...")
    for case in cases:
        print(f"  Rendering network for {case.case_code} ...")
    qsf_data = build_merged_survey(cases, example_b64)

    out = OUT_DIR / "PHOENIX_PRE_MERGED_ALL10.qsf"
    out.write_text(json.dumps(qsf_data, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    n_q = len([e for e in qsf_data["SurveyElements"] if e["Element"] == "SQ"])
    n_b = len(qsf_data["SurveyElements"][0]["Payload"])
    size_mb = out.stat().st_size / (1024 * 1024)

    print(f"\n✓  {out.name}")
    print(f"   Size: {size_mb:.1f} MB  |  Blocks: {n_b}  |  Questions: {n_q}")
    print(f"\nSurvey flow:")
    print("  EmbeddedData: CaseAssigned = '' (initialised; overwritten per group)")
    print("  [Shared] Toestemming  (consent + branch-to-end if refused)")
    print("  [Shared] Instructiepagina  (2 pages)")
    print("  [Shared] Intake  (background essay + AI Likert + network Likert)")
    print("  [Shared] Deel 3 — Instructies & Voorbeeld  (network example, shown once)")
    print("  [RandomizationV2  EvenPresentation=true  SubSet=1]")
    for case in cases:
        print(f"    Group {case.case_code}:  EmbeddedData CaseAssigned={case.case_code}"
              f"  +  6 blocks  (Titel, Deel 1–6)")
    print("  [Shared] Afronding")
    print()
    print("ForceResponse=ON on all interactive questions (MC, matrix, RO, essay).")
    print("TE/FORM (symptom/treatment labels) intentionally not forced (variable fields).")
    print("All data (text, matrices, rank order, checkboxes) saved automatically.")
    print()
    print("─── HOW TO IMPORT ───────────────────────────────────────────────────")
    print("  1. Qualtrics → Create new project → Survey → Import QSF")
    print("     → select PHOENIX_PRE_MERGED_ALL10.qsf")
    print()
    print("─── OPTION A: single link (automatic round-robin assignment) ────────")
    print("  EvenPresentation=true means Qualtrics cycles through all 10 cases")
    print("  before repeating any. With exactly 10 HCPs each case is used once.")
    print("  2a. Surveys → Activate → Distributions → Get anonymous link")
    print("  2b. Send that ONE link to all 10 HCPs.")
    print()
    print("─── OPTION B: individual links (guaranteed 1-to-1 assignment) ───────")
    print("  Append ?CaseAssigned=CXX to the anonymous link for each HCP.")
    print("  BUT: RandomizationV2 ignores URL params — for this approach you")
    print("  must switch to Branch logic in Qualtrics Survey Flow after import,")
    print("  OR simply trust Option A (EvenPresentation is sufficient for N=10).")
    print()
    print("─── OPTION C: quotas (hard limit 1 per case) ────────────────────────")
    print("  After import, in Qualtrics: Survey Flow → Quotas → add 10 quotas,")
    print("  one per case. Condition: CaseAssigned = CXX, Limit = 1.")
    print("  Quota action: exclude from randomizer when full.")
    print("  This gives the strictest guarantee: exactly 1 HCP per case.")
    print()
    print("  Data export column 'CaseAssigned' shows which case each HCP got.")
    print(f"\nFile: {out}")


if __name__ == "__main__":
    main()
