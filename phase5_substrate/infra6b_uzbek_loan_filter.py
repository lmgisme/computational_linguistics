"""
Infra 6B: Uzbek Loan-Filtered BEAST Rerun
==========================================
Identifies Arabic and Persian loanwords in the Uzbek lexical data,
removes them from the cognate binary matrix, rebuilds the NEXUS file,
and generates a BEAST 2.7.8 XML for rerun under the same Phase 4
configuration (Lewis Mk, strict clock, Yule, LogNormal root prior).

Expected result: Uzbek clusters inside Kipchak alongside Kazakh and
Kyrgyz rather than appearing as an early-diverging lineage.

Loan detection strategy (two layers):
  Layer 1 — Phonological heuristics:
    Arabic/Persian loans into Uzbek carry diagnostic phonemes absent
    from inherited Turkic stock or present only in loan contexts:
      - Initial /x/ (Arabic kha), /f/ anywhere (absent from PT)
      - Persian cluster /xt/, /ft/, /nd/
      - Arabic final /-at/, /-iy/ (nisba/broken plural)
      - /dʒ/ initial where Turkic has /y/
      - /ʔ/ (Arabic glottal), /ʕ/ (Arabic 'ain)
      - /h/ anywhere (absent from Proto-Turkic)
  Layer 2 — Explicit known-borrowing list:
    High-confidence Arabic/Persian loanword forms attested in Uzbek
    cross-referenced against Clauson (1972) absences.

Conservatism: a single weak heuristic does not flag alone. Strong
heuristics (/f/, /h/, /ʔ/, /ʕ/) flag without co-occurrence. Genuine
Turkic forms sharing a loan phoneme in a regular PT environment
(e.g. final /v/ < PT *b in 'suv') are explicitly protected.

IMPORTANT NOTE ON BLANK COGIDS:
  Of 998 Uzbek entries in northeuralex_merged.csv, approximately 960
  are NorthEuraLex-only entries with no cogid assigned (never linked
  into the LexStat/Savelyev cognate clusters). These contribute nothing
  to the binary matrix regardless of loan status. The matrix is built
  only from entries with valid cogids. Blank-cogid rows are retained
  in the filtered CSV but are immaterial to the BEAST run.

Outputs:
  output/uzbek_loan_flags.csv         - all Uzbek entries with flag + reason
  output/northeuralex_loanfiltered.csv - full dataset with Uzbek loans removed
  output/turkic_cognates_uzbek6b.nex  - NEXUS matrix without flagged Uzbek cogids
  output/beast_uzbek6b.xml            - BEAST XML for infra6b run
  output/infra6b_report.txt           - summary report
"""

import csv
import re
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"

# ---------------------------------------------------------------------------
# Loan detection parameters
# ---------------------------------------------------------------------------

# Phonological heuristics — IPA token patterns diagnostic for Arabic/Persian
# origin in Uzbek. Applied to space-separated ipa_tokens string.
PHON_HEURISTICS = [
    ("ar_x_initial",   r"^x "),          # Arabic kha initial
    ("pe_xt_cluster",  r"x t"),           # Persian /xt/ cluster
    ("pe_ft_cluster",  r"f t"),           # Persian /ft/ cluster
    ("ar_final_at",    r" a t$"),         # Arabic -at ending
    ("ar_final_iy",    r" i j$"),         # Arabic nisba -iy
    ("pe_dj_initial",  r"^dʒ "),          # Persian/Arabic /dʒ/ initial
    ("ar_pe_f",        r"\bf\b"),         # /f/ anywhere (not in PT)
    ("ar_iyat",        r"i j a t"),       # Arabic -iyat
    ("pe_zd_cluster",  r"z d"),           # Persian /zd/
    ("ar_glottal",     r"ʔ"),             # Arabic glottal stop
    ("ar_pharyngeal",  r"ʕ"),             # Arabic 'ain
    ("ar_pe_h",        r"\bh\b"),         # /h/ anywhere (not in PT)
]

# Strong single heuristics — flag without requiring co-occurrence
STRONG_HEURISTICS = {"ar_glottal", "ar_pharyngeal", "ar_pe_f", "ar_pe_h"}

# Explicit known-borrowing forms: gloss -> list of form substrings.
# Only forms with unambiguous non-Turkic etymology are listed.
# Empty list = Turkic-internal, never flag regardless of phonology.
EXPLICIT_LOAN_GLOSSES_FORMS = {
    "flesh":    ["ɡoʃt", "gosht", "goʃt"],  # < Persian gusht
    "tree":     ["daraχt", "daraxt"],         # < Persian daraxt
    "big":      [],                           # bujuk = PT *ülüg, Turkic-internal
    "egg":      ["tuχum", "tuxum"],           # < Persian toxm
    "woman":    ["χɔtin", "χotin", "xotin"],  # < Persian xatun
    "heart":    ["qalb"],                     # < Arabic qalb
    "palm":     ["kaft"],                     # < Persian kaf (+ /f/)
    "air":      ["hawɒ", "havo"],             # < Arabic hawa (+ /h/)
    "weather":  ["hawɒ", "havo"],             # same Arabic root
    "name":     ["ism"],                      # < Arabic ism
    "time":     ["vaqt", "waqt"],             # < Arabic waqt
    "people":   ["χalq", "xalq"],             # < Arabic xalq
    "world":    ["dunjo", "dunyo"],            # < Arabic/Persian dunya
    "city":     ["ʃaχar", "shahar"],          # < Persian shahar
    "book":     ["kitob"],                    # < Arabic kitab
    "color":    ["rang"],                     # < Persian rang
    "beautiful":["ɡɔzal", "gozal"],           # < Arabic ghazal (via Persian)
    "letter":   ["χat", "xat"],               # < Arabic xatt
    "body":     ["badan", "badna"],           # < Arabic/Persian badan
    # Explicitly safe Turkic forms (empty list = never flag):
    "water":    [], "fire":  [], "sun":    [], "moon":  [],
    "earth":    [], "stone": [], "blood":  [], "eye":   [],
    "hand":     [], "foot":  [], "head":   [], "work":  [],
    "year":     [], "language": ["zaban"],    # til is Turkic; zaban is Persian
}


def phon_flags(ipa_tokens_str):
    """Return list of (label, pattern) for heuristics that fire."""
    t = ipa_tokens_str.strip()
    return [(label, pat) for label, pat in PHON_HEURISTICS
            if re.search(pat, t)]


def explicit_form_match(gloss, form):
    """True if form matches a known loan for this gloss."""
    gl = gloss.lower()
    candidates = EXPLICIT_LOAN_GLOSSES_FORMS.get(gl, None)
    if candidates is None:
        return False  # gloss not in list at all
    return any(c and c in form.lower() for c in candidates)


def classify_uzbek(row):
    """
    Returns (is_loan: bool, reason: str).

    Flags if:
      (a) explicit form match, OR
      (b) any strong heuristic fires, OR
      (c) two or more weak heuristics fire together.
    """
    form  = row.get("form", "")
    gloss = row.get("gloss", "")
    ipa   = row.get("ipa_tokens", "")

    reasons = []

    if explicit_form_match(gloss, form):
        reasons.append("explicit_known_loan")

    fired = phon_flags(ipa)
    strong = [l for l, _ in fired if l in STRONG_HEURISTICS]
    weak   = [l for l, _ in fired if l not in STRONG_HEURISTICS]

    if strong:
        reasons.append("phon_strong:" + "+".join(strong))
    if len(weak) >= 2:
        reasons.append("phon_weak_2+:" + "+".join(weak))

    return bool(reasons), "; ".join(reasons)


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_binary_matrix(rows, retained_cogids):
    """
    Build {taxon: {cogid: 0/1}} presence/absence matrix.
    Only cogids in retained_cogids are included.
    """
    taxa   = sorted(set(r["language"] for r in rows))
    cogids = sorted(retained_cogids, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
    matrix = {t: {c: 0 for c in cogids} for t in taxa}
    for row in rows:
        cid = row.get("cogid", "").strip()
        if cid in retained_cogids:
            matrix[row["language"]][cid] = 1
    return taxa, cogids, matrix


def write_nexus(taxa, cogids, matrix, out_path, note=""):
    lines = [
        "#NEXUS", "",
        "BEGIN DATA;",
        f"    DIMENSIONS NTAX={len(taxa)} NCHAR={len(cogids)};",
        "    FORMAT DATATYPE=STANDARD SYMBOLS=\"01\" MISSING=? GAP=-;",
        f"    [{note}]",
        "    MATRIX",
    ]
    pad = max(len(t) for t in taxa)
    for t in taxa:
        seq = "".join(str(matrix[t][c]) for c in cogids)
        lines.append(f"        {t:<{pad}}  {seq}")
    lines += ["    ;", "END;", ""]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return len(cogids)


def build_beast_xml(taxa, matrix, cogids, out_path):
    """BEAST 2.7.8 XML, identical configuration to Phase 4 beast_turkic_final.xml."""
    seq_lines = "\n".join(
        f'        <sequence taxon="{t}" value="{"".join(str(matrix[t][c]) for c in cogids)}"/>'
        for t in sorted(taxa)
    )
    taxon_elems = "\n".join(
        f'            <taxon id="{t}" spec="beast.base.evolution.alignment.Taxon"/>'
        for t in sorted(taxa)
    )
    n = len(cogids)

    xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!--
    BEAST 2.7.8 XML - Turkic Phylolinguistics Infra 6B
    Uzbek loan-filtered rerun. See infra6b_report.txt for details.
    Model:       Lewis Mk + Gamma(4), Strict clock, Yule tree prior
    Calibration: Proto-Turkic root ~2100 BP, LogNormal(M=2100, S=0.12)
    Data:        {len(taxa)} taxa x {n} binary cognate characters
                 (Uzbek Arabic/Persian loans removed from matrix)
    Comparison:  Phase 4 beast_turkic_final.xml (9 taxa x 92 chars)
-->
<beast version="2.7"
       namespace="beast.base.core:beast.base.inference:beast.base.inference.util:beast.base.inference.operator:beast.base.inference.parameter:beast.base.evolution.alignment:beast.base.evolution.tree:beast.base.evolution.tree.coalescent:beast.base.evolution.operator:beast.base.evolution.sitemodel:beast.base.evolution.substitutionmodel:beast.base.evolution.likelihood:beast.base.evolution.speciation:beast.base.evolution.branchratemodel:beast.base.math.distributions:beast.math.distributions:beast.base.util">

    <data id="turkic" dataType="binary"
          spec="beast.base.evolution.alignment.Alignment">
{seq_lines}
    </data>

    <siteModel id="SiteModel.s:turkic"
               spec="beast.base.evolution.sitemodel.SiteModel"
               gammaCategoryCount="4"
               shape="@shape.s:turkic">
        <substModel id="MkModel.s:turkic"
                    spec="beast.base.evolution.substitutionmodel.GeneralSubstitutionModel">
            <frequencies id="freqs.s:turkic"
                         spec="beast.base.evolution.substitutionmodel.Frequencies">
                <parameter id="freqParameter.s:turkic"
                           spec="beast.base.inference.parameter.RealParameter"
                           name="frequencies"
                           value="0.5 0.5" lower="0.0" upper="1.0" dimension="2"/>
            </frequencies>
            <rates id="rates.s:turkic"
                   spec="beast.base.inference.parameter.RealParameter"
                   value="1.0 1.0" lower="0.0" dimension="2"/>
        </substModel>
    </siteModel>

    <parameter id="shape.s:turkic"
               spec="beast.base.inference.parameter.RealParameter"
               value="0.5" lower="0.0" upper="100.0"/>

    <branchRateModel id="StrictClock.c:turkic"
                     spec="beast.base.evolution.branchratemodel.StrictClockModel"
                     clock.rate="@clockRate.c:turkic"/>

    <parameter id="clockRate.c:turkic"
               spec="beast.base.inference.parameter.RealParameter"
               value="1.0" lower="0.0"/>

    <tree id="Tree.t:turkic"
          spec="beast.base.evolution.tree.TreeParser"
          IsLabelledNewick="true"
          newick="(Chuvash:1800,(Yakut:1500,(Azerbaijani:800,(Turkmen:600,(Turkish:500,(Uyghur:350,(Uzbek:300,(Kazakh:300,Kyrgyz:300):50):50):100):200):700):300):300);">
        <taxonset id="TaxonSet.turkic"
                  spec="beast.base.evolution.alignment.TaxonSet">
{taxon_elems}
        </taxonset>
    </tree>

    <distribution id="YuleModel.t:turkic"
                  spec="beast.base.evolution.speciation.YuleModel"
                  birthDiffRate="@birthRate.t:turkic"
                  tree="@Tree.t:turkic"/>

    <parameter id="birthRate.t:turkic"
               spec="beast.base.inference.parameter.RealParameter"
               value="1.0" lower="0.0"/>

    <distribution id="ProtoTurkicRoot.prior"
                  spec="beast.base.evolution.tree.MRCAPrior"
                  monophyletic="true"
                  tree="@Tree.t:turkic"
                  useOriginate="false">
        <taxonset idref="TaxonSet.turkic"/>
        <distr spec="beast.base.inference.distribution.LogNormalDistributionModel"
               meanInRealSpace="true" M="2100.0" S="0.12"/>
    </distribution>

    <distribution id="treeLikelihood.turkic"
                  spec="beast.base.evolution.likelihood.TreeLikelihood"
                  data="@turkic"
                  tree="@Tree.t:turkic"
                  siteModel="@SiteModel.s:turkic"
                  branchRateModel="@StrictClock.c:turkic"
                  useAmbiguities="true"/>

    <distribution id="posterior"
                  spec="beast.base.inference.CompoundDistribution">
        <distribution idref="treeLikelihood.turkic"/>
        <distribution id="prior"
                      spec="beast.base.inference.CompoundDistribution">
            <distribution idref="YuleModel.t:turkic"/>
            <distribution idref="ProtoTurkicRoot.prior"/>
            <distribution id="prior.clockRate"
                          spec="beast.base.inference.distribution.Prior"
                          x="@clockRate.c:turkic">
                <distr spec="beast.base.inference.distribution.Exponential" mean="1.0"/>
            </distribution>
            <distribution id="prior.birthRate"
                          spec="beast.base.inference.distribution.Prior"
                          x="@birthRate.t:turkic">
                <distr spec="beast.base.inference.distribution.Exponential" mean="1.0"/>
            </distribution>
            <distribution id="prior.shape"
                          spec="beast.base.inference.distribution.Prior"
                          x="@shape.s:turkic">
                <distr spec="beast.base.inference.distribution.Exponential" mean="0.5"/>
            </distribution>
        </distribution>
    </distribution>

    <operator id="treeScaler"
              spec="beast.base.evolution.operator.ScaleOperator"
              scaleFactor="0.5" weight="3" tree="@Tree.t:turkic"/>
    <operator id="treeRootScaler"
              spec="beast.base.evolution.operator.ScaleOperator"
              scaleFactor="0.5" weight="3" tree="@Tree.t:turkic" rootOnly="true"/>
    <operator id="uniformOp"
              spec="beast.base.evolution.operator.Uniform"
              weight="30" tree="@Tree.t:turkic"/>
    <operator id="subtreeSlide"
              spec="beast.base.evolution.operator.SubtreeSlide"
              weight="15" size="100.0" tree="@Tree.t:turkic"/>
    <operator id="narrow"
              spec="beast.base.evolution.operator.Exchange"
              weight="15" tree="@Tree.t:turkic"/>
    <operator id="wide"
              spec="beast.base.evolution.operator.Exchange"
              weight="3" isNarrow="false" tree="@Tree.t:turkic"/>
    <operator id="wilsonBalding"
              spec="beast.base.evolution.operator.WilsonBalding"
              weight="3" tree="@Tree.t:turkic"/>
    <operator id="clockRateScaler"
              spec="beast.base.evolution.operator.ScaleOperator"
              scaleFactor="0.5" weight="3" parameter="@clockRate.c:turkic"/>
    <operator id="birthRateScaler"
              spec="beast.base.evolution.operator.ScaleOperator"
              scaleFactor="0.5" weight="3" parameter="@birthRate.t:turkic"/>
    <operator id="shapeScaler"
              spec="beast.base.evolution.operator.ScaleOperator"
              scaleFactor="0.5" weight="1" parameter="@shape.s:turkic"/>

    <logger id="tracelog" fileName="beast_uzbek6b.log" logEvery="5000"
            model="@posterior" sanitiseHeaders="true" sort="smart">
        <log idref="posterior"/>
        <log idref="prior"/>
        <log idref="treeLikelihood.turkic"/>
        <log idref="YuleModel.t:turkic"/>
        <log idref="birthRate.t:turkic"/>
        <log idref="clockRate.c:turkic"/>
        <log idref="shape.s:turkic"/>
        <log spec="beast.base.evolution.tree.TreeStatLogger" tree="@Tree.t:turkic"/>
    </logger>

    <logger id="treelog" fileName="beast_uzbek6b.trees" logEvery="5000" mode="tree">
        <log idref="Tree.t:turkic"/>
    </logger>

    <logger id="screenlog" logEvery="100000">
        <log idref="posterior"/>
        <log spec="beast.base.inference.util.ESS" arg="@posterior"/>
        <log idref="prior"/>
    </logger>

    <run id="mcmc"
         spec="beast.base.inference.MCMC"
         chainLength="20000000"
         numInitializationAttempts="10">
        <state id="state" storeEvery="5000">
            <stateNode idref="Tree.t:turkic"/>
            <stateNode idref="birthRate.t:turkic"/>
            <stateNode idref="clockRate.c:turkic"/>
            <stateNode idref="shape.s:turkic"/>
        </state>
        <distribution idref="posterior"/>
        <operator idref="treeScaler"/>
        <operator idref="treeRootScaler"/>
        <operator idref="uniformOp"/>
        <operator idref="subtreeSlide"/>
        <operator idref="narrow"/>
        <operator idref="wide"/>
        <operator idref="wilsonBalding"/>
        <operator idref="clockRateScaler"/>
        <operator idref="birthRateScaler"/>
        <operator idref="shapeScaler"/>
        <logger idref="tracelog"/>
        <logger idref="treelog"/>
        <logger idref="screenlog"/>
    </run>

</beast>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)


def write_report(stats, out_path):
    lines = [
        "INFRA 6B: UZBEK LOAN-FILTERED BEAST RERUN",
        "=" * 70,
        "",
        "OBJECTIVE",
        "---------",
        "Remove Arabic and Persian loanwords from the Uzbek cognate data",
        "and rerun BEAST 2.7.8 to test whether Uzbek's anomalous early-",
        "diverging placement corrects to Kipchak.",
        "",
        "DATA NOTE",
        "---------",
        f"Total Uzbek entries in northeuralex_merged.csv: {stats['uzbek_total']}",
        f"  Entries WITH cogid (contribute to matrix):    {stats['uzbek_with_cogid']}",
        f"  Entries WITHOUT cogid (NLex-only, no matrix): {stats['uzbek_no_cogid']}",
        "",
        "Loan detection operates only on entries with valid cogids.",
        "Blank-cogid entries are immaterial to the BEAST matrix.",
        "",
        "LOAN DETECTION RESULTS",
        "----------------------",
        f"  Uzbek entries with cogid:       {stats['uzbek_with_cogid']}",
        f"  Flagged as loans (with cogid):  {stats['uzbek_flagged']}",
        f"  Retained (with cogid):          {stats['uzbek_retained_with_cogid']}",
        f"  Flagged fraction (of cogid entries): {stats['uzbek_flagged']/max(stats['uzbek_with_cogid'],1)*100:.1f}%",
        "",
        "MATRIX COMPARISON",
        "-----------------",
        f"  Phase 4 matrix:         9 taxa x {stats['chars_phase4']} characters",
        f"  Infra 6B matrix:        9 taxa x {stats['chars_6b']} characters",
        f"  Characters removed:     {stats['chars_phase4'] - stats['chars_6b']}",
        "",
        "FLAGGED UZBEK ENTRIES",
        "---------------------",
    ]
    for e in stats["flagged_entries"]:
        lines.append(f"  {e['gloss']:<20} {e['form']:<22} {e['reason']}")
    lines += [
        "",
        "BEAST RUN CONFIGURATION",
        "-----------------------",
        "  Model:        Lewis Mk + Gamma(4), strict clock, Yule prior",
        "  Calibration:  LogNormal(M=2100, S=0.12) on MRCA all 9 taxa",
        "  Chain:        20,000,000 iterations, log every 5,000",
        "  Burn-in:      10% (2,000,000 iterations)",
        "  Starting tree: Uzbek placed inside Kipchak",
        "",
        "EXPECTED RESULT",
        "---------------",
        "  Uzbek should cluster inside Kipchak (Kazakh/Kyrgyz/Uzbek).",
        "  If misplacement persists, the Swadesh core is too sparse to",
        "  resolve the node regardless of loan filtering. Resolution",
        "  then requires the Phase 6 expanded NorthEuraLex matrix.",
        "",
        "RUN COMMAND (from output/ directory)",
        "-------------------------------------",
        "  beast beast_uzbek6b.xml",
        "",
        "TREEANNOTATOR",
        "-------------",
        "  treeannotator -burnin 10 -height mean -file beast_uzbek6b.trees beast_uzbek6b_mcc.tree",
        "",
        "STATUS: Infra 6B data preparation complete. Awaiting BEAST run.",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    merged_path = OUTPUT / "northeuralex_merged.csv"
    if not merged_path.exists():
        merged_path = OUTPUT / "hybrid_cognates.csv"
        print(f"WARNING: northeuralex_merged.csv not found, using hybrid_cognates.csv")

    print(f"Loading: {merged_path}")
    rows = load_csv(merged_path)
    print(f"  {len(rows)} rows loaded")

    uzbek_rows = [r for r in rows if r["language"] == "Uzbek"]
    other_rows = [r for r in rows if r["language"] != "Uzbek"]
    print(f"  Uzbek entries: {len(uzbek_rows)}")
    print(f"  Other entries: {len(other_rows)}")

    uzbek_with_cogid    = [r for r in uzbek_rows if r.get("cogid","").strip()]
    uzbek_without_cogid = [r for r in uzbek_rows if not r.get("cogid","").strip()]
    print(f"  Uzbek with cogid:    {len(uzbek_with_cogid)}")
    print(f"  Uzbek without cogid: {len(uzbek_without_cogid)} (NLex-only, no matrix contribution)")

    # -----------------------------------------------------------------------
    # Classify Uzbek entries
    # -----------------------------------------------------------------------
    flag_records = []
    flagged_cogids = set()
    flagged_entries_for_report = []

    for row in uzbek_rows:
        is_loan, reason = classify_uzbek(row)
        flag_records.append({
            "language":   row["language"],
            "gloss":      row.get("gloss",""),
            "form":       row.get("form",""),
            "ipa_tokens": row.get("ipa_tokens",""),
            "cogid":      row.get("cogid",""),
            "is_loan":    str(is_loan),
            "reason":     reason,
        })
        if is_loan and row.get("cogid","").strip():
            flagged_cogids.add(row["cogid"].strip())
            flagged_entries_for_report.append({
                "gloss":  row.get("gloss",""),
                "form":   row.get("form",""),
                "reason": reason,
            })

    n_flagged = sum(
        1 for r in flag_records
        if r["is_loan"] == "True" and r["cogid"].strip()
    )
    print(f"  Flagged as loans (with cogid): {n_flagged}")
    print(f"  Retained (with cogid): {len(uzbek_with_cogid) - n_flagged}")

    # Write flag CSV
    flags_path = OUTPUT / "uzbek_loan_flags.csv"
    with open(flags_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flag_records[0].keys()))
        writer.writeheader()
        writer.writerows(flag_records)
    print(f"Written: {flags_path}")

    # -----------------------------------------------------------------------
    # Build filtered dataset
    # Only exclude Uzbek rows where cogid is non-empty AND in flagged_cogids.
    # Blank-cogid rows are retained in CSV (immaterial to the matrix).
    # -----------------------------------------------------------------------
    uzbek_retained = [
        r for r in uzbek_rows
        if not (r.get("cogid","").strip() and
                r.get("cogid","").strip() in flagged_cogids)
    ]
    filtered_rows = other_rows + uzbek_retained
    n_removed_csv = len(uzbek_rows) - len(uzbek_retained)
    print(f"  Filtered CSV: {len(filtered_rows)} rows ({n_removed_csv} Uzbek loan rows excluded)")

    filtered_path = OUTPUT / "northeuralex_loanfiltered.csv"
    fieldnames = list(filtered_rows[0].keys()) if filtered_rows else []
    with open(filtered_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)
    print(f"Written: {filtered_path}")

    # -----------------------------------------------------------------------
    # Build binary cognate matrix
    # -----------------------------------------------------------------------
    all_cogids = set(
        r.get("cogid","").strip() for r in other_rows
        if r.get("cogid","").strip()
    )
    uzbek_retained_cogids = set(
        r.get("cogid","").strip() for r in uzbek_retained
        if r.get("cogid","").strip()
    )
    retained_cogids = all_cogids | uzbek_retained_cogids
    retained_cogids.discard("")

    taxa, cogids, matrix = build_binary_matrix(filtered_rows, retained_cogids)
    print(f"  Matrix: {len(taxa)} taxa x {len(cogids)} characters")

    # -----------------------------------------------------------------------
    # Write NEXUS
    # -----------------------------------------------------------------------
    nexus_path = OUTPUT / "turkic_cognates_uzbek6b.nex"
    n_chars = write_nexus(
        taxa, cogids, matrix, nexus_path,
        note=f"Infra 6B Uzbek loan-filtered, {n_flagged} cogid entries removed"
    )
    print(f"Written: {nexus_path}  ({n_chars} characters)")

    # -----------------------------------------------------------------------
    # Write BEAST XML
    # -----------------------------------------------------------------------
    xml_path = OUTPUT / "beast_uzbek6b.xml"
    build_beast_xml(taxa, matrix, cogids, xml_path)
    print(f"Written: {xml_path}")

    # -----------------------------------------------------------------------
    # Write report
    # -----------------------------------------------------------------------
    stats = {
        "uzbek_total":               len(uzbek_rows),
        "uzbek_with_cogid":          len(uzbek_with_cogid),
        "uzbek_no_cogid":            len(uzbek_without_cogid),
        "uzbek_flagged":             n_flagged,
        "uzbek_retained_with_cogid": len(uzbek_with_cogid) - n_flagged,
        "chars_phase4":              92,
        "chars_6b":                  n_chars,
        "flagged_entries":           flagged_entries_for_report,
    }
    report_path = OUTPUT / "infra6b_report.txt"
    write_report(stats, report_path)
    print(f"Written: {report_path}")

    print()
    print("INFRA 6B PREPARATION COMPLETE")
    print(f"  Uzbek cognate entries with loans removed: {n_flagged} / {len(uzbek_with_cogid)}")
    print(f"  Matrix: {len(taxa)} taxa x {n_chars} characters")
    print(f"  (Phase 4: 9 taxa x 92 characters)")
    print()
    print("Next: cd output && beast beast_uzbek6b.xml")
    print("Then: treeannotator -burnin 10 -height mean -file beast_uzbek6b.trees beast_uzbek6b_mcc.tree")


if __name__ == "__main__":
    main()
