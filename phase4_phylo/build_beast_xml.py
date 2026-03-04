"""
Phase 4 - Step 2: Generate BEAST 2 XML
=======================================
Produces a fully configured BEAST 2 XML file for Bayesian phylogenetic
inference of the Turkic language family.

Configuration:
  - Substitution model: Stochastic Dollo (binary presence/absence,
    character can only be gained once but lost independently)
  - Clock model:        Relaxed log-normal (uncorrelated)
  - Tree prior:         Yule process (diversification)
  - Calibration:        Proto-Turkic root ~2100 BP (lognormal prior)

Why Stochastic Dollo over Covarion?
  Cognate characters are asymmetric: a proto-form can give rise to
  cognates independently (once) but can be lost in parallel in multiple
  lineages. Covarion models rate heterogeneity across sites but assumes
  symmetric gain/loss — less appropriate for cognate data. Stochastic
  Dollo (implemented in BEAST 2 as BirthDeathSkylineLikelihood with
  binary characters, or via the SDolloLikelihood package) is the
  standard for cognate-based phylolinguistics (Nicholls & Gray 2006).

  If the SDolloLikelihood package is not installed, the XML falls back
  to a Lewis Mk binary model (standard for morphological/cognate data).
  This is noted in the XML comments.

Calibration:
  Savelyev & Robbeets (2020) date Proto-Turkic at ~2100 BP with
  credible interval roughly 1800-2400 BP. We encode this as a
  LogNormal prior on the root height:
    M = log(2100) ≈ 7.65,  S = 0.12
  which places the median at 2100 and 95% CI approximately 1680-2630 BP.

Output: output/beast_turkic.xml
"""

from pathlib import Path
import textwrap

ROOT   = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"

TAXA = [
    "Azerbaijani", "Chuvash", "Kazakh", "Kyrgyz",
    "Turkish", "Turkmen", "Uyghur", "Uzbek", "Yakut"
]

# Calibration parameters (Proto-Turkic root, years BP)
# LogNormal(M=7.6497, S=0.12) -> median ~2100, 95% CI ~[1680, 2630]
ROOT_LOGNORMAL_M = "7.6497"
ROOT_LOGNORMAL_S = "0.12"

MCMC_CHAIN_LENGTH = "20000000"
LOG_EVERY         = "5000"


def indent(text, spaces=4):
    return textwrap.indent(text, " " * spaces)


def make_xml(nexus_filename="turkic_cognates.nex"):
    taxa_block = "\n".join(
        f'        <taxon id="{t}" spec="Taxon"/>' for t in TAXA
    )

    seq_block = "\n".join(
        f'        <sequence taxon="{t}" value="?"/>' for t in TAXA
    )

    xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!--
    BEAST 2 XML for Turkic Phylolinguistics
    Phase 4 - Computational Historical Linguistics Project
    
    SETUP INSTRUCTIONS:
    1. Install BEAST 2 (https://www.beast2.org/)
    2. Install packages: BEAST_CLASSIC (Stochastic Dollo likelihood)
       via BEAUti > File > Manage Packages
    3. Load this XML via BEAST 2 GUI or run:
       beast -threads 4 beast_turkic.xml
    4. Analyse output with Tracer; ensure ESS > 200 for all parameters.
    5. Build consensus tree with TreeAnnotator (burnin 25%).
    
    Substitution model: Stochastic Dollo (Lewis Mk fallback noted below)
    Clock model:        Relaxed LogNormal (uncorrelated)
    Tree prior:         Yule
    Calibration:        Proto-Turkic root ~ 2100 BP (LogNormal prior)
    Data:               Binary cognate presence/absence from LexStat
                        and Savelyev & Robbeets (2020)
    
    Characters: See turkic_cognates.nex
    LexStat-only characters may inflate tree uncertainty; consider
    a sensitivity run excluding those characters (CHARSET LexStat).
-->
<beast beautitemplate='Standard' beautistatus='' namespace="beast.core:beast.evolution.alignment:beast.evolution.tree.coalescent:beast.core.util:beast.evolution.nuc:beast.evolution.operators:beast.evolution.sitemodel:beast.evolution.substitutionmodel:beast.evolution.likelihood" required="" version="2.6">

    <!-- ============================================================ -->
    <!-- DATA                                                          -->
    <!-- ============================================================ -->
    <data id="turkic_alignment" dataType="standard" spec="Alignment"
          name="alignment">
        <!--
            Replace value="?" with actual sequences from turkic_cognates.nex
            before running. BEAUti will parse the NEXUS file directly if
            loaded via File > Import Alignment.
        -->
{seq_block}
    </data>

    <!-- ============================================================ -->
    <!-- SITE MODEL: Stochastic Dollo                                  -->
    <!-- If BEAST_CLASSIC package unavailable, use Lewis Mk (below)   -->
    <!-- ============================================================ -->
    <!--
    OPTION A: Stochastic Dollo (preferred for cognate data)
    Requires BEAST_CLASSIC package. Uncomment and use in place of
    the Mk model below.

    <siteModel id="siteModel" spec="beast.evolution.sitemodel.SiteModel">
        <substModel spec="beast.evolution.substitutionmodel.SDolloLikelihood"
                    deathRate="@deathRate" mu="@clockRate"/>
        <parameter id="deathRate" value="0.5" lower="0.0"/>
    </siteModel>
    -->

    <!-- OPTION B: Lewis Mk binary model (fallback, widely available) -->
    <siteModel id="siteModel" spec="beast.evolution.sitemodel.SiteModel">
        <substModel spec="beast.evolution.substitutionmodel.GeneralSubstitutionModel"
                    id="mkModel">
            <frequencies id="freqs" spec="Frequencies">
                <parameter id="freqParameter" value="0.5 0.5" lower="0.0" upper="1.0"/>
            </frequencies>
            <rates id="rates" spec="parameter.RealParameter" value="1.0"
                   lower="0.0"/>
        </substModel>
        <shape spec="parameter.RealParameter" id="gammaShape" value="0.5"
               lower="0.0"/>
        <gammaCategoryCount spec="parameter.IntegerParameter" value="4"/>
    </siteModel>

    <!-- ============================================================ -->
    <!-- CLOCK MODEL: Uncorrelated Relaxed LogNormal                   -->
    <!-- ============================================================ -->
    <branchRateModel id="relaxedClock"
                     spec="beast.evolution.branchratemodel.UCRelaxedClockModel"
                     rateCategories="@rateCategories"
                     tree="@tree"
                     normalize="true">
        <LogNormal id="LogNormalDistributionModel" meanInRealSpace="true"
                   name="distr">
            <parameter id="ucldMean" spec="parameter.RealParameter"
                       value="1.0" lower="0.0" upper="100.0" name="M"/>
            <parameter id="ucldStdev" spec="parameter.RealParameter"
                       value="0.5" lower="0.0" upper="10.0" name="S"/>
        </LogNormal>
    </branchRateModel>

    <parameter id="rateCategories" spec="parameter.IntegerParameter"
               value="1" dimension="16"/>

    <!-- ============================================================ -->
    <!-- TREE PRIOR: Yule                                              -->
    <!-- ============================================================ -->
    <treeDistribution id="yuleModel" spec="beast.evolution.speciation.YuleModel"
                      birthDiffRate="@birthRate" tree="@tree">
        <parameter id="birthRate" spec="parameter.RealParameter"
                   value="1.0" lower="0.0"/>
    </treeDistribution>

    <!-- ============================================================ -->
    <!-- TREE                                                          -->
    <!-- ============================================================ -->
    <tree id="tree" spec="beast.util.TreeParser"
          IsLabelledNewick="true"
          newick="(Chuvash,(Yakut,(Azerbaijani,(Turkmen,(Turkish,(Uzbek,(Uyghur,(Kazakh,Kyrgyz))))))));">
        <!-- Starting tree from Savelyev & Robbeets 2020 topology.    -->
        <!-- BEAST will sample tree topology via operators below.     -->
        <taxonset id="taxonSet" spec="TaxonSet">
{taxa_block}
        </taxonset>
    </tree>

    <!-- ============================================================ -->
    <!-- CALIBRATION: Proto-Turkic root ~ 2100 BP                     -->
    <!-- LogNormal(M=7.6497, S=0.12) -> median=2100, 95% CI~1680-2630  -->
    <!-- ============================================================ -->
    <distribution id="calibration.root" spec="beast.math.distributions.MRCAPrior"
                  monophyletic="true" tree="@tree" useOriginate="true">
        <taxonset idref="taxonSet"/>
        <LogNormal id="LogNormal.root" name="distr" offset="0.0">
            <parameter spec="parameter.RealParameter" value="{ROOT_LOGNORMAL_M}" name="M"/>
            <parameter spec="parameter.RealParameter" value="{ROOT_LOGNORMAL_S}" name="S"/>
        </LogNormal>
    </distribution>

    <!-- ============================================================ -->
    <!-- TREE LIKELIHOOD                                               -->
    <!-- ============================================================ -->
    <distribution id="treeLikelihood" spec="TreeLikelihood"
                  data="@turkic_alignment"
                  tree="@tree"
                  siteModel="@siteModel"
                  branchRateModel="@relaxedClock"
                  useAmbiguities="true"/>

    <!-- ============================================================ -->
    <!-- POSTERIOR                                                     -->
    <!-- ============================================================ -->
    <distribution id="posterior" spec="util.CompoundDistribution">
        <distribution idref="treeLikelihood"/>
        <distribution id="prior" spec="util.CompoundDistribution">
            <distribution idref="yuleModel"/>
            <distribution idref="calibration.root"/>
            <!-- Clock priors -->
            <prior id="ucldMeanPrior" x="@ucldMean" spec="Prior">
                <LogNormal name="distr" M="0.0" S="1.0"/>
            </prior>
            <prior id="ucldStdevPrior" x="@ucldStdev" spec="Prior">
                <Exponential name="distr" mean="0.3333"/>
            </prior>
            <!-- Birth rate prior -->
            <prior id="birthRatePrior" x="@birthRate" spec="Prior">
                <Exponential name="distr" mean="1.0"/>
            </prior>
            <!-- Gamma shape prior -->
            <prior id="gammaShapePrior" x="@gammaShape" spec="Prior">
                <Exponential name="distr" mean="0.5"/>
            </prior>
        </distribution>
    </distribution>

    <!-- ============================================================ -->
    <!-- OPERATORS                                                     -->
    <!-- ============================================================ -->
    <operator id="treeScaler"       spec="ScaleOperator"    scaleFactor="0.5"  weight="3"   tree="@tree"/>
    <operator id="treeRootScaler"   spec="ScaleOperator"    scaleFactor="0.5"  weight="3"   tree="@tree" rootOnly="true"/>
    <operator id="uniformTree"      spec="Uniform"          weight="30"        tree="@tree"/>
    <operator id="subtreeSlide"     spec="SubtreeSlide"     weight="15"        tree="@tree" size="0.1"/>
    <operator id="narrow"           spec="Exchange"         weight="15"        tree="@tree"/>
    <operator id="wide"             spec="Exchange"         weight="3"         tree="@tree" isNarrow="false"/>
    <operator id="wilsonBalding"    spec="WilsonBalding"    weight="3"         tree="@tree"/>
    <operator id="birthRateScaler"  spec="ScaleOperator"    scaleFactor="0.5"  weight="3"   parameter="@birthRate"/>
    <operator id="ucldMeanScaler"   spec="ScaleOperator"    scaleFactor="0.5"  weight="3"   parameter="@ucldMean"/>
    <operator id="ucldStdevScaler"  spec="ScaleOperator"    scaleFactor="0.5"  weight="3"   parameter="@ucldStdev"/>
    <operator id="rateCategories"   spec="IntRandomWalkOperator" weight="10"   parameter="@rateCategories" windowSize="1"/>
    <operator id="gammaShapeScaler" spec="ScaleOperator"    scaleFactor="0.5"  weight="1"   parameter="@gammaShape"/>

    <!-- ============================================================ -->
    <!-- LOGGERS                                                       -->
    <!-- ============================================================ -->
    <logger id="tracelog" fileName="beast_turkic.log" logEvery="{LOG_EVERY}"
            model="@posterior" sanitiseHeaders="true" sort="smart">
        <log idref="posterior"/>
        <log idref="prior"/>
        <log idref="treeLikelihood"/>
        <log idref="yuleModel"/>
        <log idref="birthRate"/>
        <log idref="ucldMean"/>
        <log idref="ucldStdev"/>
        <log idref="gammaShape"/>
        <log spec="beast.evolution.tree.TreeStatLogger" tree="@tree"/>
    </logger>

    <logger id="treelog" fileName="beast_turkic.trees" logEvery="{LOG_EVERY}"
            mode="tree">
        <log idref="tree"/>
    </logger>

    <logger id="screenlog" logEvery="100000">
        <log idref="posterior"/>
        <log spec="util.ESS" arg="@posterior"/>
    </logger>

    <!-- ============================================================ -->
    <!-- MCMC                                                          -->
    <!-- ============================================================ -->
    <run id="mcmc" spec="MCMC" chainLength="{MCMC_CHAIN_LENGTH}"
         numInitializationAttempts="10">
        <state id="state" storeEvery="5000">
            <stateNode idref="tree"/>
            <stateNode idref="birthRate"/>
            <stateNode idref="ucldMean"/>
            <stateNode idref="ucldStdev"/>
            <stateNode idref="rateCategories"/>
            <stateNode idref="gammaShape"/>
        </state>
        <distribution idref="posterior"/>
        <operator idref="treeScaler"/>
        <operator idref="treeRootScaler"/>
        <operator idref="uniformTree"/>
        <operator idref="subtreeSlide"/>
        <operator idref="narrow"/>
        <operator idref="wide"/>
        <operator idref="wilsonBalding"/>
        <operator idref="birthRateScaler"/>
        <operator idref="ucldMeanScaler"/>
        <operator idref="ucldStdevScaler"/>
        <operator idref="rateCategories"/>
        <operator idref="gammaShapeScaler"/>
        <logger idref="tracelog"/>
        <logger idref="treelog"/>
        <logger idref="screenlog"/>
    </run>

</beast>
"""
    return xml


def main():
    xml = make_xml()
    outpath = OUTPUT / "beast_turkic.xml"
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"BEAST XML written to: {outpath}")
    print(f"Chain length: {MCMC_CHAIN_LENGTH} iterations, log every {LOG_EVERY}")
    print(f"Root calibration: LogNormal(M={ROOT_LOGNORMAL_M}, S={ROOT_LOGNORMAL_S})")
    print(f"  -> Median ~2100 BP, 95% CI ~1680-2630 BP")
    print()
    print("NEXT STEPS:")
    print("  1. Install BEAST 2 (https://www.beast2.org/)")
    print("  2. Install BEAST_CLASSIC package (Stochastic Dollo)")
    print("  3. Load turkic_cognates.nex via BEAUti to populate sequences")
    print("  4. Run: beast -threads 4 output/beast_turkic.xml")
    print("  5. Diagnose in Tracer; target ESS > 200 on all parameters")
    print("  6. Build MCC tree: TreeAnnotator -burnin 25 beast_turkic.trees")


if __name__ == "__main__":
    main()
