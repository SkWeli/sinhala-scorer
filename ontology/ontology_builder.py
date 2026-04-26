"""
Colonial Sri Lanka Ontology using owlready2.
Builds an OWL 2.0 ontology with key concepts, rulers, events, and relationships.
"""

from owlready2 import *
import os
from config.settings import ONTOLOGY_PATH


def build_ontology():
    onto = get_ontology("http://colonial-sl.owl#")

    with onto:

        # ── Classes ──────────────────────────────────────────────────────────
        class ColonialPower(Thing):        pass
        class Ruler(Thing):                pass
        class Event(Thing):                pass
        class Policy(Thing):               pass
        class EconomicSystem(Thing):       pass
        class ReligiousImpact(Thing):      pass
        class AdministrativeReform(Thing): pass
        class Location(Thing):             pass
        class LegalSystem(Thing):          pass

        # ── Object Properties ─────────────────────────────────────────────────
        class ruled_by(ObjectProperty):
            domain = [Location]
            range  = [ColonialPower]

        class introduced(ObjectProperty):
            domain = [ColonialPower]
            range  = [Policy]

        class had_ruler(ObjectProperty):
            domain = [ColonialPower]
            range  = [Ruler]

        class succeeded_by(ObjectProperty):
            domain = [ColonialPower]
            range  = [ColonialPower]

        class caused(ObjectProperty):
            domain = [Event]
            range  = [Policy]

        # ── Colonial Power Instances ──────────────────────────────────────────
        portuguese = ColonialPower("Portuguese")
        dutch      = ColonialPower("Dutch")
        british    = ColonialPower("British")

        portuguese.succeeded_by = [dutch]
        dutch.succeeded_by      = [british]

        # ── Ruler Instances ───────────────────────────────────────────────────
        afonso_albuquerque = Ruler("AlfonsoDeAlbuquerque")
        lopo_soares        = Ruler("LopoSoaresDeAlbergaria")
        rijckloff          = Ruler("RijckloffVanGoens")
        robert_brownrigg   = Ruler("RobertBrownrigg")

        portuguese.had_ruler = [afonso_albuquerque, lopo_soares]
        dutch.had_ruler      = [rijckloff]
        british.had_ruler    = [robert_brownrigg]

        # ── Policy / Economic / Administrative Instances ──────────────────────
        cinnamon_monopoly  = Policy("CinnamonMonopoly")
        forced_conversion  = Policy("ForcedConversion")
        rajakariya         = Policy("Rajakariya_ForcedLabour")
        voc_trade          = EconomicSystem("VOC_TradeMonopoly")
        cb_reforms         = AdministrativeReform("ColebrookeCameronReforms_1833")
        abolish_rajakariya = AdministrativeReform("AbolitionOfRajakariya_1832")
        plantation_economy = EconomicSystem("PlantationEconomy")
        roman_dutch_law    = LegalSystem("RomanDutchLaw")
        english_law        = LegalSystem("EnglishCommonLaw")

        portuguese.introduced = [cinnamon_monopoly, forced_conversion, rajakariya]
        dutch.introduced      = [voc_trade, roman_dutch_law]
        british.introduced    = [cb_reforms, abolish_rajakariya,
                                  plantation_economy, english_law]

        # ── Location Instances ────────────────────────────────────────────────
        colombo = Location("Colombo")
        galle   = Location("Galle")
        jaffna  = Location("Jaffna")
        kandy   = Location("Kandy")

        colombo.ruled_by = [portuguese, dutch, british]
        galle.ruled_by   = [portuguese, dutch]
        jaffna.ruled_by  = [portuguese, dutch]
        kandy.ruled_by   = [british]

        # ── Religious Impact Sub-classes ──────────────────────────────────────
        class CatholicMission(ReligiousImpact):       pass
        class DutchReformedChurch(ReligiousImpact):   pass
        class BuddhistRevival(ReligiousImpact):        pass

        # ── Event Instances ───────────────────────────────────────────────────
        Event("PortugueseArrival_1505")
        Event("DutchConquest_1658")
        Event("BritishTakeover_1796")
        Event("KandyanConvention_1815")

    # Save OWL file
    os.makedirs(os.path.dirname(ONTOLOGY_PATH), exist_ok=True)
    onto.save(file=ONTOLOGY_PATH, format="rdfxml")
    print(f"[Ontology] Saved → {ONTOLOGY_PATH}")
    return onto


def load_ontology():
    """Load existing ontology or build fresh if missing."""
    if os.path.exists(ONTOLOGY_PATH):
        return get_ontology(
            f"file://{os.path.abspath(ONTOLOGY_PATH)}"
        ).load()
    return build_ontology()


def query_ontology_concepts(topic_keywords: list) -> list:
    """
    Return ontology class/individual names relevant to given keywords.
    Used by the Retrieval Agent to inject semantic context into the scorer.
    """
    onto = load_ontology()
    results = []
    kws = [k.lower() for k in topic_keywords]

    for cls in onto.classes():
        name = cls.name.lower().replace("_", " ")
        if any(k in name for k in kws):
            results.append(f"Class: {cls.name}")

    for ind in onto.individuals():
        name = ind.name.lower().replace("_", " ")
        if any(k in name for k in kws):
            props = []
            for prop in ind.get_properties():
                for val in prop[ind]:
                    v_name = val.name if hasattr(val, "name") else str(val)
                    props.append(f"{prop.name}→{v_name}")
            entry = f"Entity: {ind.name}"
            if props:
                entry += f" [{'; '.join(props)}]"
            results.append(entry)

    return results