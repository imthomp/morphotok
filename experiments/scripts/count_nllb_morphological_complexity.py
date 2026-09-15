#!/usr/bin/env python3
"""
Count how many NLLB-200 / FLORES-200 languages are morphologically complex.

SSOT output: results/nllb_morphology_stats.json

Methodology
-----------
1. Language list: the 204 FLORES-200 language-script entries, scraped from the
   official FLORES-200 README (facebookresearch/flores, flores200/README.md,
   fetched 2026-08-19). NLLB-200's headline "200 languages" collapses a
   handful of same-language dual-script pairs (Acehnese, Banjar, Kashmiri,
   Central Kanuri, Minangkabau, Tamasheq, Modern Standard Arabic Latn/Arab,
   Chinese Simplified/Traditional) into single languages. We report BOTH the
   raw 204-entry count and a "collapsed by base ISO 639-3 code" count so the
   discrepancy with the commonly-quoted "200" is explicit and auditable
   rather than silently fudged.

2. Morphological classification per language, in order of preference:
   a. WALS (Dryer & Haspelmath, eds. 2013, WALS Online) Chapter 20A
      "Fusion of Selected Inflectional Formatives" -- used directly where a
      FLORES-200 language has a WALS datapoint under the same or a
      near-synonymous name (e.g. "Arabic (Egyptian)" for arz, "Indonesian"
      for ind, "Modern Standard Arabic" family for arb/ars/ary/acm/aeb/apc/
      ajp/acq/arz). WALS 20A distinguishes isolating / exclusively
      concatenative / tonal / ablaut fusion patterns, which directly and
      unambiguously identifies isolating, tonal, and introflexive
      (root-and-pattern / ablaut) languages.
   b. WALS does NOT by itself distinguish agglutinative from fusional among
      "concatenative" languages (that distinction is WALS Chapter 21A,
      Exponence of Selected Inflectional Formatives, which we did not have
      full-coverage data for in this pass). For concatenative languages
      without a directly fetched 21A value, we fall back to (c).
   c. Standard family/genus-level typological consensus from the general
      linguistic-typology literature (Comrie, "Language Universals and
      Linguistic Typology", 2nd ed. 1989; Whaley, "Introduction to Typology",
      1997; Velupillai, "An Introduction to Linguistic Typology", 2012;
      WALS genus/family metadata, Dryer & Haspelmath 2013). This is applied
      at the genus level (e.g. "Slavic -> fusional", "Turkic ->
      agglutinative", "Bantu -> agglutinative", "Semitic -> introflexive",
      "Kwa/Mande -> isolating", "Philippine Austronesian -> agglutinative",
      "Malayic/western Malayo-Polynesian -> isolating") with a handful of
      well-documented exceptions applied individually (English and
      Afrikaans as the analytic outliers of Germanic; creoles as
      prototypically isolating; Munda as the agglutinative outlier within
      Austroasiatic).
   d. Any language whose genus/family typology is not confidently known to
      the authors of this script, or that mixes types too idiosyncratically
      to place, is marked "unclassified" rather than guessed. See the
      "unclassified" list in the output JSON.

3. "Morphologically complex" is defined as belonging to {agglutinative,
   fusional, introflexive, polysynthetic} -- i.e. everything except
   isolating/analytic. This definition is a variable (COMPLEX_TYPES below)
   so it can be trivially changed (e.g. to exclude fusional, or to require
   agglutinative+polysynthetic only) and the script re-run.

Known limitations (must be disclosed to any reader of the resulting number):
  - This is a coarse, categorical proxy. Real morphological complexity is
    continuous (see WALS 22A synthesis counts) and many languages are
    genuinely mixed-type (e.g. English is analytic in the nominal system but
    retains some fusional verb morphology; Georgian and Basque combine
    agglutinative affixation with polypersonal agreement approaching
    polysynthesis).
  - Classification (c) is genus/family-level, not per-language-verified for
    every one of the ~170 languages that fall outside direct WALS 20A
    coverage. Treat those entries as "typological consensus", not
    WALS-verified.
  - This script's language list and WALS values were transcribed by an LLM
    agent from web-fetched sources on 2026-08-19 without a human spot-check
    against the primary WALS/FLORES-200 pages. Isaac should independently
    verify a random sample (recommend >= 15 languages) against wals.info and
    github.com/facebookresearch/flores before quoting this number to his
    advisor or in a paper.
  - Mapudungun (arn), Nahuatl (nah), and Inuktitut (ikt) -- all three cited
    as primary polysynthetic languages in plan.md's Language Selection table
    -- are NOT present in the official 204-entry FLORES-200 list fetched for
    this script. If they are truly targeted, plan.md should either drop the
    "NLLB coverage" selection criterion for them or flag them as
    out-of-NLLB-200 additions requiring a different (e.g. OPUS/Americas NLP)
    parallel corpus, which the "Polysynthetic notes" section already
    partially acknowledges for Nahuatl/Inuktitut but not for Mapudungun.

Sources:
  - Team, NLLB, et al. (2022). "No Language Left Behind: Scaling
    Human-Centered Machine Translation." arXiv:2207.04672.
  - facebookresearch/flores, flores200/README.md (language table).
  - Dryer, Matthew S. & Haspelmath, Martin (eds.) 2013. WALS Online
    (v2020.3). Chapter 20A "Fusion of Selected Inflectional Formatives",
    https://wals.info/chapter/20 . Zenodo: https://doi.org/10.5281/zenodo.7385533
  - Comrie, B. (1989). Language Universals and Linguistic Typology, 2nd ed.
  - Velupillai, V. (2012). An Introduction to Linguistic Typology.
"""

import json
import os
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_TSV = os.path.join(SCRIPT_DIR, "..", "data", "flores200_languages_raw.tsv")
OUT_JSON = os.path.join(SCRIPT_DIR, "..", "results", "nllb_morphology_stats.json")

# Definition of "morphologically complex" -- change this set and re-run to
# get a different (explicitly-labeled) number.
COMPLEX_TYPES = {"agglutinative", "fusional", "introflexive", "polysynthetic"}

WALS_20A_SOURCE = "WALS_20A (Dryer & Haspelmath 2013, wals.info/chapter/20)"
FAMILY_SOURCE = "family/genus typological consensus (Comrie 1989; Velupillai 2012; WALS genus metadata)"

# base ISO 639-3 code (before the script suffix) -> (morph_type, source, note)
CLASSIFICATION = {
    # --- Arabic varieties: Semitic, root-and-pattern (introflexive) ---
    "acm": ("introflexive", FAMILY_SOURCE, "Mesopotamian Arabic, Semitic"),
    "acq": ("introflexive", FAMILY_SOURCE, "Ta'izzi-Adeni Arabic, Semitic"),
    "aeb": ("introflexive", FAMILY_SOURCE, "Tunisian Arabic, Semitic"),
    "ajp": ("introflexive", FAMILY_SOURCE, "South Levantine Arabic, Semitic"),
    "apc": ("introflexive", FAMILY_SOURCE, "North Levantine Arabic, Semitic"),
    "arb": ("introflexive", WALS_20A_SOURCE, "Modern Standard Arabic; WALS genus match Arabic (Egyptian)=ablaut/concatenative"),
    "ars": ("introflexive", FAMILY_SOURCE, "Najdi Arabic, Semitic"),
    "ary": ("introflexive", FAMILY_SOURCE, "Moroccan Arabic, Semitic"),
    "arz": ("introflexive", WALS_20A_SOURCE, "Egyptian Arabic; WALS direct: ablaut/concatenative"),
    "heb": ("introflexive", WALS_20A_SOURCE, "Hebrew (Modern); WALS direct: ablaut/concatenative"),
    "amh": ("introflexive", FAMILY_SOURCE, "Amharic, Semitic, templatic + agglutinative affixation"),
    "tir": ("introflexive", FAMILY_SOURCE, "Tigrinya, Semitic"),
    "mlt": ("introflexive", FAMILY_SOURCE, "Maltese, Semitic base despite heavy Romance borrowing"),
    "kab": ("introflexive", WALS_20A_SOURCE, "Kabyle; WALS genus match Berber (Middle Atlas)=ablaut/concatenative"),
    "tzm": ("introflexive", WALS_20A_SOURCE, "Central Atlas Tamazight; WALS genus match Berber=ablaut/concatenative"),
    # --- Cushitic ---
    "gaz": ("agglutinative", FAMILY_SOURCE, "West Central Oromo, Cushitic"),
    # --- Chadic ---
    "hau": ("isolating", WALS_20A_SOURCE, "Hausa; WALS direct: exclusively isolating"),
    # --- Germanic ---
    "eng": ("isolating", FAMILY_SOURCE, "English; most analytic Germanic language, minimal inflectional synthesis"),
    "afr": ("isolating", FAMILY_SOURCE, "Afrikaans; most analytic Germanic language, lost nearly all inflection"),
    "deu": ("fusional", WALS_20A_SOURCE, "German; WALS direct concatenative + traditional fusional case/gender system"),
    "nld": ("fusional", FAMILY_SOURCE, "Dutch, Germanic"),
    "dan": ("fusional", FAMILY_SOURCE, "Danish, Germanic"),
    "swe": ("fusional", FAMILY_SOURCE, "Swedish, Germanic"),
    "nob": ("fusional", FAMILY_SOURCE, "Norwegian Bokmal, Germanic"),
    "nno": ("fusional", FAMILY_SOURCE, "Norwegian Nynorsk, Germanic"),
    "isl": ("fusional", FAMILY_SOURCE, "Icelandic, Germanic, retains rich case system"),
    "fao": ("fusional", FAMILY_SOURCE, "Faroese, Germanic"),
    "ltz": ("fusional", FAMILY_SOURCE, "Luxembourgish, Germanic"),
    "lim": ("fusional", FAMILY_SOURCE, "Limburgish, Germanic"),
    "ydd": ("fusional", FAMILY_SOURCE, "Eastern Yiddish, Germanic"),
    # --- Romance ---
    "fra": ("fusional", WALS_20A_SOURCE, "French; WALS direct: exclusively concatenative + traditional fusional"),
    "spa": ("fusional", WALS_20A_SOURCE, "Spanish; WALS direct: exclusively concatenative + traditional fusional"),
    "por": ("fusional", FAMILY_SOURCE, "Portuguese, Romance"),
    "ita": ("fusional", FAMILY_SOURCE, "Italian, Romance"),
    "cat": ("fusional", FAMILY_SOURCE, "Catalan, Romance"),
    "ron": ("fusional", FAMILY_SOURCE, "Romanian, Romance, retains case marking"),
    "glg": ("fusional", FAMILY_SOURCE, "Galician, Romance"),
    "ast": ("fusional", FAMILY_SOURCE, "Asturian, Romance"),
    "oci": ("fusional", FAMILY_SOURCE, "Occitan, Romance"),
    "fur": ("fusional", FAMILY_SOURCE, "Friulian, Romance"),
    "lij": ("fusional", FAMILY_SOURCE, "Ligurian, Romance"),
    "lmo": ("fusional", FAMILY_SOURCE, "Lombard, Romance"),
    "scn": ("fusional", FAMILY_SOURCE, "Sicilian, Romance"),
    "srd": ("fusional", FAMILY_SOURCE, "Sardinian, Romance"),
    "vec": ("fusional", FAMILY_SOURCE, "Venetian, Romance"),
    "hat": ("isolating", FAMILY_SOURCE, "Haitian Creole; prototypically analytic creole"),
    "pap": ("isolating", FAMILY_SOURCE, "Papiamento; prototypically analytic creole"),
    # --- Slavic ---
    "bel": ("fusional", FAMILY_SOURCE, "Belarusian, Slavic"),
    "bos": ("fusional", FAMILY_SOURCE, "Bosnian, Slavic"),
    "bul": ("fusional", FAMILY_SOURCE, "Bulgarian, Slavic (lost case, but verb morphology remains fusional)"),
    "ces": ("fusional", FAMILY_SOURCE, "Czech, Slavic"),
    "hrv": ("fusional", FAMILY_SOURCE, "Croatian, Slavic"),
    "mkd": ("fusional", FAMILY_SOURCE, "Macedonian, Slavic"),
    "pol": ("fusional", FAMILY_SOURCE, "Polish, Slavic"),
    "rus": ("fusional", WALS_20A_SOURCE, "Russian; WALS direct: exclusively concatenative + traditional fusional"),
    "slk": ("fusional", FAMILY_SOURCE, "Slovak, Slavic"),
    "slv": ("fusional", FAMILY_SOURCE, "Slovenian, Slavic"),
    "srp": ("fusional", FAMILY_SOURCE, "Serbian, Slavic"),
    "ukr": ("fusional", FAMILY_SOURCE, "Ukrainian, Slavic"),
    "szl": ("fusional", FAMILY_SOURCE, "Silesian, Slavic"),
    # --- Baltic ---
    "lit": ("fusional", FAMILY_SOURCE, "Lithuanian, Baltic, archaic case system"),
    "lvs": ("fusional", FAMILY_SOURCE, "Standard Latvian, Baltic"),
    "ltg": ("fusional", FAMILY_SOURCE, "Latgalian, Baltic"),
    # --- Celtic ---
    "cym": ("fusional", FAMILY_SOURCE, "Welsh, Celtic"),
    "gle": ("fusional", FAMILY_SOURCE, "Irish, Celtic"),
    "gla": ("fusional", FAMILY_SOURCE, "Scottish Gaelic, Celtic"),
    # --- Hellenic / Armenian / Albanian ---
    "ell": ("fusional", WALS_20A_SOURCE, "Greek; WALS direct: exclusively concatenative + traditional fusional"),
    "hye": ("agglutinative", FAMILY_SOURCE, "Armenian; classical IE but agglutinative-leaning nominal case system"),
    "als": ("fusional", FAMILY_SOURCE, "Tosk Albanian"),
    # --- Indo-Aryan ---
    "asm": ("fusional", FAMILY_SOURCE, "Assamese, Indo-Aryan"),
    "awa": ("fusional", FAMILY_SOURCE, "Awadhi, Indo-Aryan"),
    "ben": ("fusional", FAMILY_SOURCE, "Bengali, Indo-Aryan"),
    "bho": ("fusional", FAMILY_SOURCE, "Bhojpuri, Indo-Aryan"),
    "guj": ("fusional", FAMILY_SOURCE, "Gujarati, Indo-Aryan"),
    "hin": ("fusional", WALS_20A_SOURCE, "Hindi; WALS direct: exclusively concatenative + traditional fusional"),
    "hne": ("fusional", FAMILY_SOURCE, "Chhattisgarhi, Indo-Aryan"),
    "mag": ("fusional", FAMILY_SOURCE, "Magahi, Indo-Aryan"),
    "mai": ("fusional", FAMILY_SOURCE, "Maithili, Indo-Aryan"),
    "mar": ("fusional", FAMILY_SOURCE, "Marathi, Indo-Aryan"),
    "npi": ("fusional", FAMILY_SOURCE, "Nepali, Indo-Aryan"),
    "ory": ("fusional", FAMILY_SOURCE, "Odia, Indo-Aryan"),
    "pan": ("fusional", FAMILY_SOURCE, "Eastern Panjabi, Indo-Aryan"),
    "san": ("fusional", FAMILY_SOURCE, "Sanskrit, classical Indo-Aryan, archetypal fusional case system"),
    "sin": ("fusional", FAMILY_SOURCE, "Sinhala, Indo-Aryan"),
    "snd": ("fusional", FAMILY_SOURCE, "Sindhi, Indo-Aryan"),
    "urd": ("fusional", FAMILY_SOURCE, "Urdu, Indo-Aryan"),
    # --- Iranian ---
    "ckb": ("fusional", FAMILY_SOURCE, "Central Kurdish (Sorani), Iranian"),
    "kmr": ("fusional", FAMILY_SOURCE, "Northern Kurdish (Kurmanji), Iranian"),
    "pbt": ("fusional", FAMILY_SOURCE, "Southern Pashto, Iranian"),
    "pes": ("fusional", WALS_20A_SOURCE, "Western Persian; WALS genus match Persian=exclusively concatenative + traditional fusional"),
    "prs": ("fusional", FAMILY_SOURCE, "Dari, Iranian"),
    "tgk": ("fusional", FAMILY_SOURCE, "Tajik, Iranian"),
    # --- Turkic ---
    "azb": ("agglutinative", FAMILY_SOURCE, "South Azerbaijani, Turkic"),
    "azj": ("agglutinative", FAMILY_SOURCE, "North Azerbaijani, Turkic"),
    "bak": ("agglutinative", FAMILY_SOURCE, "Bashkir, Turkic"),
    "crh": ("agglutinative", FAMILY_SOURCE, "Crimean Tatar, Turkic"),
    "kaz": ("agglutinative", FAMILY_SOURCE, "Kazakh, Turkic"),
    "kir": ("agglutinative", FAMILY_SOURCE, "Kyrgyz, Turkic"),
    "tat": ("agglutinative", FAMILY_SOURCE, "Tatar, Turkic"),
    "tuk": ("agglutinative", FAMILY_SOURCE, "Turkmen, Turkic"),
    "tur": ("agglutinative", WALS_20A_SOURCE, "Turkish; WALS direct: exclusively concatenative + textbook agglutinative"),
    "uig": ("agglutinative", FAMILY_SOURCE, "Uyghur, Turkic"),
    "uzn": ("agglutinative", FAMILY_SOURCE, "Northern Uzbek, Turkic"),
    # --- Uralic ---
    "est": ("agglutinative", FAMILY_SOURCE, "Estonian, Uralic"),
    "fin": ("agglutinative", WALS_20A_SOURCE, "Finnish; WALS direct: exclusively concatenative + textbook agglutinative"),
    "hun": ("agglutinative", WALS_20A_SOURCE, "Hungarian; WALS direct: exclusively concatenative + textbook agglutinative"),
    # --- Mongolic ---
    "khk": ("agglutinative", FAMILY_SOURCE, "Halh Mongolian; WALS genus Khalkha=exclusively concatenative + textbook agglutinative"),
    # --- Kartvelian / isolates ---
    "kat": ("agglutinative", WALS_20A_SOURCE, "Georgian; WALS direct: exclusively concatenative + textbook agglutinative w/ polypersonal agreement"),
    "eus": ("agglutinative", WALS_20A_SOURCE, "Basque; WALS direct: exclusively concatenative + textbook (highly) agglutinative"),
    # --- Japanese / Korean ---
    "jpn": ("agglutinative", WALS_20A_SOURCE, "Japanese; WALS direct: exclusively concatenative + textbook agglutinative"),
    "kor": ("agglutinative", WALS_20A_SOURCE, "Korean; WALS direct: exclusively concatenative + textbook agglutinative"),
    # --- Sino-Tibetan ---
    "yue": ("isolating", FAMILY_SOURCE, "Yue Chinese (Cantonese), Sinitic"),
    "zho": ("isolating", WALS_20A_SOURCE, "Mandarin Chinese; WALS direct: isolating/concatenative"),
    "mya": ("isolating", WALS_20A_SOURCE, "Burmese; WALS direct: exclusively concatenative but textbook isolating; low synthesis (22A)"),
    "bod": ("agglutinative", FAMILY_SOURCE, "Standard Tibetan; agglutinative case/verb particles"),
    "dzo": ("agglutinative", FAMILY_SOURCE, "Dzongkha, Bodish, agglutinative like Tibetan"),
    "mni": ("agglutinative", FAMILY_SOURCE, "Meitei, Sino-Tibetan (Kuki-Chin-Meitei), agglutinative verb morphology"),
    "kac": ("agglutinative", FAMILY_SOURCE, "Jingpho, Sino-Tibetan, agglutinative"),
    "lus": ("isolating", FAMILY_SOURCE, "Mizo, Kuki-Chin, tonal/isolating like other Kuki-Chin languages"),
    # --- Tai-Kadai ---
    "tha": ("isolating", WALS_20A_SOURCE, "Thai; WALS direct: isolating/concatenative"),
    "lao": ("isolating", FAMILY_SOURCE, "Lao, Tai-Kadai"),
    "shn": ("isolating", FAMILY_SOURCE, "Shan, Tai-Kadai"),
    # --- Austroasiatic ---
    "khm": ("isolating", FAMILY_SOURCE, "Khmer, Austroasiatic (Mon-Khmer)"),
    "vie": ("isolating", WALS_20A_SOURCE, "Vietnamese; WALS direct: exclusively isolating"),
    "sat": ("agglutinative", FAMILY_SOURCE, "Santali, Munda branch of Austroasiatic; Munda languages are the agglutinative outlier within AA"),
    "khs": ("isolating", WALS_20A_SOURCE, "Khasi (family reference for Austroasiatic isolating type); WALS direct: exclusively isolating"),
    # --- Hmong-Mien ---
    # (none directly in FLORES-200 list)
    # --- Dravidian ---
    "kan": ("agglutinative", WALS_20A_SOURCE, "Kannada; WALS direct: exclusively concatenative + textbook agglutinative"),
    "mal": ("agglutinative", FAMILY_SOURCE, "Malayalam, Dravidian"),
    "tam": ("agglutinative", FAMILY_SOURCE, "Tamil, Dravidian"),
    "tel": ("agglutinative", FAMILY_SOURCE, "Telugu, Dravidian"),
    # --- Austronesian: Philippine-type (rich voice/focus affixation) -> agglutinative ---
    "ceb": ("agglutinative", FAMILY_SOURCE, "Cebuano, Philippine Austronesian"),
    "ilo": ("agglutinative", FAMILY_SOURCE, "Ilocano, Philippine Austronesian"),
    "pag": ("agglutinative", FAMILY_SOURCE, "Pangasinan, Philippine Austronesian"),
    "tgl": ("agglutinative", WALS_20A_SOURCE, "Tagalog; WALS direct: exclusively concatenative + textbook agglutinative voice system"),
    "war": ("agglutinative", FAMILY_SOURCE, "Waray, Philippine Austronesian"),
    # --- Austronesian: Malayic / western Malayo-Polynesian -> isolating (per Indonesian proxy) ---
    "ace": ("isolating", FAMILY_SOURCE, "Acehnese, Malayo-Chamic Austronesian"),
    "ban": ("isolating", FAMILY_SOURCE, "Balinese, Austronesian"),
    "bjn": ("isolating", FAMILY_SOURCE, "Banjar, Malayic Austronesian"),
    "bug": ("isolating", FAMILY_SOURCE, "Buginese, Austronesian"),
    "ind": ("isolating", WALS_20A_SOURCE, "Indonesian; WALS direct: exclusively isolating"),
    "jav": ("isolating", FAMILY_SOURCE, "Javanese, Austronesian"),
    "min": ("isolating", FAMILY_SOURCE, "Minangkabau, Malayic Austronesian"),
    "sun": ("isolating", FAMILY_SOURCE, "Sundanese, Austronesian"),
    "zsm": ("isolating", FAMILY_SOURCE, "Standard Malay, sister of Indonesian"),
    # --- Austronesian: Oceanic / Polynesian ---
    "fij": ("isolating", WALS_20A_SOURCE, "Fijian; WALS direct: exclusively isolating"),
    "mri": ("isolating", WALS_20A_SOURCE, "Maori; WALS direct: isolating/concatenative"),
    "smo": ("isolating", FAMILY_SOURCE, "Samoan, Polynesian, like Maori/Fijian"),
    "tpi": ("isolating", FAMILY_SOURCE, "Tok Pisin; prototypically analytic English-lexifier creole"),
    "plt": ("agglutinative", WALS_20A_SOURCE, "Plateau Malagasy; WALS genus match Malagasy=exclusively concatenative + textbook agglutinative verbal morphology"),
    # --- Niger-Congo: Bantu -> agglutinative ---
    "bem": ("agglutinative", FAMILY_SOURCE, "Bemba, Bantu"),
    "cjk": ("agglutinative", FAMILY_SOURCE, "Chokwe, Bantu"),
    "kam": ("agglutinative", FAMILY_SOURCE, "Kamba, Bantu"),
    "kik": ("agglutinative", FAMILY_SOURCE, "Kikuyu, Bantu"),
    "kin": ("agglutinative", FAMILY_SOURCE, "Kinyarwanda, Bantu"),
    "kmb": ("agglutinative", FAMILY_SOURCE, "Kimbundu, Bantu"),
    "kon": ("agglutinative", FAMILY_SOURCE, "Kikongo, Bantu"),
    "lin": ("agglutinative", FAMILY_SOURCE, "Lingala, Bantu"),
    "lua": ("agglutinative", FAMILY_SOURCE, "Luba-Kasai, Bantu"),
    "lug": ("agglutinative", FAMILY_SOURCE, "Ganda, Bantu"),
    "nso": ("agglutinative", FAMILY_SOURCE, "Northern Sotho, Bantu"),
    "nya": ("agglutinative", FAMILY_SOURCE, "Nyanja, Bantu"),
    "run": ("agglutinative", FAMILY_SOURCE, "Rundi, Bantu"),
    "sna": ("agglutinative", FAMILY_SOURCE, "Shona, Bantu"),
    "sot": ("agglutinative", FAMILY_SOURCE, "Southern Sotho, Bantu"),
    "ssw": ("agglutinative", FAMILY_SOURCE, "Swati, Bantu"),
    "swh": ("agglutinative", WALS_20A_SOURCE, "Swahili; WALS direct: exclusively concatenative + textbook agglutinative noun-class system"),
    "tsn": ("agglutinative", FAMILY_SOURCE, "Tswana, Bantu"),
    "tso": ("agglutinative", FAMILY_SOURCE, "Tsonga, Bantu"),
    "tum": ("agglutinative", FAMILY_SOURCE, "Tumbuka, Bantu"),
    "umb": ("agglutinative", FAMILY_SOURCE, "Umbundu, Bantu"),
    "xho": ("agglutinative", FAMILY_SOURCE, "Xhosa, Bantu"),
    "zul": ("agglutinative", WALS_20A_SOURCE, "Zulu; WALS direct: exclusively concatenative + textbook agglutinative noun-class system"),
    "luo": ("agglutinative", FAMILY_SOURCE, "Luo, Nilotic (Western Nilotic), agglutinative verbal morphology"),
    # --- Niger-Congo: Kwa / Kru / Gur / Mande / Atlantic -> mostly isolating (West African analytic belt) ---
    "aka": ("isolating", FAMILY_SOURCE, "Akan, Kwa, analytic/tonal"),
    "bam": ("isolating", FAMILY_SOURCE, "Bambara, Mande, prototypically isolating"),
    "dyu": ("isolating", FAMILY_SOURCE, "Dyula, Mande"),
    "ewe": ("isolating", FAMILY_SOURCE, "Ewe, Kwa, analytic/tonal"),
    "fon": ("isolating", FAMILY_SOURCE, "Fon, Gbe/Kwa, analytic/tonal"),
    "ibo": ("isolating", FAMILY_SOURCE, "Igbo, Igboid, analytic/tonal"),
    "kbp": ("agglutinative", FAMILY_SOURCE, "Kabiyah, Gur, noun-class agglutinative morphology"),
    "mos": ("isolating", FAMILY_SOURCE, "Mossi, Gur"),
    "twi": ("isolating", FAMILY_SOURCE, "Twi (Akan), Kwa"),
    "yor": ("isolating", WALS_20A_SOURCE, "Yoruba; WALS direct: tonal/isolating"),
    "grb": ("isolating", FAMILY_SOURCE, "Grebo (Kru genus WALS match), analytic/tonal"),
    "wol": ("agglutinative", FAMILY_SOURCE, "Wolof, Atlantic Niger-Congo, agglutinative noun-class system"),
    "fuv": ("agglutinative", FAMILY_SOURCE, "Nigerian Fulfulde, Atlantic Niger-Congo, agglutinative noun-class system"),
    "sag": ("isolating", WALS_20A_SOURCE, "Sango; WALS direct: exclusively concatenative but textbook analytic Ubangi-based creole/pidgin-origin lingua franca"),
    # --- Nilo-Saharan / Nilotic -> fusional (ablaut/tonal fusion per WALS genus matches for Lango/Nandi/Maasai) ---
    "dik": ("fusional", WALS_20A_SOURCE, "Southwestern Dinka; WALS genus match Western Nilotic (Lango)=ablaut/concatenative"),
    "nus": ("fusional", WALS_20A_SOURCE, "Nuer; WALS genus match Western Nilotic (Lango)=ablaut/concatenative"),
    # --- Saharan (Kanuri) ---
    "knc": ("agglutinative", FAMILY_SOURCE, "Central Kanuri, Saharan (Nilo-Saharan), agglutinative"),
    # --- Andean ---
    "ayr": ("agglutinative", FAMILY_SOURCE, "Central Aymara, Aymaran; highly agglutinative"),
    "quy": ("agglutinative", FAMILY_SOURCE, "Ayacucho Quechua, Quechuan; highly agglutinative (not strictly polysynthetic)"),
    "grn": ("agglutinative", WALS_20A_SOURCE, "Guarani; WALS direct: exclusively concatenative; textbook agglutinative w/ some polysynthetic traits"),
    # --- Constructed ---
    "epo": ("agglutinative", FAMILY_SOURCE, "Esperanto; designed as a regular agglutinative language"),
    # --- Isolates / unclassified-typology-uncertain ---
    "kea": ("isolating", FAMILY_SOURCE, "Kabuverdianu; prototypically analytic Portuguese-lexifier creole"),
    # --- previously missing entries (added on review, not genuinely uncertain) ---
    "kas": ("fusional", FAMILY_SOURCE, "Kashmiri, Indo-Aryan (Dardic branch)"),
    "som": ("agglutinative", FAMILY_SOURCE, "Somali, Cushitic; rich agglutinative nominal/verbal morphology"),
    "taq": ("introflexive", FAMILY_SOURCE, "Tamasheq, Berber (Tuareg branch); root-and-pattern like other Berber languages"),
}

TYPE_ORDER = ["agglutinative", "fusional", "introflexive", "isolating", "polysynthetic", "unclassified"]


def load_languages(path):
    langs = []
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            name, code = line.split("\t")
            base = code.split("_")[0]
            langs.append({"name": name, "flores_code": code, "base_code": base})
    return langs


def classify(langs):
    for lang in langs:
        entry = CLASSIFICATION.get(lang["base_code"])
        if entry is None:
            lang["morph_type"] = "unclassified"
            lang["source"] = None
            lang["note"] = "no classification available in this pass; needs manual sourcing before use"
        else:
            morph_type, source, note = entry
            lang["morph_type"] = morph_type
            lang["source"] = source
            lang["note"] = note
    return langs


def merge_json(path, new_data):
    """Read-merge-write: never blind-overwrite an existing SSOT JSON."""
    existing = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}
    existing.update(new_data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    langs = load_languages(DATA_TSV)
    langs = classify(langs)

    n_raw = len(langs)  # 204 flores-code entries
    base_codes_seen = {}
    for lang in langs:
        base_codes_seen.setdefault(lang["base_code"], lang)
    n_collapsed = len(base_codes_seen)  # collapsed same-language script duplicates

    def summarize(entries):
        counts = Counter(e["morph_type"] for e in entries)
        total = len(entries)
        n_complex = sum(v for k, v in counts.items() if k in COMPLEX_TYPES)
        n_unclassified = counts.get("unclassified", 0)
        n_classified = total - n_unclassified
        return {
            "total_languages": total,
            "counts_by_type": {t: counts.get(t, 0) for t in TYPE_ORDER},
            "n_unclassified": n_unclassified,
            "n_classified": n_classified,
            "n_morphologically_complex": n_complex,
            "pct_complex_of_all": round(100 * n_complex / total, 1),
            "pct_complex_of_classified": round(100 * n_complex / n_classified, 1) if n_classified else None,
        }

    raw_summary = summarize(langs)
    collapsed_summary = summarize(list(base_codes_seen.values()))

    out = {
        "nllb_morphology_stats": {
            "generated_by": "scripts/count_nllb_morphological_complexity.py",
            "definition_of_complex": sorted(COMPLEX_TYPES),
            "definition_note": "complex = everything except isolating/analytic; see COMPLEX_TYPES in script to change",
            "raw_204_flores_entries": raw_summary,
            "collapsed_by_base_iso639_3_code": collapsed_summary,
            "headline_number_caveat": (
                "NLLB-200's marketed '200 languages' does not exactly match either "
                "the 204 raw FLORES-200 entries or our 196-entry base-code collapse; "
                "see script docstring. Use collapsed_by_base_iso639_3_code as the "
                "closer proxy to '200 languages', but disclose n=196 (or n=204), not 200."
            ),
            "per_language": [
                {
                    "name": l["name"],
                    "flores_code": l["flores_code"],
                    "base_code": l["base_code"],
                    "morph_type": l["morph_type"],
                    "source": l["source"],
                    "note": l["note"],
                }
                for l in langs
            ],
            "sources": [
                "Team, NLLB, et al. (2022). 'No Language Left Behind: Scaling Human-Centered Machine Translation.' arXiv:2207.04672.",
                "facebookresearch/flores, flores200/README.md (language table), fetched 2026-08-19.",
                "Dryer, Matthew S. & Haspelmath, Martin (eds.) 2013. WALS Online v2020.3, Chapter 20A 'Fusion of Selected Inflectional Formatives'. https://wals.info/chapter/20",
                "Comrie, B. (1989). Language Universals and Linguistic Typology, 2nd ed.",
                "Velupillai, V. (2012). An Introduction to Linguistic Typology.",
            ],
            "limitations": [
                "WALS 20A distinguishes fusion type (isolating/concatenative/tonal/ablaut) but NOT agglutinative vs. fusional among concatenative languages; that distinction (WALS 21A, exponence) was not fetched for this pass, so most agglutinative/fusional calls rely on family-level typological consensus, not a WALS datapoint for that specific language.",
                "Only entries tagged with a WALS_20A source string were checked against an actual WALS datapoint; all others are genus/family-level consensus calls made by an LLM agent and are not independently human-verified.",
                "Mapudungun (arn), Nahuatl (nah), and Inuktitut (ikt), cited in plan.md as primary polysynthetic languages, are NOT present in the official FLORES-200 list used here.",
                "Categorical typology is a simplification; several languages (English, Georgian, Basque, Bengali) are borderline/mixed and reasonable typologists could disagree with the single label assigned here.",
            ],
        }
    }

    merge_json(OUT_JSON, out)

    s = collapsed_summary
    print(f"Raw FLORES-200 entries: {raw_summary['total_languages']}")
    print(f"Collapsed (base ISO 639-3) languages: {s['total_languages']}")
    print(f"Unclassified: {s['n_unclassified']}")
    print(f"Morphologically complex (of {s['total_languages']}): {s['n_morphologically_complex']} "
          f"({s['pct_complex_of_all']}% of all, {s['pct_complex_of_classified']}% of classified)")
    print(f"By type: {s['counts_by_type']}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
