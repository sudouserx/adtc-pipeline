#!/usr/bin/env python3
"""Kuza English -> Kenyan Swahili SFT translation with strict QC.

Translates the Kuza agricultural SFT dataset with Groq
`openai/gpt-oss-120b`.

Source dataset:
https://huggingface.co/datasets/kuzaai/agri_sft_prod_dedup_25k

Source file:
gemma4_agri_sft_25k.jsonl

Requires:
GROQ_API_KEY

Optional for private Hugging Face access:
HF_TOKEN
HUGGINGFACE_HUB_TOKEN

Dependencies:
huggingface_hub
groq
tqdm

Two execution modes (`--mode`):

online (default) - concurrent synchronous chat-completion calls, retried
with a growing token budget and QC feedback. Good for small runs and for
finishing off whatever a batch run didn't.

batch - submits the work through Groq's Batch API first (50% cheaper,
its own separate rate-limit pool), then finishes any rows the batch
left unresolved through the online path above.

The run is resumable: every translated row that passes QC is appended
to a checkpoint file (`<output>.progress.jsonl`) as soon as it's done.
`--revalidate` re-runs the current QC rules over that checkpoint and
re-translates any row that no longer passes; use it after tightening
the QC lists below instead of paying for a full `--fresh` re-run.

Rows that fail even after retries are written to:
<output>.failed.jsonl

Batch state is stored in:
<output>.batch_state.json

The final translated dataset is written to:
kuza_agri_sft_swahili.jsonl

QC in this version:
- Normalizes malformed unit patterns such as:
  `2 kg ha 1` -> `2 kg/ha`
  `50 g m 2` -> `50 g/m²` (rule ordering fixed; previously `50 g m²`)
  `5 km h 1` -> `5 km/h`
  `2-4 L m 1 h 1` -> `2-4 L/m/h`
  `Ficusspp.` -> `Ficus spp.`
- Strips mathematically wrong acre->kg/ha parentheticals from the
  English source before translation (e.g. `50 kg / acre ( 12 kg / ha)`)
  so the bogus conversion stops leaking into the Swahili.
- Fails rows that lose numerals present in the English source.
- Fails rows containing known mistranslations and invented or hybrid
  forms (mapopo, wanyama wa asili, kupiga mayai, weka vibaya, ugali wa
  maji, komposti ya mashine, safi safi, msimu ya, kununyizia,
  kalibrated, sanitiza, ideali, fungashio, yayeyesha, kibio, manyu,
  spra, iandamizwe, ...).
- Fails rows containing untranslated English terms, now including time
  and spacing words (years, weeks, yr, apart) and common nouns (silage,
  loam, slope, solution, syrup, tray, rations, legume, mask, ...).
  Every blocklisted term now has a glossary replacement so the model is
  pushed toward real Kiswahili instead of coinages like `spra`.
- Fails rows that add imperial/unit conversions not present in English.
- Fails rows that expand too much compared with the English source.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from groq import (
    APIConnectionError,
    APIStatusError,
    AsyncGroq,
    AuthenticationError,
    BadRequestError,
    Groq,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from tqdm.auto import tqdm


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SOURCE_REPO = "kuzaai/agri_sft_prod_dedup_25k"
SOURCE_FILENAME = "gemma4_agri_sft_25k.jsonl"

TRANSLATOR_MODEL = "openai/gpt-oss-120b"
REASONING_EFFORT = "medium"
REASONING_RESERVE_TOKENS = 4096
MIN_OUTPUT_TOKENS = 4096
MAX_OUTPUT_TOKENS = 16384

ONLINE_CONCURRENCY = 6
DEFAULT_OUTPUT_NAME = "kuza_agri_sft_swahili.jsonl"

# Non-retryable errors.
FATAL_EXCEPTIONS = (
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
)

# Groq Batch configuration.
BATCH_CHUNK_SIZE = 1000
BATCH_COMPLETION_WINDOW = "24h"
BATCH_POLL_SECONDS = 30
BATCH_TERMINAL_STATUSES = {
    "completed",
    "failed",
    "expired",
    "cancelled",
}

# QC length guards. Swahili can be somewhat longer than English, but the
# original prompt requires compact output for a 1024-token edge model.
QC_MAX_ASSISTANT_RATIO = 1.8
QC_MAX_ASSISTANT_PADDING = 256
QC_MAX_USER_RATIO = 2.6
QC_MAX_USER_PADDING = 140

# Translation retry policy.
# 1 initial attempt + 2 QC correction attempts.
MAX_QC_RETRIES = 2

# Transient API failures can be retried independently.
MAX_TRANSIENT_RETRIES = 5


# --------------------------------------------------------------------------
# Translation glossary
# --------------------------------------------------------------------------
#
# NOTE: every term that appears in QC_ENGLISH_TERMS (the hard blocklist)
# must have a usable Swahili replacement here. Blocklisting an English
# word without giving the model a replacement pushes it into invented
# escape forms (spra, ideali, manyu, kalibrated, sanitiza, Ficusspp).

GLOSSARY: list[tuple[str, str]] = [
    ("calf / calves", "ndama"),
    ("heifer", "ndama jike / ng'ombe jike mchanga"),
    (
        "steam a heifer / steaming up",
        "kuongeza lishe kabla ya kujifungua; si kuchemsha",
    ),
    ("bull", "fahali / ng'ombe dume"),
    ("cow / dairy cow", "ng'ombe / ng'ombe wa maziwa"),
    ("sheep / ewe / ram", "kondoo; si mbuzi"),
    ("goat", "mbuzi; si kondoo"),
    ("pig / piglet / sow", "nguruwe / mdogo wa nguruwe; si kifaranga"),
    (
        "rabbit / doe (rabbit)",
        "sungura / sungura jike; si kifaranga, si bundi",
    ),
    ("chick / chicks", "kifaranga / vifaranga; si chick au kijana"),
    ("chicken / poultry", "kuku"),
    ("hive / apiary", "mzinga / mizinga / apiari; si shimo wala mashimo"),
    ("breed / breeds", "aina / mifugo; si mbadala, si bundi"),
    ("calved / freshened", "alijifungua; si alizaliwa"),
    ("vaccinate / vaccination", "chanja / chanjo; si kupandikiza"),
    ("colostrum", "kolostramu / maziwa ya kwanza"),
    ("mastitis", "mastitis / uvimbe wa matiti"),
    ("milking / milking parlor", "kukama / chumba cha kukama; si kufuga"),
    ("BCS / Body Condition Score", "BCS (Body Condition Score)"),
    ("lactation", "kipindi cha kutoa maziwa / unyonyeshaji; si kikombe"),
    ("silage", "silaji; si silage"),
    ("Napier / nappier", "nyasi ya Napier"),
    ("Brachiaria", "Brachiaria"),
    ("dairy meal", "dairy meal / unga wa maziwa"),
    ("concentrate (dairy feed)", "concentrate / chakula cha nyongeza"),
    ("premix", "premix / mchanganyiko wa virutubisho"),
    ("maize germ", "kiini cha mahindi; si mtama"),
    ("sunflower cake", "keki ya alizeti; si keki ya jua"),
    ("fish meal", "unga wa samaki; si samaki mkunjo"),
    ("CAN fertilizer", "mbolea ya CAN"),
    ("NPK", "NPK"),
    (
        "fertilizer application",
        "kuweka / kutumia mbolea; si maombi",
    ),
    ("teat dip", "dawa ya kuzamisha chuchu"),
    ("navel / umbilical cord", "kitovu"),
    ("iodine", "iodini"),
    ("AI / artificial insemination", "ufugaji bandia (AI)"),
    ("weaning", "kuachisha kunyonya"),
    ("dry period", "kipindi cha kavu"),
    ("crop rotation", "mzunguko wa mazao"),
    ("intercropping", "kilimo mchanganyiko"),
    ("mulching / mulch", "kufunika udongo; si mulch, si fungashio"),
    ("drip irrigation", "umwagiliaji wa matone"),
    (
        "spray / spraying",
        "kunyunyizia / kupulizia; si kunanyua, si kunyonya, "
        "si kununua, si kununyizia",
    ),
    ("extension officer", "afisa wa ugani"),
    (
        "veterinarian / veterinary officer",
        "daktari wa mifugo; si afisa wa ugani",
    ),
    (
        "KALRO / Kenya Agricultural and Livestock Research Organization",
        "KALRO tu; si jina kamili la Kiingereza",
    ),
    ("pyrethrum", "pirethramu; tafsiri neno kila mara"),
    ("coffee berry disease / CBD", "CBD (Coffee Berry Disease)"),
    ("coffee leaf rust", "ukungu wa majani ya kahawa"),
    (
        "coffee beans / coffee berries",
        "buni / matunda ya kahawa; si maziwa ya kahawa, si karanga",
    ),
    (
        "coffee farming",
        "kilimo cha kahawa; si ufugaji wa kahawa",
    ),
    (
        "Ruiru 11 / SL28 / SL34 / Batian",
        "hifadhi majina bila kutafsiri",
    ),
    (
        "H614D / DK8031",
        "hifadhi misimbo bila kutafsiri",
    ),
    (
        "Black Soldier Fly / BSF",
        "Black Soldier Fly (BSF)",
    ),
    (
        "BSF larvae / larvae (BSF)",
        "mabuu / vifaranga vya BSF; si mapopo (mapopo = popo)",
    ),
    ("water hyacinth", "kiambiti / hyacinth ya maji"),
    ("probiotics", "probiotics"),
    ("molasses", "molasi; si molasses"),
    ("starter feed", "chakula cha mwanzo; si ugali wa mwanzo"),
    ("grower feed", "chakula cha kukuza"),
    (
        "fungicide",
        "dawa ya ukungu / dawa ya fangasi; neno fungishia halipo",
    ),
    (
        "biofungicide",
        "dawa ya ukungu ya kibiolojia; si biofungicide",
    ),
    ("copper-based", "yenye shaba"),
    (
        "copper sulfate / copper sulphate",
        "sulfate ya shaba; si sukari ya shaba",
    ),
    (
        "hydrated lime",
        "chokaa kilichotiwa maji; si majivu ya lime",
    ),
    (
        "wood shavings",
        "visusi / vipande vya mbao; si mchanga wa mbao",
    ),
    ("bedding / litter", "matandiko"),
    ("pen / hutch", "zizi / banda; si peni"),
    ("dip (navel or teat)", "chovya / zamisha; si chomeka"),
    (
        "lesion / lesions",
        "kidonda / vidonda / madoa; si maumivu",
    ),
    ("necrosis", "kufa kwa tishu"),
    ("bud / buds", "chipukizi / vitumba; si viini"),
    (
        "cultural control",
        "udhibiti kwa mbinu za kilimo; si kitamaduni",
    ),
    (
        "certified disease-free planting material",
        "vikonyo / miche iliyothibitishwa isiyo na magonjwa",
    ),
    (
        "vine / vines (sweet potato)",
        "vikonyo / mashina",
    ),
    (
        "mould / mouldy feed",
        "ukungu / chakula chenye ukungu",
    ),
    ("aphids", "vidukari"),
    ("whiteflies", "inzi weupe"),
    ("thrips", "wadudu wembamba; si thrips"),
    (
        "leafminer / tomato leafminer (Tuta absoluta)",
        "dudu la kuchimba majani (Tuta absoluta); si leafminer",
    ),
    ("stem borer", "mdudu wa shina; si stem borer, si mchafu"),
    (
        "Fall Armyworm",
        "mdudu wa majani ya mahindi; si Fall Armyworm, si shinyanga",
    ),
    ("coffee berry borer / CBB", "dudu la buni (CBB); si borer"),
    ("early blight", "ukungu wa mapema; si early blight"),
    ("late blight", "ukungu wa baadaye; si late blight"),
    ("bacterial wilt", "unyauke wa bakteria; si bacterial wilt"),
    ("blight", "ukungu; si blight"),
    ("wilt", "unyauke; si wilt"),
    ("lady beetle / ladybird", "mende wa bwana / ladybird; si beetle wa kike"),
    ("parasitic wasps", "nyigu wa parasiti / nyigu wadogo; si nyuki"),
    ("mites", "vidudu vidogo; si mites"),
    ("pupae / pupa", "bukuu / mabukuu; si pupae"),
    ("maggot / maggots", "funza; si magugu"),
    ("soil test", "upimaji wa udongo; si majaribio"),
    (
        "acidity (soil)",
        "asidi ya udongo; si uchungu",
    ),
    ("dry matter", "malisho kavu / dry matter"),
    (
        "compost / composting",
        "komposti / kutengeneza komposti; si mbolea ya maziwa",
    ),
    (
        "well-rotted compost / manure",
        "komposti / samadi iliyooza vizuri; si iliyochomwa, si iliyoyeyuka",
    ),
    ("farmyard manure", "samadi ya banda; si komposti ya mashine"),
    (
        "resistant varieties",
        "aina zinazostahimili; si aina za kudumu",
    ),
    (
        "shallow feeder",
        "chombo kifuupi / kisicho kirefu",
    ),
    (
        "inexpensive / cheap",
        "bei nafuu / si ghali; usiseme ni ghali",
    ),
    (
        "dying / dies",
        "kufa / zinakufa; si kufika/zinafika",
    ),
    (
        "plants dry out",
        "kukauka; si inuka",
    ),
    (
        "mucus (nasal / livestock)",
        "ute / kamasi; si ute wa mkojo isipokuwa mkojo umetajwa",
    ),
    (
        "seedling / seedlings",
        "miche; si viumbe, si kiumbe",
    ),
    (
        "spacing / plant intervals",
        "umbali; si vipindi vya muda",
    ),
    (
        "late evening",
        "jioni; si jioni ghafla",
    ),
    (
        "nursing period (mammal)",
        "kipindi cha kunyonya; si kupandikiza",
    ),
    (
        "rabbit kits / young rabbits",
        "watoto wa sungura; si vifaranga",
    ),
    (
        "AI / vet semen",
        "shahawa / semen ya AI; si semen ya daktari",
    ),
    ("Botrytis", "Botrytis; hifadhi jina la kilatini"),
    ("Rhizoctonia", "Rhizoctonia; hifadhi jina la kilatini"),
    ("Trichoderma", "Trichoderma; hifadhi jina la kilatini"),
    ("IPM", "IPM"),
    (
        "copper oxychloride / copper hydroxide",
        "copper oxychloride / copper hydroxide; si copper peke yake",
    ),

    # Additional glossary entries added from earlier QC review.
    (
        "droppings / dung / manure",
        "kinyesi / mtombo / samadi; si mchana",
    ),
    (
        "nasal discharge",
        "ute wa pua / kamasi ya pua; si kinyesi la pua",
    ),
    (
        "brooding",
        "utunzaji wa vifaranga / joto la vifaranga; si ukandamizaji",
    ),
    (
        "prune / pruning",
        "kata / pogoa / kupogoa; si panda",
    ),
    (
        "coffee parchment / coffee husk",
        "buni / ganda la kahawa; si maziwa ya kahawa",
    ),
    (
        "wheat bran",
        "chemba za ngano; si mchanga wa ngano",
    ),
    ("bran (maize / wheat / general)", "chemba; si bran"),
    (
        "seed dressing / seed coating",
        "utibabu wa mbegu / kifuniko cha mbegu; si mnyororo",
    ),
    (
        "raised beds",
        "matuta yaliyoinuliwa; si vitongoji",
    ),
    (
        "fermentation / ensiling",
        "uchachushaji / ufermentaji; si ichele",
    ),
    (
        "ensiling / silage making",
        "utengenezaji wa silaji; si kunyunyizia",
    ),
    (
        "cull",
        "ondoa / zua; si kataa",
    ),
    (
        "foot bath",
        "bakuli la miguu / kuoga miguu",
    ),
    (
        "lymph nodes",
        "tezi za limfu; si node",
    ),
    (
        "bacterial culture",
        "ukuaji wa bakteria / kultcha ya bakteria; si utamaduni",
    ),
    ("shade / shading", "kivuli; si kinyume"),
    ("bacterial ooze", "ute wa bakteria; si mto"),
    ("swarming (bees)", "kuunguka; si kuja kwa roho"),
    ("layer (poultry)", "kuku wa mayai; si ukelea"),
    ("varieties", "aina; si mavazi"),
    ("absorb", "fyonza; si nyunyizia"),
    ("palpate / feel", "gusa / kagua; si piga mgongo"),
    ("flooded / submerge", "zagaza / funika na maji; si fukuza"),
    ("plant (verb)", "panda; si ponya"),
    ("grind / mill", "saga; si sasa"),
    ("deworm", "toa dawa ya minyoo / dawa ya vimelea; si fungua"),
    ("dry cows", "ng'ombe walio katika kipindi cha kavu; si ng'ombe kavu"),
    ("tasseling", "kutoa maua / kutoa pua; si tasseling"),
    ("neem oil", "mafuta ya neem / mafuta ya mwarobaini; si neem oil"),

    # Mistranslation traps found in the first full pass.
    ("natural enemies (biological control)", "maadui wa asili; si wanyama wa asili"),
    ("lay eggs", "kutaga mayai; si kupiga mayai"),
    ("dissolve", "yeyusha; si yayeyesha"),
    ("non-host crop", "mazao yasiyobeba ugonjwa huo; si yasiyo wa wageni"),
    ("weevil / banana weevil", "dudu la ndizi (weevil); si weevili"),
    ("larvae / larva (general)", "mabuu / bukuu; si larvae"),
    ("nursery / nurseries (seedlings)", "zizi la miche / kituo cha miche; si mabwawa"),

    # Equipment / input terms (blocklisted in English, so a replacement
    # is mandatory here).
    ("nozzle / nozzles", "kichwa cha kunyunyizia; si nozzle, si manyu"),
    ("sprayer", "ombo la kunyunyizia; si sprayer, si spra"),
    ("sprinkler / sprinklers", "mnyunyizio; si sprinkler"),
    ("PPE", "vifaa vya kinga; si PPE"),
    ("mask (respirator)", "barakoa; si mask"),
    ("herbicide", "dawa ya magugu; si herbicide"),
    ("trellis / staking", "nguzo / fito za kuegemeza; si trellis"),
    ("teat sealant", "kifuniko cha chuchu; si sealant"),
    ("tray / trays", "tabo / sini; si tray"),
    ("strips (e.g. formic acid)", "vipande; si strips"),
    ("dryer / solar dryer", "kikaushio / kikaushio cha jua; si dryer"),
    ("silo / silos", "siloo; si silo, si silos"),
    ("super / supers (beehive)", "sanduku la asali (super); si super"),
    ("first-flush diverter", "kifaa cha kuepusha maji ya kwanza; si diverter"),
    ("smoker (beekeeping)", "smoker / chombo cha moshi"),
    ("bleach", "bleach / JIK (dawa ya kusafisha)"),

    # Agronomy / feed / process terms.
    ("starch", "wanga; si starch"),
    ("loam", "udongo mchanganyiko; si loam"),
    ("slope", "mteremko; si slope"),
    ("tillering", "kutoa mashada; si tillering"),
    ("ration / rations (feed)", "lishe / chakula kilichopangwa; si rations"),
    ("legume / legumes", "mazao ya jamii ya mkunde (maharagwe, njano); si legume"),
    ("solution (chemical)", "suluhisho; si solution"),
    ("syrup / sugar syrup", "sirupu ya sukari; si syrup"),
    ("active ingredient", "kipengele kinachotenda; si active"),
    ("calibrated (equipment)", "zilizopimwa / kalibritwa; si calibrated"),
    ("batch (group)", "kundi; si batch"),
    ("regenerative (agriculture)", "kilimo cha kufufua ubora wa udongo; si regenerative"),
    ("somatic cell count", "hesabu ya seli za mwili; si somatic"),
    ("anticoccidial", "dawa ya kuzuia kokidiasi; si anticoccidial"),
    ("paddy / paddy rice", "mpunga uliovunwa (paddy); si paddy rice"),
    ("ideal", "inayopendekezwa; si ideal, si ideali"),
    ("keep (maintain)", "dumisha / endelea; si keep"),
    (
        "time words: years / weeks / months / days / hours",
        "miaka / wiki / miezi / siku / masaa; si years, weeks, yr, months, "
        "days, hours",
    ),
    ("apart (spacing)", "mbali / vigae; si apart"),
]

GLOSSARY_BLOCK = "\n".join(
    f"- {english} -> {swahili}"
    for english, swahili in GLOSSARY
)


# --------------------------------------------------------------------------
# QC blocklists
# --------------------------------------------------------------------------

# These are known bad Swahili forms or wrong domain terms observed in QC.
QC_BAD_PHRASES: list[str] = [
    "jioni ghafla",
    "fungishia",
    "vichepeto",
    "formuleni",
    "formuliza",
    "imilisha",
    "kunanyua",
    "kunyonya majani",
    "kunyonya mafuta",
    "kunyonyesha majani",
    "kunyonyesha mafuta",
    "kinyesi la pua",
    "kinyesi cha pua",
    "mchana wa kuku",
    "mchana una damu",
    "usafishaji wa mchana",
    "maziwa ya kahawa",
    "karanga za kahawa",
    "ubora wa karanga",
    "majaribio ya udongo",
    "ugali wa mwanzo",
    "chovya ya chipukizi",
    "viumbe vya nyanya",
    "viumbe vikaribie",
    "viumbe vizuri",
    "mtandaoni",
    "piga sungura",
    "majani ya majani",
    "kaanga vipande jua",
    "kaanga vipande",
    "nanga larvae",
    "straw mulch",
    "paddy rice",
    "formuliza rations",
    "mchanga wa ngano",
    "vitongoji vilivyoinuliwa",
    "mbolea iliyoyeyuka",
    "matibabu ya mbegu ya mnyororo",
    "asidi ya fomiki",
    "kisulfuli",
    "sprekla",
    "viboko vya madini",
    "kuogelea kinyesi",
    "utamaduni wa bakteria",
    "node za limfa",
    "bots za pua",
    "joto la ukandamizaji",
    "kununua majani",
    "kununua mafuta",
    "kununua kwenye",
    "kununua dawa",
    "changanya na kununua",
    "beetle wa kike",
    "nyuki wa parasiti",
    "mto wa bakteria",
    "kuja kwa roho",
    "kikombe, ukelea",
    "shinyanga",
    "mchafu wa shina",
    "kinyume cha kutosha",
    # Semantic hallucinations
    "kinyesi kinatolewa",
    "kunyonyeza kinyesi",
    "vilivyochomwa",  # roasted instead of dried
    "yamefukuzwa",   # chased away instead of flooded
    "magugu wa vitunguu",  # weeds instead of maggots
    "piga mgongo",   # hit the back instead of palpate
    "hunyunyizia maji",  # spray water instead of absorb
    "Mavazi mawili",  # clothes instead of varieties
    "ponya mbegu",   # heal instead of plant
    "Sasa viambato",  # now instead of grind
    "fungua kulingana",  # open instead of deworm
    "jioni ya kuchelewa",  # late evening glossary violation
    "ng'ombe kavu",  # dry cow literal translation

    # --- Added after reviewing the first full translation pass ---
    # Species / organism swaps.
    "mapopo",             # bats used for BSF larvae (use mabuu)
    "wanyama wa asili",   # wild animals used for natural enemies
    "kupiga mayai",       # should be kutaga mayai
    "weevili",            # hybrid plural of weevil
    # Mistranslated phrases.
    "ugali wa maji",      # 'continuous flooding' hallucination
    "weka vibaya",        # 'dispose of' mistranslated as 'place badly'
    "funika vibaya",      # 'bury' mistranslated as 'cover badly'
    "yasiyo wa wageni",   # 'non-host' mistranslated as 'non-guest'
    "funua kifuniko",     # 'apply mulch' mistranslated as 'uncover mulch'
    "komposti ya mashine",   # farmyard manure hallucination
    "komposti iliyoyeyuka",
    "samadi iliyoyeyuka",
    "msimu ya",           # wrong noun-class agreement (msimu wa ...)
    "safi, safi",         # 'clean, fresh' collapsed into a stutter
    "mkia wa mkia",       # 'tail of tail' duplication error
    "nanga viambato",     # 'weigh' mistranslated as 'anchor'
    # Hybrid / invented forms.
    "sanitiza",
    "kalibrated",
    "ideali",
    "fungashio",
    "yayeyesha",
    "iandamizwe",
    # Misspelled spray verbs (correct: kunyunyizia / kupulizia).
    "kununyizia",
    "kununyiza",
]

# Ordinary English terms that should normally be translated into Kiswahili.
# Proper nouns, brands, variety codes, Latin names, and product chemistries
# are still allowed, so this list is intentionally focused on common nouns,
# verbs, time words, and spacing words.
#
# NOTE: 'spp.' is deliberately NOT listed - Latin abbreviations such as
# 'Ficus spp.' are allowed by rule 4, and blocklisting 'spp.' only pushed
# the model into 'Ficusspp.' (which the word-boundary check could not even
# see). The concatenated form is normalized in normalize_units instead.
# Every term below has a glossary replacement.
QC_ENGLISH_TERMS: list[str] = [
    # Pest and disease names.
    "early blight",
    "late blight",
    "bacterial wilt",
    "blight",
    "wilt",
    "stem borer",
    "fall armyworm",
    "coffee berry borer",
    "borer",
    "whiteflies",
    "whitefly",
    "thrips",
    "leafminer",
    "tomato leafminer",
    "mites",
    "larvae",
    "pupae",
    # Equipment and inputs.
    "nozzle",
    "sprayer",
    "sprinkler",
    "sprinklers",
    "spray",
    "PPE",
    "mask",
    "herbicide",
    "fungicide",
    "biofungicide",
    "diverter",
    "diverters",
    "trellis",
    "sealant",
    "tray",
    "trays",
    "strip",
    "strips",
    "dryer",
    "silos",
    "silo",
    "supers",
    # Agronomy, feed, and process terms.
    "mulch",
    "mulching",
    "starch",
    "bran",
    "silage",
    "loam",
    "slope",
    "tillering",
    "ration",
    "rations",
    "legume",
    "legumes",
    "solution",
    "syrup",
    "active",
    "calibrated",
    "batch",
    "regenerative",
    "somatic",
    "anticoccidial",
    "molasses",
    "neem oil",
    # Time and spacing words: never leave these in English.
    "years",
    "year",
    "yr",
    "weeks",
    "week",
    "months",
    "month",
    "days",
    "day",
    "hours",
    "hour",
    "apart",
    # Institutional names: use KALRO.
    "Livestock Research Organization",
    # Ordinary adjectives / verbs.
    "ideal",
    "keep",
]

# English headings observed in bad translations.
QC_ENGLISH_HEADINGS: list[str] = [
    "Cultivation",
    "Harvesting",
    "Processing",
    "Feeding Rate",
    "Region Specific Note",
    "General Biosecurity",
    "Health Monitoring",
    "Veterinary Collaboration",
    "Probiotics",
    "Premix",
    "Electrolytes",
    "Early Blight",
    "Late Blight",
    "Bacterial Wilt",
]

# Regex QC checks run after normalization.
QC_BAD_REGEXES: list[tuple[str, str]] = [
    (
        r"(?i)\b(?:L|kg|g|t|mL|ml)\s+ha\s+1\b",
        "malformed unit: use L/ha, kg/ha, etc.",
    ),
    (
        r"(?i)\b(?:L|kg|g|t|mL|ml)\s*/\s*ha\s+1\b",
        "malformed unit: use L/ha, kg/ha, etc.",
    ),
    (
        r"(?i)\b(?:L|kg|g|t|mL|ml)\s+m\s+2\b",
        "malformed unit: use /m²",
    ),
    (
        r"(?i)\b(?:km|m|cm|mm|L|mL|ml)\s+h\s+1\b",
        "malformed unit: use /h",
    ),
    (
        r"(?i)\b(?:km|cm|mm|m)\s*[-]?\s*1\b",
        "malformed unit: use /m (e.g. 2-4 L/m), never m 1 / m-1",
    ),
    (
        r"(?i)\bm\s+2\b",
        "malformed unit: use m²",
    ),
    (
        r"(?i)\(.*\b(?:ft|feet|in|inches)\b.*\)",
        "added imperial conversion",
    ),
    (
        r"(?i)\b(?:viumbe|kiumbe)\b",
        "use miche/vikonyo/mmea or the animal's name; avoid viumbe/kiumbe",
    ),
    (
        r"(?i)\bkibio\b",
        "invented form: write kibiolojia / kibiashara, never kibio",
    ),
    (
        r"(?i)\bmanyu\b",
        "invented form for nozzle: write kichwa cha kunyunyizia",
    ),
    (
        r"(?i)\bspra\b",
        "clipped English form: write vyombo vya kunyunyizia, never spra",
    ),
    (
        r"(?i)\bdalili za kutokwa(?!\s+na\b)",
        "incomplete phrase: write kuharisha or kutokwa na choo",
    ),
]


# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are an expert translator for Kenyan smallholder agriculture.

Translate farmer Q&A from English into natural Kenyan Kiswahili: practical East African farmer language, not Sheng and not stiff word-for-word text.

Style (match the English card, compact for a 1024-token edge model):
- Keep the user turn as short as the English farmer question.
- Mirror English structure: a paragraph stays a paragraph; numbered steps only if English already lists. Do not add a lead sentence and then restate it as item 1. No markdown headings or extra advice.
- Keep the assistant no longer than the English assistant. Compress wording, but retain all essential instructions, numbers, doses, and steps.
- Delete adjectives, not numerals. Copy every figure exactly (18-20 %, 1 g/L, 2 kg/ha, brand rates).
- Do not convert units (75 cm stays 75 cm, not 0.75 m; 50 kg/acre stays 50 kg/acre unless the English already gives another unit).

Rules:
1. Translate meaning, not word-for-word. Keep short questions short.
2. Both user and assistant must be Kiswahili.
3. Do not add advice, remove steps, or alter facts, numbers, doses, spacing, or recommendations. If the English is agronomically wrong, still translate it; do not silently rewrite extension advice.
4. Preserve numerals, units, variety/product codes, Latin names (Botrytis, Rhizoctonia, Trichoderma, Ficus spp.), places, brands, chemicals, and acronyms (IPM, BCS, BSF, AI, CBD, KALRO).
5. Write units cleanly: 2.5-3.5 L/ha, 50 g/m², 5 km/h, 2-4 L/m/h. Never write L ha 1, L/ha 1, kg ha 1, m 2, km h 1, L m 1 h 1, or a leftover superscript 1.
6. Translate every ordinary English noun, including time and spacing words: miaka 3-4 (never 3-4 years), wiki 2-3 (never 2-3 weeks), kwa mwaka (never per yr), safu mbali (never rows apart). Only proper nouns, brands, codes, Latin names, product chemistries such as copper oxychloride, and units may remain English.
7. Use real Kiswahili. Never invent or hybridize forms such as fungishia, vichepeto, formuleni, imilisha, kunanyua, kalibrated, sanitiza, ideali, manyu, weevili, spra, kibio, fungashio, or yayeyesha.
8. Do not invent veterinary prescriptions. Preserve uncertainty from the English.
9. If the English user turn is typo-ridden, resolve intent from the English assistant (e.g. breads->breeds, steam a heifer, littles->young stock). Do not guess a different species.
10. Never swap livestock or crop species. Never reverse cheap/expensive, shallow/deep, dying/arriving, or dry/rise.
11. If the payload contains quality_feedback, fix all listed QC problems and return only the corrected JSON.

Hard QC examples:
- Write 2 kg/ha, never 2 kg ha 1 or 2 kg/ha 1.
- Write 50 g/m², never 50 g m 2.
- Write 5 km/h, never 5 km h 1.
- Write 2-4 L/m/h, never 2-4 L m 1 h 1.
- Write jioni, never jioni ghafla.
- Write miche or vikonyo for seedlings, never viumbe or kiumbe.
- Write kunyunyizia or kupulizia for spraying, never kunyonya, kunanyua, kununyizia, or kununua.
- Write kinyesi for droppings, never mchana.
- Write buni or matunda ya kahawa for coffee beans, never maziwa ya kahawa.
- Write upimaji wa udongo for soil testing, never majaribio ya udongo.
- Write mabuu or vifaranga vya BSF for BSF larvae, never mapopo (mapopo = popo).
- Write kutaga mayai for lay eggs, never kupiga mayai.
- Write maadui wa asili for natural enemies, never wanyama wa asili.
- Write samadi ya banda for farmyard manure, never komposti ya mashine ya shamba.
- Write silaji for silage, udongo mchanganyiko for loam, mteremko for slope, suluhisho for solution, sirupu for syrup, barakoa for mask, tabo for tray, and never leave 'apart' untranslated.
- Write KALRO, never the long English organization name.
- Write Ficus spp. and Grevillea spp. with a space, never Ficusspp.
- Do not leave ordinary English headings such as Cultivation:, Harvesting:, Processing:, Feeding Rate:.

Locked glossary:
{GLOSSARY_BLOCK}

Return only JSON with keys user and assistant."""


# --------------------------------------------------------------------------
# JSON schema
# --------------------------------------------------------------------------

JSON_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "swahili_pair",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "user": {"type": "string"},
                "assistant": {"type": "string"},
            },
            "required": ["user", "assistant"],
            "additionalProperties": False,
        },
    },
}


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def default_output_path() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working") / DEFAULT_OUTPUT_NAME
    return Path.cwd() / DEFAULT_OUTPUT_NAME


def checkpoint_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".progress.jsonl")


def failed_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".failed.jsonl")


def batch_state_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".batch_state.json")


# --------------------------------------------------------------------------
# API key loading
# --------------------------------------------------------------------------

def load_api_key() -> None:
    """Load GROQ_API_KEY from the environment or Kaggle secret."""
    if Path("/kaggle/working").exists() and not os.environ.get("GROQ_API_KEY"):
        try:
            from kaggle_secrets import UserSecretsClient

            os.environ["GROQ_API_KEY"] = (
                UserSecretsClient().get_secret("GROQ_API_KEY")
            )
        except Exception as exc:
            raise RuntimeError(
                "Kaggle detected, but GROQ_API_KEY could not be loaded "
                "from Kaggle Secrets."
            ) from exc

    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Export it with:\n"
            "  export GROQ_API_KEY='gsk_...'"
        )


# --------------------------------------------------------------------------
# Hugging Face token loading
# --------------------------------------------------------------------------

def get_hf_token() -> str | None:
    """
    Return an explicit Hugging Face token if supplied.
    Public datasets can be downloaded without a token.
    """
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or None
    )


# --------------------------------------------------------------------------
# Dataset extraction
# --------------------------------------------------------------------------

def content_from_turn(turn: dict[str, Any]) -> str:
    """
    Extract text from a message/turn.

    Supports:
    {"content": "..."}
    {"value": "..."}
    {"content": [{"text": "..."}]}
    """
    value = turn.get("content", turn.get("value", ""))
    if isinstance(value, list):
        value = " ".join(
            str(part.get("text", ""))
            if isinstance(part, dict)
            else str(part)
            for part in value
        )
    return str(value).strip()


def extract_pair(row: Any) -> tuple[str, str] | None:
    """
    Extract the first user -> assistant pair from a dataset row.
    Supports both dict formats (e.g., {"messages": [...]}) and direct list formats ([...]).
    """
    if isinstance(row, list):
        turns = row
    elif isinstance(row, dict):
        turns = row.get("messages") or row.get("conversations")
    else:
        return None

    if not isinstance(turns, list):
        return None

    user = ""
    for turn in turns:
        if not isinstance(turn, dict):
            continue

        role = str(
            turn.get(
                "role",
                turn.get("from", ""),
            )
        ).lower()
        content = content_from_turn(turn)

        if role in {"user", "human"} and content and not user:
            user = content
        elif (
            role in {"assistant", "model", "gpt"}
            and content
            and user
        ):
            return user, content

    return None


def load_english_rows() -> list[dict[str, str]]:
    """
    Download and load the actual source JSONL file from Hugging Face.

    Repository:
        kuzaai/agri_sft_prod_dedup_25k

    File:
        gemma4_agri_sft_25k.jsonl

    This intentionally avoids datasets.load_dataset() because the source
    is a plain JSONL file and direct Hub download is more deterministic.

    `strip_bad_conversions` (defined in the QC helpers section below)
    removes mathematically wrong acre->kg/ha parentheticals from the
    English source before anything else sees them.
    """
    hf_token = get_hf_token()

    print(
        "Downloading dataset file:\n"
        f"  repo: {SOURCE_REPO}\n"
        f"  file: {SOURCE_FILENAME}"
    )

    try:
        local_path = hf_hub_download(
            repo_id=SOURCE_REPO,
            filename=SOURCE_FILENAME,
            repo_type="dataset",
            token=hf_token,
        )
    except Exception as exc:
        raise RuntimeError(
            "\nFailed to download the Hugging Face source dataset.\n"
            f"Repository: {SOURCE_REPO}\n"
            f"Filename: {SOURCE_FILENAME}\n"
            f"Error: {type(exc).__name__}: {exc}\n\n"
            "For a private repository, authenticate with:\n"
            "  hf auth login\n"
            "or set:\n"
            "  export HF_TOKEN=hf_..."
        ) from exc

    english_rows: list[dict[str, str]] = []

    with open(local_path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on dataset line {index + 1}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                continue

            pair = extract_pair(row)
            if not pair:
                continue

            source_id = str(
                row.get("id")
                or f"english-{index}"
            )

            english_rows.append(
                {
                    "source_id": source_id,
                    "en_user": strip_bad_conversions(pair[0]),
                    "en_assistant": strip_bad_conversions(pair[1]),
                }
            )

    if not english_rows:
        raise RuntimeError(
            "No valid English user/assistant pairs were found in "
            f"{SOURCE_FILENAME}."
        )

    source_ids = [
        row["source_id"]
        for row in english_rows
    ]
    if len(set(source_ids)) != len(source_ids):
        raise RuntimeError(
            "Duplicate source_id values detected in the source dataset."
        )

    print(f"Loaded {len(english_rows):,} English pairs.")
    print(f"Local Hugging Face cache file: {local_path}")
    return english_rows


# --------------------------------------------------------------------------
# QC helpers
# --------------------------------------------------------------------------

class TranslationQCError(ValueError):
    """QC failure with optional raw model output for retry feedback."""

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


def normalize_units(text: str) -> str:
    if not text:
        return ""

    text = text.replace("‑", "-").replace("–", "-")

    # Fix hyphenated thousands (e.g., 5-000 -> 5000).
    text = re.sub(r"\b(\d)-(\d{3})\b", r"\1\2", text)

    # Fix scientific-notation typos (1-109 / 109 -> 10^9), but only when
    # directly attached to CFU or conidia. A bare 1109 might be a real
    # quantity (price, yield, altitude) and must never be rewritten.
    text = re.sub(r"(?i)\b1-?109\s+(cfu|conidia)\b", r"10^9 \1", text)
    text = re.sub(r"(?i)\b(cfu|conidia)\s+1-?109\b", r"\1 10^9", text)

    # Latin-name concatenations from the source: 'Ficusspp.' -> 'Ficus spp.'
    # (idempotent: already-spaced 'Ficus spp.' passes through unchanged).
    text = re.sub(r"(?i)\b([a-z]{4,})\s*spp\.?", r"\1 spp.", text)

    # Per-hectare rates: '2 kg ha 1', '2 kg/ha 1', '2 kg ha-1',
    # '2 kg ha^-1', '2 kg ha' -> '2 kg/ha'.
    text = re.sub(r"(?i)\b(l|kg|g|t|ml)\s*/?\s*ha\s*[-]?\s*1\b", r"\1/ha", text)
    text = re.sub(r"(?i)\b(l|kg|g|t|ml)\s+ha\b", r"\1/ha", text)
    text = re.sub(r"(?i)\b(kg|g|t)\s+(n|p2o5|k2o|p|k)\s+ha\b", r"\1 \2/ha", text)
    text = re.sub(r"(?i)\b(l|kg|g|t|ml)\s*ha\s*\^\s*-\s*1\b", r"\1/ha", text)

    # Per-square-meter rates: '50 g m 2', '50 g / m 2' -> '50 g/m²'.
    # Must run BEFORE the bare 'm 2' -> 'm²' rules below, otherwise
    # '50 g m 2' would only become '50 g m²'.
    text = re.sub(r"(?i)\b(l|kg|g|t|ml)\s*/?\s*m\s+2\b", r"\1/m²", text)

    # Compound flow rates: '2-4 L m 1 h 1' -> '2-4 L/m/h'.
    text = re.sub(
        r"(?i)\b(l|ml)\s+m\s*[-]?\s*1\s+h\s*[-]?\s*1\b",
        r"\1/m/h",
        text,
    )

    # Per-hour and per-day rates.
    text = re.sub(r"(?i)\b(km|m|cm|mm|l|ml)\s*h\s*[-]?\s*1\b", r"\1/h", text)
    text = re.sub(r"(?i)\b(l|ml|kg|g|t)\s*d\s*[-]?\s*1\b", r"\1/d", text)

    # CFU/L and conidia/L: '10^9 CFU L 1' -> '10^9 CFU/L'.
    text = re.sub(r"(?i)\b(cfu|conidia)\s*/?\s*l\s*[-]?\s*1\b", r"\1/L", text)

    # Clean remaining slash forms.
    text = re.sub(r"(?i)/\s*ha\s*[-]?\s*1\b", "/ha", text)
    text = re.sub(r"(?i)\bha\s*[-]?\s*1\b", "ha", text)

    # Bare 'm 2' -> 'm²' (unit form). Never fire when the 2 begins a
    # decimal number ('m 2.5' is a depth, not m²) and never inside an
    # 'x' dimension pair ('2.5 m x 2.5 m' contains no 'm 2' anyway).
    # The old destructive 'm².5 -> m x 5' patch is gone: the lookahead
    # makes it unnecessary, and it used to eat real digits.
    text = re.sub(r"(?i)(\d)\s*m\s*2(?!\.\d)\b", r"\1 m²", text)
    text = re.sub(r"(?i)\bm\s+2(?!\.\d)\b", "m²", text)
    text = re.sub(r"(?i)\bm\s*\^\s*2\b", "m²", text)

    return text


def strip_bad_conversions(text: str) -> str:
    """
    Remove mathematically incorrect parenthetical acre->kg/ha conversions
    from the English source before it is sent to the model.

    The source dataset contains advice such as:

        "50 kg / acre ( 12 kg / ha)"

    where the parenthesis is simply wrong (50 kg/acre is roughly
    125 kg/ha). Left in place, the model faithfully copies the bogus
    conversion into the Swahili output. Only the acre-rate +
    bogus-parenthesis pattern is removed; legitimate standalone kg/ha
    rates elsewhere in the text are untouched.
    """
    text = re.sub(
        r"(?i)(\b\d+(?:[.,]\d+)?\s*kg\s*/\s*acre\s*)"
        r"\(\s*\d+(?:[.,]\d+)?\s*kg\s*/\s*ha\s*\)",
        r"\1",
        text,
    )
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_text(text: str) -> str:
    """Normalize whitespace and units without destroying paragraph breaks."""
    if not text:
        return ""

    text = text.strip()
    text = normalize_units(text)

    # Collapse horizontal whitespace but keep newlines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)

    # Tighten slashed rates left over from the source:
    # 'kg / acre' -> 'kg/acre', 'mara/ siku' -> 'mara/siku'.
    text = re.sub(
        r"(?i)\b(kg|g|t|l|ml|mm|cm|km|m)\s*/\s*(ha|acre|m²|h|d)\b",
        r"\1/\2",
        text,
    )
    text = re.sub(
        r"(?i)\b(mara|baada|kabla)\s*/\s*(siku|wiki|mwezi|mwaka)\b",
        r"\1/\2",
        text,
    )

    return text.strip()


def canonical_number(num: str) -> str:
    """
    Canonicalize simple numeric strings for comparison.

    Strips trailing zeros ('2.50' -> '2.5') and leading zeros
    ('04' vs '4' compare equal, so clock times written either way do
    not trigger spurious missing-numeral failures).
    """
    if "." in num:
        num = num.rstrip("0").rstrip(".")
        if num in ("", "."):
            return "0"
        whole, _, frac = num.partition(".")
        whole = whole.lstrip("0") or "0"
        return f"{whole}.{frac}" if frac else whole

    num = num.lstrip("0")
    return num if num else "0"


def extract_numbers(text: str) -> set[str]:
    """
    Extract normalized numeric tokens.

    This does not try to be a full numeric parser. It is designed to catch
    missing doses, rates, percentages, spacings, and variety codes.
    """
    if not text:
        return set()

    text = normalize_units(text)

    # Normalize explicit thousands separators: 1,000 -> 1000.
    text = re.sub(
        r"\b(\d{1,3}(?:,\d{3})+)\b",
        lambda m: m.group(1).replace(",", ""),
        text,
    )

    # Treat remaining decimal commas as decimal points.
    text = re.sub(
        r"\b(\d+),(\d+)\b",
        r"\1.\2",
        text,
    )

    nums = re.findall(r"\d+(?:\.\d+)?", text)
    return {canonical_number(n) for n in nums}


def find_bad_phrases(text: str) -> list[str]:
    """Find known bad Swahili phrases."""
    lower = text.lower()
    found = set()
    for phrase in QC_BAD_PHRASES:
        if phrase.lower() in lower:
            found.add(phrase)
    return sorted(found)


def find_english_terms(text: str) -> list[str]:
    """Find common English terms that should have been translated."""
    found = set()
    for term in QC_ENGLISH_TERMS:
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.add(term)
    return sorted(found)


def find_english_headings(text: str) -> list[str]:
    """Find English label/heading lines such as `Cultivation:`."""
    found = set()
    for heading in QC_ENGLISH_HEADINGS:
        pattern = r"(?m)^\s*" + re.escape(heading) + r"\s*:"
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.add(f"{heading}:")
    return sorted(found)


def find_bad_regexes(text: str) -> list[str]:
    """Run regex-based QC checks."""
    found = set()
    for pattern, label in QC_BAD_REGEXES:
        if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
            found.add(label)
    return sorted(found)


def find_added_conversions(en_text: str, sw_text: str) -> list[str]:
    """
    Detect unit conversions added in Swahili that are absent from English.
    """
    issues = set()

    en_lower = en_text.lower()
    sw_lower = sw_text.lower()

    imperial_pattern = (
        r"\d+(?:[.,]\d+)?\s*"
        r"(?:ft|feet|in|inches|yd|yards|lb|lbs|pounds?)\b"
    )

    if re.search(imperial_pattern, sw_lower) and not re.search(
        imperial_pattern,
        en_lower,
    ):
        issues.add("added imperial conversion")

    paren_pattern = (
        r"\([^)]*\b(?:ft|feet|in|inches|yd|yards|lb|lbs)\b[^)]*\)"
    )

    if re.search(paren_pattern, sw_lower) and not re.search(
        paren_pattern,
        en_lower,
    ):
        issues.add("added conversion parentheses")

    # If Swahili mentions acre and ha together, but English never mentions
    # hectares, it is likely an added conversion.
    if (
        re.search(r"(?i)\bacre\b", sw_text)
        and re.search(r"(?i)\bha\b", sw_text)
        and not re.search(r"(?i)\bha\b|\bhectare\b", en_text)
    ):
        if re.search(r"(?i)\bacre\b[^.\n]*\bha\b", sw_text):
            issues.add("added acre-to-hectare conversion")

    return sorted(issues)


def validate_translation(
    sw: dict[str, str],
    en_user: str = "",
    en_assistant: str = "",
) -> None:
    """
    Apply vital translation QC gates.

    This mutates `sw` by normalizing whitespace and units.
    """
    if not sw.get("user", "").strip():
        raise ValueError("empty translated user turn")

    if not sw.get("assistant", "").strip():
        raise ValueError("empty translated assistant turn")

    sw["user"] = normalize_text(sw["user"])
    sw["assistant"] = normalize_text(sw["assistant"])

    if sw["user"].strip() == sw["assistant"].strip():
        raise ValueError("user and assistant translations are identical")

    combined_sw = sw["user"] + "\n" + sw["assistant"]

    bad_phrases = find_bad_phrases(combined_sw)
    if bad_phrases:
        raise ValueError(
            "QC forbidden Swahili phrases: "
            + ", ".join(bad_phrases[:12])
        )

    english_terms = find_english_terms(combined_sw)
    if english_terms:
        raise ValueError(
            "QC untranslated English terms: "
            + ", ".join(english_terms[:12])
        )

    english_headings = find_english_headings(combined_sw)
    if english_headings:
        raise ValueError(
            "QC English headings: "
            + ", ".join(english_headings[:12])
        )

    bad_regexes = find_bad_regexes(combined_sw)
    if bad_regexes:
        raise ValueError(
            "QC pattern failures: "
            + ", ".join(bad_regexes[:12])
        )

    if en_user or en_assistant:
        en_combined = normalize_text(
            (en_user or "") + "\n" + (en_assistant or "")
        )

        en_numbers = extract_numbers(en_combined)
        sw_numbers = extract_numbers(combined_sw)
        missing_numbers = en_numbers - sw_numbers

        if missing_numbers:
            raise ValueError(
                "missing numerals from English: "
                + ", ".join(sorted(missing_numbers)[:12])
            )

        conversion_issues = find_added_conversions(
            en_combined,
            combined_sw,
        )
        if conversion_issues:
            raise ValueError(
                "QC conversion issues: "
                + ", ".join(conversion_issues[:12])
            )

        if en_assistant:
            max_assistant_len = (
                int(len(en_assistant) * QC_MAX_ASSISTANT_RATIO)
                + QC_MAX_ASSISTANT_PADDING
            )
            if len(sw["assistant"]) > max_assistant_len:
                raise ValueError(
                    "assistant translation too long: "
                    f"{len(sw['assistant'])} chars > "
                    f"{max_assistant_len} chars"
                )

        if en_user:
            max_user_len = (
                int(len(en_user) * QC_MAX_USER_RATIO)
                + QC_MAX_USER_PADDING
            )
            if len(sw["user"]) > max_user_len:
                raise ValueError(
                    "user translation too long: "
                    f"{len(sw['user'])} chars > "
                    f"{max_user_len} chars"
                )


# --------------------------------------------------------------------------
# Prompt / parsing / validation
# --------------------------------------------------------------------------

def parse_translation_json(raw: str) -> dict[str, str]:
    """
    Parse a translation response as JSON.

    Handles:
    - regular JSON
    - fenced ```json blocks
    - accidental surrounding text containing one JSON object
    """
    text = raw.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            text,
            flags=re.I,
        )

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(
            text[start : end + 1]
        )

    if not isinstance(data, dict):
        raise ValueError(
            "Translation response is not a JSON object."
        )

    user = str(
        data.get("user", "")
    ).strip()
    assistant = str(
        data.get("assistant", "")
    ).strip()

    if not user or not assistant:
        raise ValueError(
            "empty user or assistant"
        )

    return {
        "user": user,
        "assistant": assistant,
    }


def chat_messages(
    en_user: str,
    en_asst: str,
    feedback: str | None = None,
) -> list[dict[str, str]]:
    """
    Construct the Groq chat payload.
    """
    payload: dict[str, str] = {
        "english_user": en_user,
        "english_assistant": en_asst,
    }

    if feedback:
        payload["quality_feedback"] = feedback

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
            ),
        },
    ]


def qc_feedback(
    raw: str,
    error: str,
) -> str:
    """Build retry feedback for QC failures."""
    snippet = raw.strip()[:1500] if raw else ""

    parts = [
        "The previous translation failed quality control.",
        f"Error: {error}",
    ]

    if snippet:
        parts.append("Previous output:")
        parts.append(snippet)

    parts.append(
        "Translate again into natural Kenyan Kiswahili. "
        "Fix units (for example 2 kg/ha, 50 g/m², 5 km/h), "
        "translate every ordinary English word including time words "
        "(years, weeks, apart), remove ordinary English words and "
        "headings, obey the glossary, preserve all numbers, and return "
        "only JSON with keys user and assistant."
    )

    return "\n".join(parts)


def token_budget(
    en_user: str,
    en_asst: str,
    retry: int = 0,
) -> int:
    """
    Estimate completion budget.

    gpt-oss reasoning tokens consume the same completion budget, so reserve
    space for reasoning plus visible translation output.
    """
    visible = int(
        (len(en_user) + len(en_asst)) / 3
    ) + 64

    budget = max(
        MIN_OUTPUT_TOKENS,
        REASONING_RESERVE_TOKENS + visible,
    ) * (2 ** retry)

    return min(
        MAX_OUTPUT_TOKENS,
        budget,
    )


def completion_body(
    en_user: str,
    en_asst: str,
    max_tokens: int,
    feedback: str | None = None,
) -> dict[str, Any]:
    """
    Build a Groq chat-completion request.
    """
    return {
        "model": TRANSLATOR_MODEL,
        "messages": chat_messages(
            en_user,
            en_asst,
            feedback,
        ),
        "temperature": 0.2,
        "max_completion_tokens": max_tokens,
        "reasoning_effort": REASONING_EFFORT,
        "response_format": JSON_SCHEMA,
    }


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """
    Write records as UTF-8 JSONL.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# --------------------------------------------------------------------------
# Checkpoint handling
# --------------------------------------------------------------------------

def load_checkpoint(
    path: Path,
) -> dict[str, dict[str, Any]]:
    """
    Load already translated rows keyed by source_id.

    A malformed/torn final line is ignored.
    """
    if not path.exists():
        return {}

    done: dict[str, dict[str, Any]] = {}

    with path.open(
        encoding="utf-8"
    ) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            source_id = record.get("source_id")
            if source_id:
                done[source_id] = record

    return done


def to_record(
    source_id: str,
    sw: dict[str, str],
) -> dict[str, Any]:
    """
    Convert a translation into final SFT message format.
    """
    return {
        "source_id": source_id,
        "messages": [
            {
                "role": "user",
                "content": sw["user"],
            },
            {
                "role": "assistant",
                "content": sw["assistant"],
            },
        ],
    }


# --------------------------------------------------------------------------
# Online translation
# --------------------------------------------------------------------------

def message_text(
    completion: Any,
) -> str:
    """
    Extract visible message content from a Groq response.
    """
    message = completion.choices[0].message
    content = getattr(
        message,
        "content",
        None,
    )

    if content:
        return str(content)

    finish_reason = getattr(
        completion.choices[0],
        "finish_reason",
        None,
    )

    if finish_reason == "length":
        raise ValueError(
            "response was truncated before any content was produced "
            "(finish_reason=length)"
        )

    raise ValueError(
        f"empty model content "
        f"(finish_reason={finish_reason!r})"
    )


def error_body(
    exc: Exception,
) -> dict[str, Any]:
    """
    Safely extract an API error body.
    """
    body = getattr(
        exc,
        "body",
        None,
    )

    return (
        body
        if isinstance(body, dict)
        else {}
    )


def recoverable_json_failure(
    exc: Exception,
) -> bool:
    """
    Detect completion errors that may be resolved by allocating more tokens.
    """
    err = (
        error_body(exc)
        .get("error")
        or {}
    )

    blob = (
        f"{err.get('code', '')} "
        f"{err.get('message', '')} "
        f"{exc}"
    ).lower()

    return (
        "json_validate_failed" in blob
        or "max completion tokens" in blob
    )


def parse_completion(
    completion: Any,
    en_user: str = "",
    en_asst: str = "",
) -> dict[str, str]:
    """
    Parse and validate a Groq completion.
    """
    raw = message_text(completion)

    try:
        parsed = parse_translation_json(raw)
        validate_translation(
            parsed,
            en_user=en_user,
            en_assistant=en_asst,
        )
    except Exception as exc:
        raise TranslationQCError(str(exc), raw=raw) from exc

    return parsed


class OnlineLimiter:
    """
    Shared concurrency + rate-limit backoff controller.
    """

    def __init__(
        self,
        concurrency: int,
    ):
        self.sema = asyncio.Semaphore(
            concurrency
        )
        self.pause_until = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        delay = (
            self.pause_until
            - time.time()
        )

        if delay > 0:
            await asyncio.sleep(
                delay
                + random.random() * 0.25
            )
        else:
            jitter = (
                random.random()
                * 0.05
            )
            if jitter:
                await asyncio.sleep(
                    jitter
                )

    async def pause(
        self,
        seconds: float,
    ) -> None:
        async with self.lock:
            self.pause_until = max(
                self.pause_until,
                time.time()
                + max(
                    seconds,
                    1.0,
                ),
            )


def retry_after(
    exc: Exception,
    attempt: int,
) -> float:
    """
    Respect Retry-After when supplied, otherwise use exponential backoff.
    """
    response = getattr(
        exc,
        "response",
        None,
    )

    headers = (
        {
            str(k).lower(): v
            for k, v
            in getattr(
                response,
                "headers",
                {},
            ).items()
        }
        if response
        else {}
    )

    try:
        return float(
            headers.get(
                "retry-after",
                "",
            )
        )
    except ValueError:
        return min(
            60.0,
            2 ** attempt
            + random.random(),
        )

async def translate_async(
    client: AsyncGroq,
    limiter: OnlineLimiter,
    en_user: str,
    en_asst: str,
) -> dict[str, str]:
    """
    Translate one pair.

    Policy:
      - First generation is attempt 1.
      - QC failures get at most MAX_QC_RETRIES corrections.
      - After that, the row is rejected/skipped.
      - Transient API/rate-limit failures are handled separately.
    """
    retry = 0
    transient_retries = 0

    last_error: Exception | None = None
    feedback: str | None = None

    while True:
        await limiter.wait()

        budget = token_budget(
            en_user,
            en_asst,
            retry,
        )

        async with limiter.sema:
            try:
                result = await client.chat.completions.create(
                    **completion_body(
                        en_user,
                        en_asst,
                        budget,
                        feedback,
                    )
                )

                return parse_completion(
                    result,
                    en_user=en_user,
                    en_asst=en_asst,
                )

            except RateLimitError as exc:
                last_error = exc
                transient_retries += 1

                if transient_retries > MAX_TRANSIENT_RETRIES:
                    raise RuntimeError(
                        f"rate limit persisted after "
                        f"{MAX_TRANSIENT_RETRIES} retries: {exc}"
                    ) from exc

                wait = retry_after(
                    exc,
                    transient_retries,
                )
                await limiter.pause(wait)
                await asyncio.sleep(wait)

            except FATAL_EXCEPTIONS:
                raise

            except BadRequestError as exc:
                last_error = exc

                if not recoverable_json_failure(exc):
                    raise

                # Don't keep increasing token budget forever.
                if budget >= MAX_OUTPUT_TOKENS:
                    raise RuntimeError(
                        f"completion failed at maximum token budget: {exc}"
                    ) from exc

                retry += 1

                if retry > MAX_QC_RETRIES:
                    raise RuntimeError(
                        f"translation rejected after "
                        f"{MAX_QC_RETRIES} QC retries: {exc}"
                    ) from exc

            except TranslationQCError as exc:
                last_error = exc

                # We already have a bad translation. Give the model a
                # couple of chances to correct it, then abandon the row.
                retry += 1

                if retry > MAX_QC_RETRIES:
                    raise RuntimeError(
                        f"translation rejected after "
                        f"{MAX_QC_RETRIES} QC retries: {exc}"
                    ) from exc

                feedback = qc_feedback(
                    exc.raw,
                    str(exc),
                )

                await asyncio.sleep(
                    min(
                        10.0,
                        1.0 + random.random(),
                    )
                )

            except (
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                last_error = exc
                retry += 1

                if retry > MAX_QC_RETRIES:
                    raise RuntimeError(
                        f"translation rejected after "
                        f"{MAX_QC_RETRIES} correction retries: {exc}"
                    ) from exc

                feedback = qc_feedback(
                    "",
                    str(exc),
                )

                await asyncio.sleep(
                    min(
                        10.0,
                        1.0 + random.random(),
                    )
                )

            except (
                APIConnectionError,
                APIStatusError,
            ) as exc:
                last_error = exc
                transient_retries += 1

                if transient_retries > MAX_TRANSIENT_RETRIES:
                    raise RuntimeError(
                        f"API failure persisted after "
                        f"{MAX_TRANSIENT_RETRIES} retries: {exc}"
                    ) from exc

                await asyncio.sleep(
                    min(
                        30.0,
                        2 ** min(transient_retries, 5)
                        + random.random(),
                    )
                )
                

def load_failed_ids(
    path: Path,
) -> set[str]:
    """
    Load source IDs that were previously rejected/skipped.
    """
    if not path.exists():
        return set()

    failed: set[str] = set()

    with path.open(
        encoding="utf-8",
    ) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            source_id = record.get("source_id")
            if source_id:
                failed.add(str(source_id))

    return failed


async def translate_all(
    rows: list[dict[str, str]],
    checkpoint: Path,
    failed_log: Path,
    concurrency: int,
) -> int:
    """
    Translate remaining rows and append successful rows to the checkpoint.
    """
    if not rows:
        return 0

    client = AsyncGroq(
        api_key=os.environ["GROQ_API_KEY"]
    )

    limiter = OnlineLimiter(
        concurrency
    )

    queue: asyncio.Queue[
        dict[str, str]
    ] = asyncio.Queue()

    for row in rows:
        queue.put_nowait(row)

    failures: list[
        tuple[str, str]
    ] = []
    fatal: Exception | None = None

    checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    progress = tqdm(
        total=len(rows),
        desc="translate (online)",
    )

    async def worker(
        ckpt_handle: Any,
    ) -> None:
        nonlocal fatal

        while fatal is None:
            try:
                row = queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            try:
                sw = await translate_async(
                    client,
                    limiter,
                    row["en_user"],
                    row["en_assistant"],
                )

                ckpt_handle.write(
                    json.dumps(
                        to_record(
                            row["source_id"],
                            sw,
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                ckpt_handle.flush()

                progress.update(1)

            except FATAL_EXCEPTIONS as exc:
                fatal = exc

            except Exception as exc:
                failures.append(
                    (
                        row["source_id"],
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                progress.update(1)

            finally:
                queue.task_done()

    try:
        with checkpoint.open(
            "a",
            encoding="utf-8",
        ) as ckpt_handle:
            await asyncio.gather(
                *(
                    worker(ckpt_handle)
                    for _ in range(concurrency)
                )
            )
    finally:
        progress.close()
        await client.close()

    if fatal is not None:
        raise RuntimeError(
            "Aborting: non-retryable API error "
            f"({type(fatal).__name__}: {fatal}). "
            "Fix credentials/permissions and re-run. "
            f"Already translated rows are saved in {checkpoint}."
        ) from fatal

    if failures:
        with failed_log.open(
            "a",
            encoding="utf-8",
        ) as handle:
            for source_id, reason in failures:
                handle.write(
                    json.dumps(
                        {
                            "source_id": source_id,
                            "reason": reason,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        print(
            f"{len(failures):,} rows failed after retries; "
            f"see {failed_log}"
        )

    return len(failures)


# --------------------------------------------------------------------------
# Batch mode
# --------------------------------------------------------------------------

def load_batch_state(
    path: Path,
) -> dict[str, Any]:
    if path.exists():
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    return {
        "chunks": []
    }


def save_batch_state(
    path: Path,
    state: dict[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            state,
            indent=2,
        ),
        encoding="utf-8",
    )


def batch_line(
    row: dict[str, str],
) -> str:
    """
    Convert a dataset row to one Groq Batch API JSONL request.
    """
    body = completion_body(
        row["en_user"],
        row["en_assistant"],
        MAX_OUTPUT_TOKENS,
    )

    payload = {
        "custom_id": row["source_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
    )


def submit_batches(
    client: Groq,
    rows: list[dict[str, str]],
    state: dict[str, Any],
    state_path: Path,
) -> None:
    """
    Submit fresh rows in Batch API chunks.
    """
    already_submitted = {
        source_id
        for chunk in state["chunks"]
        for source_id
        in chunk.get(
            "source_ids",
            [],
        )
    }

    fresh_rows = [
        row
        for row in rows
        if row["source_id"]
        not in already_submitted
    ]

    if not fresh_rows:
        return

    chunks = [
        fresh_rows[i : i + BATCH_CHUNK_SIZE]
        for i in range(
            0,
            len(fresh_rows),
            BATCH_CHUNK_SIZE,
        )
    ]

    next_index = len(
        state["chunks"]
    )

    for offset, chunk in enumerate(chunks):
        index = (
            next_index
            + offset
        )

        chunk_path = Path(
            f"/tmp/kuza_batch_input_{index}.jsonl"
        )

        with chunk_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            for row in chunk:
                handle.write(
                    batch_line(row)
                    + "\n"
                )

        with chunk_path.open(
            "rb"
        ) as fh:
            upload = client.files.create(
                file=fh,
                purpose="batch",
            )

        job = client.batches.create(
            completion_window=BATCH_COMPLETION_WINDOW,
            endpoint="/v1/chat/completions",
            input_file_id=upload.id,
        )

        state["chunks"].append(
            {
                "chunk_index": index,
                "source_ids": [
                    row["source_id"]
                    for row in chunk
                ],
                "input_file_id": upload.id,
                "batch_id": job.id,
                "status": job.status,
                "output_file_id": None,
                "error_file_id": None,
                "ingested": False,
            }
        )

        save_batch_state(
            state_path,
            state,
        )

        print(
            f"submitted chunk {index}: "
            f"batch_id={job.id} "
            f"({len(chunk)} rows)"
        )


def ingest_batch_chunk(
    client: Groq,
    chunk: dict[str, Any],
    ckpt_handle: Any,
    rows_by_id: dict[str, dict[str, str]],
) -> int:
    """
    Download, parse, validate, and checkpoint successful batch responses.

    Rows that fail QC are intentionally not checkpointed. They will be
    retried later by the online finishing pass.
    """
    if not chunk.get(
        "output_file_id"
    ):
        return 0

    out_path = Path(
        f"/tmp/kuza_batch_output_"
        f"{chunk['chunk_index']}.jsonl"
    )

    content = client.files.content(
        chunk["output_file_id"]
    )
    content.write_to_file(
        str(out_path)
    )

    ingested = 0

    with out_path.open(
        encoding="utf-8"
    ) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            source_id = item.get(
                "custom_id"
            )

            response = (
                item.get("response")
                or {}
            )

            if (
                not source_id
                or item.get("error")
                or response.get(
                    "status_code"
                )
                != 200
            ):
                continue

            en_row = rows_by_id.get(
                source_id
            )

            if not en_row:
                continue

            try:
                content_text = (
                    response["body"]
                    ["choices"][0]
                    ["message"]
                    ["content"]
                )

                sw = parse_translation_json(
                    content_text
                )

                validate_translation(
                    sw,
                    en_user=en_row["en_user"],
                    en_assistant=en_row["en_assistant"],
                )

            except (
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                continue

            ckpt_handle.write(
                json.dumps(
                    to_record(
                        source_id,
                        sw,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
            ingested += 1

    ckpt_handle.flush()
    return ingested


def run_batch_mode(
    rows: list[dict[str, str]],
    checkpoint: Path,
    state_path: Path,
    failed_log: Path,
    poll_timeout: float,
) -> bool:
    """
    Submit unsubmitted chunks, poll for completion, and ingest results.

    Returns:
        True  -> all chunks terminal
        False -> some chunks still processing
    """
    client = Groq(
        api_key=os.environ["GROQ_API_KEY"]
    )

    already_done = load_checkpoint(
        checkpoint
    )
    failed_ids = load_failed_ids(failed_log)

    pending_rows = [
        row
        for row in rows
        if (
            row["source_id"] not in already_done
            and row["source_id"] not in failed_ids
        )
    ]

    state = load_batch_state(
        state_path
    )

    if pending_rows:
        submit_batches(
            client,
            pending_rows,
            state,
            state_path,
        )
    else:
        print(
            "All rows already checkpointed; "
            "nothing to submit."
        )

    rows_by_id = {
        row["source_id"]: row
        for row in rows
    }

    checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    deadline = (
        time.time()
        + poll_timeout
    )

    with checkpoint.open(
        "a",
        encoding="utf-8",
    ) as ckpt_handle:
        while True:
            all_terminal = True

            for chunk in state["chunks"]:
                if (
                    chunk["status"]
                    not in BATCH_TERMINAL_STATUSES
                ):
                    job = client.batches.retrieve(
                        chunk["batch_id"]
                    )

                    chunk["status"] = (
                        job.status
                    )
                    chunk["output_file_id"] = (
                        job.output_file_id
                    )
                    chunk["error_file_id"] = (
                        job.error_file_id
                    )

                    counts = (
                        job.request_counts
                    )
                    total = (
                        counts.total
                        if counts
                        else 0
                    )
                    completed = (
                        counts.completed
                        if counts
                        else 0
                    )
                    failed = (
                        counts.failed
                        if counts
                        else 0
                    )

                    print(
                        f"chunk {chunk['chunk_index']}: "
                        f"{job.status} "
                        f"({completed}/{total} done, "
                        f"{failed} failed)"
                    )

                if (
                    chunk["status"]
                    in BATCH_TERMINAL_STATUSES
                    and not chunk.get(
                        "ingested"
                    )
                ):
                    n = ingest_batch_chunk(
                        client,
                        chunk,
                        ckpt_handle,
                        rows_by_id,
                    )

                    print(
                        f"chunk {chunk['chunk_index']}: "
                        f"ingested {n} translated rows"
                        + (
                            f" "
                            f"(see error file "
                            f"{chunk['error_file_id']} "
                            f"for the rest)"
                            if chunk.get(
                                "error_file_id"
                            )
                            else ""
                        )
                    )

                    chunk["ingested"] = True

                if (
                    chunk["status"]
                    not in BATCH_TERMINAL_STATUSES
                ):
                    all_terminal = False

            save_batch_state(
                state_path,
                state,
            )

            if all_terminal:
                return True

            if time.time() >= deadline:
                return False

            time.sleep(
                BATCH_POLL_SECONDS
            )


# --------------------------------------------------------------------------
# Checkpoint revalidation
# --------------------------------------------------------------------------

def revalidate_checkpoint(
    checkpoint: Path,
    english_rows: list[dict[str, str]],
) -> int:
    """
    Re-run the current QC rules over checkpointed translations.

    Rows that fail are removed from the checkpoint so they are
    re-translated by the online pass afterwards. This lets you tighten
    the QC lists and clean up an existing run without paying for a
    full `--fresh` re-translation of all 25k rows.

    Returns the number of rows dropped.
    """
    done = load_checkpoint(checkpoint)
    if not done:
        print("revalidate: checkpoint is empty; nothing to do.")
        return 0

    rows_by_id = {
        row["source_id"]: row
        for row in english_rows
    }

    kept: dict[str, dict[str, Any]] = {}
    dropped: list[tuple[str, str]] = []

    for source_id, record in done.items():
        en_row = rows_by_id.get(source_id)
        messages = record.get("messages") or []

        if (
            en_row is None
            or len(messages) != 2
            or not isinstance(messages[0], dict)
            or not isinstance(messages[1], dict)
            or messages[0].get("role") != "user"
            or messages[1].get("role") != "assistant"
        ):
            # Unknown source or malformed record: keep as-is. It is
            # filtered out during final assembly anyway.
            kept[source_id] = record
            continue

        sw = {
            "user": str(messages[0].get("content", "")),
            "assistant": str(messages[1].get("content", "")),
        }

        try:
            # Pass a copy: validate_translation normalizes its argument.
            validate_translation(
                dict(sw),
                en_user=en_row["en_user"],
                en_assistant=en_row["en_assistant"],
            )
            kept[source_id] = record
        except ValueError as exc:
            dropped.append((source_id, str(exc)))

    if dropped:
        write_jsonl(checkpoint, list(kept.values()))
        print(
            f"revalidate: dropped {len(dropped)} of {len(done)} "
            "checkpointed rows."
        )
        for source_id, reason in dropped[:20]:
            print(f"  - {source_id}: {reason[:160]}")
        if len(dropped) > 20:
            print(f"  ... and {len(dropped) - 20} more.")
        print("They will be re-translated by the online pass.")
    else:
        print(
            f"revalidate: all {len(done)} rows still pass the current QC."
        )

    return len(dropped)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split(
            "\n",
            1,
        )[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help=(
            f"JSONL output path "
            f"(default: {DEFAULT_OUTPUT_NAME} "
            f"in /kaggle/working or cwd)"
        ),
    )

    parser.add_argument(
        "--mode",
        choices=[
            "online",
            "batch",
        ],
        default="online",
        help=(
            "'online': concurrent API calls. "
            "'batch': cheaper Groq Batch API first, "
            "then online finishing pass."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Only translate the first N rows. "
            "Useful for smoke testing."
        ),
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=ONLINE_CONCURRENCY,
        help=(
            "Concurrent in-flight requests for "
            "the online path."
        ),
    )

    parser.add_argument(
        "--batch-poll-timeout",
        type=float,
        default=3300.0,
        help=(
            "Seconds to poll batch jobs before "
            "exiting so the command can be re-run."
        ),
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Ignore existing checkpoint/batch-state "
            "files and start over."
        ),
    )

    parser.add_argument(
        "--revalidate",
        action="store_true",
        help=(
            "Re-run the current QC rules over checkpointed rows and "
            "drop (then re-translate) any row that fails. Use after "
            "tightening the QC lists instead of --fresh."
        ),
    )

    return parser.parse_args()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    load_api_key()

    output_path: Path = args.output
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = checkpoint_path(
        output_path
    )
    state_path = batch_state_path(
        output_path
    )
    failed_log = failed_path(
        output_path
    )

    if args.fresh:
        for path in (
            checkpoint,
            state_path,
            failed_log,
        ):
            path.unlink(
                missing_ok=True
            )

    print(
        "final output:",
        output_path,
    )
    print(
        "checkpoint:",
        checkpoint,
    )
    print(
        "source repo:",
        SOURCE_REPO,
    )
    print(
        "source file:",
        SOURCE_FILENAME,
    )
    print(
        "glossary entries:",
        len(GLOSSARY),
        "| prompt chars:",
        len(SYSTEM_PROMPT),
    )

    # ----------------------------------------------------------------------
    # Load source dataset
    # ----------------------------------------------------------------------

    english_rows = load_english_rows()

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(
                "--limit must be greater than 0"
            )

        english_rows = english_rows[
            : args.limit
        ]

    print(
        "English pairs:",
        len(english_rows),
    )

    # ----------------------------------------------------------------------
    # Optional: re-run current QC over the checkpoint
    # ----------------------------------------------------------------------

    if args.revalidate:
        revalidate_checkpoint(
            checkpoint,
            english_rows,
        )

    # ----------------------------------------------------------------------
    # Batch mode
    # ----------------------------------------------------------------------

    if args.mode == "batch":
        all_terminal = run_batch_mode(
            english_rows,
            checkpoint,
            state_path,
            failed_log,
            args.batch_poll_timeout,
        )

        if not all_terminal:
            print(
                f"Batch jobs are still processing on Groq's side "
                f"(state saved to {state_path}). "
                "Re-run this exact command to keep polling. "
                "Already finished rows are checkpointed and "
                "will not be resubmitted."
            )
            return

    # ----------------------------------------------------------------------
    # Online finishing pass
    # ----------------------------------------------------------------------

    done = load_checkpoint(checkpoint)
    failed_ids = load_failed_ids(failed_log)

    remaining = [
        row
        for row in english_rows
        if (
            row["source_id"] not in done
            and row["source_id"] not in failed_ids
        )
    ]

    if remaining:
        print(
            f"Translating {len(remaining):,} rows online "
            f"({len(english_rows) - len(remaining):,} already done)..."
        )

        asyncio.run(
            translate_all(
                remaining,
                checkpoint,
                failed_log,
                args.concurrency,
            )
        )

    # ----------------------------------------------------------------------
    # Assemble final output
    # ----------------------------------------------------------------------

    done = load_checkpoint(
        checkpoint
    )

    order = {
        row["source_id"]: i
        for i, row
        in enumerate(english_rows)
    }

    final_records = sorted(
        (
            rec
            for sid, rec
            in done.items()
            if sid in order
        ),
        key=lambda rec: order[
            rec["source_id"]
        ],
    )

    # Final output format: bare message arrays, exactly as requested.
    final_output = [rec["messages"] for rec in final_records]

    write_jsonl(
        output_path,
        final_output,
    )

    # ----------------------------------------------------------------------
    # Final coverage validation
    # ----------------------------------------------------------------------
    # NOTE: the final lines are bare [{"role": ...}, ...] arrays, NOT dicts
    # with source_id. The old code indexed row["source_id"] on each parsed
    # line and crashed with TypeError after every successful run.

    with output_path.open(
        encoding="utf-8"
    ) as handle:
        written = 0
        for line in handle:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            if not (
                isinstance(record, list)
                and len(record) == 2
                and isinstance(record[0], dict)
                and isinstance(record[1], dict)
                and record[0].get("role") == "user"
                and record[1].get("role") == "assistant"
            ):
                raise RuntimeError(
                    "Malformed line in final output; "
                    "the file may be corrupt."
                )

            written += 1

    if written != len(final_records):
        raise RuntimeError(
            f"Final output mismatch: wrote {written} lines for "
            f"{len(final_records)} records."
        )

    failed_ids = load_failed_ids(failed_log)
    missing = [
        row["source_id"]
        for row in english_rows
        if (
            row["source_id"] not in done
            and row["source_id"] not in failed_ids
        )
    ]

    skipped_in_run = sum(
        1
        for source_id in failed_ids
        if source_id in {row["source_id"] for row in english_rows}
    )

    print(
        f"Wrote {written:,}/"
        f"{len(english_rows):,} rows "
        f"to {output_path}"
    )

    if missing:
        print(
            f"{len(missing):,} rows remain unresolved. "
            f"Check {failed_log} and re-run after fixing the cause."
        )
        raise SystemExit(1)

    if failed_ids:
        print(
            f"Completed with {skipped_in_run:,} intentionally skipped rows. "
            f"See {failed_log} for their source_id and reason."
        )
    else:
        print(
            "All rows translated successfully."
        )
    print(
        f"Checkpoint: {checkpoint}"
    )
    print(
        "The checkpoint is safe to delete "
        "once you have verified the final output."
    )


if __name__ == "__main__":
    main()