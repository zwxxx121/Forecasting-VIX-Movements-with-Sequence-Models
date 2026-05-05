from __future__ import annotations

import copy
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

FOMC_MEETINGS: List[Tuple[str, bool]] = [
    ('20060131', False), ('20060328', False), ('20060510', False),
    ('20060629', False), ('20060920', False), ('20061025', False),
    ('20061212', False), ('20070131', False), ('20070321', False),
    ('20070509', False), ('20070618', False), ('20070807', False),
    ('20070918', False), ('20071031', False), ('20071211', False),
    ('20080122', False), ('20080130', False), ('20080318', False),
    ('20080430', False), ('20080625', False), ('20080805', False),
    ('20080916', False), ('20081008', False), ('20081029', False),
    ('20081216', False), ('20090128', False), ('20090318', False),
    ('20090429', False), ('20090624', False), ('20090812', False),
    ('20090923', False), ('20091104', False), ('20091216', False),
    ('20100127', False), ('20100316', False), ('20100428', False),
    ('20100623', False), ('20100810', False), ('20100921', False),
    ('20101103', False), ('20101214', False), ('20110126', False),
    ('20110315', False), ('20110427', True), ('20110622', True),
    ('20110809', False), ('20110921', True), ('20111102', False),
    ('20111213', True), ('20120125', True), ('20120313', False),
    ('20120425', True), ('20120620', True), ('20120801', False),
    ('20120913', True), ('20121024', False), ('20121212', True),
    ('20130130', False), ('20130320', True), ('20130501', False),
    ('20130619', True), ('20130731', False), ('20130918', True),
    ('20131030', False), ('20131218', True), ('20140129', False),
    ('20140319', True), ('20140430', False), ('20140618', True),
    ('20140730', False), ('20140917', True), ('20141029', False),
    ('20141217', True), ('20150128', False), ('20150318', True),
    ('20150429', False), ('20150617', True), ('20150729', False),
    ('20150917', True), ('20151028', False), ('20151216', True),
    ('20160127', False), ('20160316', True), ('20160427', False),
    ('20160615', True), ('20160727', False), ('20160921', True),
    ('20161102', False), ('20161214', True), ('20170201', False),
    ('20170315', True), ('20170503', False), ('20170614', True),
    ('20170726', False), ('20170920', True), ('20171101', False),
    ('20171213', True), ('20180131', False), ('20180321', True),
    ('20180502', False), ('20180613', True), ('20180801', False),
    ('20180926', True), ('20181108', False), ('20181219', True),
    ('20190130', False), ('20190320', True), ('20190501', False),
    ('20190619', True), ('20190731', False), ('20190918', True),
    ('20191030', False), ('20191211', True), ('20200129', False),
    ('20200303', False), ('20200315', False), ('20200429', False),
    ('20200610', True), ('20200729', False), ('20200916', True),
    ('20201105', False), ('20201216', True), ('20210127', False),
    ('20210317', True), ('20210428', False), ('20210616', True),
    ('20210728', False), ('20210922', True), ('20211103', False),
    ('20211215', True), ('20220126', False), ('20220316', True),
    ('20220504', False), ('20220615', True), ('20220727', False),
    ('20220921', True), ('20221102', False), ('20221214', True),
    ('20230201', False), ('20230322', True), ('20230503', False),
    ('20230614', True), ('20230726', False), ('20230920', True),
    ('20231101', False), ('20231213', True), ('20240131', False),
    ('20240320', True), ('20240501', False), ('20240612', True),
    ('20240731', False), ('20240918', True), ('20241107', False),
    ('20241218', True), ('20250129', False), ('20250319', True),
    ('20250507', False), ('20250618', True), ('20250730', False),
    ('20250917', True), ('20251029', False), ('20251210', True),
    ('20260128', False), ('20260318', True)
]

BASE_FOMC_URL = "https://www.federalreserve.gov/newsevents/pressreleases/monetary{}a.htm"

BOILERPLATE_PATTERNS = [
    r"Voting for the monetary policy action.*",
    r"Implementation Note.*",
    r"For release at.*",
    r"\*\s*\*\s*\*",
]

LEXICONS: Dict[str, List[str]] = {
    "hawkish": [
        "raise", "hike", "tighten", "restrictive", "elevated", "persistent",
        "above target", "upside risks", "inflation remains", "further increases",
        "not yet", "remain firm", "additional firming",
    ],
    "dovish": [
        "cut", "lower", "easing", "accommodative", "supportive", "slow",
        "softening", "below target", "downside risks", "pause", "patient",
        "gradual", "modest", "progress toward",
    ],
    "uncertainty": [
        "uncertain", "uncertainty", "risk", "risks", "global", "stress",
        "financial conditions", "volatile", "monitor", "closely watching",
        "remain attentive", "geopolitical",
    ],
    "inflation": [
        "inflation", "price", "prices", "cpi", "pce", "price stability",
        "price pressures", "disinflation", "2 percent", "above 2",
    ],
    "labor": [
        "employment", "unemployment", "labor market", "job", "jobs",
        "payroll", "wage", "wages", "job gains", "labor force",
        "maximum employment", "labor demand",
    ],
}

RATE_DECISION_PATTERNS = {
    "hike": [
        r"raise.{0,30}target range",
        r"increas.{0,30}federal funds",
        r"federal funds rate.{0,30}to \d",
    ],
    "cut": [
        r"lower.{0,30}target range",
        r"decreas.{0,30}federal funds",
        r"reduc.{0,30}target range",
    ],
    "hold": [
        r"maintain.{0,30}target range",
        r"hold.{0,30}federal funds",
        r"leave.{0,30}unchanged",
    ],
}

DECISION_MAP = {"cut": -1, "hold": 0, "hike": 1, "unknown": 0}
