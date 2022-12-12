from dataclasses import dataclass, field
from typing import List

@dataclass
class KlueDPInputExample:
    guid: str = ""
    text: str = ""
    sent_id: int = -1
    token_id: int = -1
    token: str = ""
    pos: str = ""
    head: str = ""
    dep: str = ""

@dataclass
class KlueDPInputFeatures:
    guid: str = ""
    ids: List[int] = field(default_factory=list)
    mask: List[int] = field(default_factory=list)
    bpe_head_mask: List[int] = field(default_factory=list)
    bpe_tail_mask: List[int] = field(default_factory=list)
    head_ids: List[int] = field(default_factory=list)
    dep_ids: List[int] = field(default_factory=list)
    pos_ids: List[int] = field(default_factory=list)


dep_labels = [
        "NP",
        "NP_AJT",
        "VP",
        "NP_SBJ",
        "VP_MOD",
        "NP_OBJ",
        "AP",
        "NP_CNJ",
        "NP_MOD",
        "VNP",
        "DP",
        "VP_AJT",
        "VNP_MOD",
        "NP_CMP",
        "VP_SBJ",
        "VP_CMP",
        "VP_OBJ",
        "VNP_CMP",
        "AP_MOD",
        "X_AJT",
        "VP_CNJ",
        "VNP_AJT",
        "IP",
        "X",
        "X_SBJ",
        "VNP_OBJ",
        "VNP_SBJ",
        "X_OBJ",
        "AP_AJT",
        "L",
        "X_MOD",
        "X_CNJ",
        "VNP_CNJ",
        "X_CMP",
        "AP_CMP",
        "AP_SBJ",
        "R",
        "NP_SVJ",
    ]

pos_labels = [
        "NNG",
        "NNP",
        "NNB",
        "NP",
        "NR",
        "VV",
        "VA",
        "VX",
        "VCP",
        "VCN",
        "MMA",
        "MMD",
        "MMN",
        "MAG",
        "MAJ",
        "JC",
        "IC",
        "JKS",
        "JKC",
        "JKG",
        "JKO",
        "JKB",
        "JKV",
        "JKQ",
        "JX",
        "EP",
        "EF",
        "EC",
        "ETN",
        "ETM",
        "XPN",
        "XSN",
        "XSV",
        "XSA",
        "XR",
        "SF",
        "SP",
        "SS",
        "SE",
        "SO",
        "SL",
        "SH",
        "SW",
        "SN",
        "NA",
    ]