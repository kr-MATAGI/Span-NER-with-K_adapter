from dataclasses import dataclass, field
from typing import List, Tuple

#============================================================
@dataclass
class EntityItem:
    word: str = ""
    start_idx: int = -1
    end_idx: int = -1
    type: str = ""

@dataclass
class KlueRE_Item:
    idx: int = -1
    guid: str = ""
    sentence: str = ""
    subj_entity: EntityItem = EntityItem()
    obj_entity: EntityItem = EntityItem()
    label: str = ""
    source: str = ""

klue_re_rel = [
    "no_relation",
    "org:dissolved",
    "org:founded",
    "org:place_of_headquarters",
    "org:alternate_names",
    "org:member_of",
    "org:members",
    "org:political/religious_affiliation",
    "org:product",
    "org:founded_by",
    "org:top_members/employees",
    "org:number_of_employees/members",
    "per:date_of_birth",
    "per:date_of_death",
    "per:place_of_birth",
    "per:place_of_death",
    "per:place_of_residence",
    "per:origin",
    "per:employee_of",
    "per:schools_attended",
    "per:alternate_names",
    "per:parents",
    "per:children",
    "per:siblings",
    "per:spouse",
    "per:other_family",
    "per:colleagues",
    "per:product",
    "per:religion",
    "per:title"
]


### MAIN ###
if "__main__" == __name__:
    print(f"[klue_dclass][__main__] MAIN!")