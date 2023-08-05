from typing import Optional, Type
from missions.mission import Mission
import missions

def get_mission_by_name(name: str) -> Optional[Type[Mission]]:
    missions_by_name = {
        "kf_transdec_23": missions.kf_transdec_23_buoy_search.KFTransdec23,
    }
    return missions_by_name.get(name)
