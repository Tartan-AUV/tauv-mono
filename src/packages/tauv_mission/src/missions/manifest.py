from typing import Optional, Type
from missions.mission import Mission
import missions

def get_mission_by_name(name: str) -> Optional[Type[Mission]]:
    missions_by_name = {
        "kf_transdec_23": missions.kf_transdec_23_buoy_search.KFTransdec23,
        "kf_transdec_23_survey": missions.kf_transdec_23_survey.KFTransdec23,
        "kf_prequal_24": missions.kf_prequal_24.KFPrequal24,
    }
    return missions_by_name.get(name)
