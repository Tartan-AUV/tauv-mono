class Coord:
    def __init__(self, x=0, y=0, z=0, deg=None):
        self.x = x
        self.y = y
        self.z = z
        self.deg = deg

class Course:
    def __init__(self, _wall_t_rear, _rear_t_course, _wall_t_ref, _ref_t_gate, 
                 _ref_t_bouy, _bouy_t_approach, _ref_t_octogon, 
                 _octogon_t_approach, _ref_t_torpedo, _torpedo_t_approach, 
                 _ref_t_marker, _marker_t_approach, _gate_t_style):
        self._wall_t_rear = Coord(**_wall_t_rear)
        self._rear_t_course = Coord(**_rear_t_course)
        self._wall_t_ref = Coord(**_wall_t_ref)
        self._ref_t_gate = Coord(**_ref_t_gate)
        self._ref_t_bouy = Coord(**_ref_t_bouy)
        self._bouy_t_approach = Coord(**_bouy_t_approach)
        self._ref_t_octagon = Coord(**_ref_t_octogon)
        self._octogon_t_approach = Coord(**_octogon_t_approach)
        self._ref_t_torpedo = Coord(**_ref_t_torpedo)
        self._torpedo_t_approach = Coord(**_torpedo_t_approach)
        self._ref_t_marker = Coord(**_ref_t_marker)
        self._marker_t_approach = Coord(**_marker_t_approach)
        self._gate_t_style = Coord(**_marker_t_approach)


