from typing import Optional
from spatialmath import SE3, SO3, SE2
from math import pi
import yaml

from actuator_client import ActuatorClient
from missions.mission import Mission
from missions.coordinates import Course
from tasks.task import Task, TaskResult
from tasks import dive, gate_dead_reckon, goto, scan_translate, torpedo_24, collect_sample, goto_relative_with_depth, buoy_24, surface, buoy_24_dead_reckon
from enum import IntEnum

# variables 
COURSE = 'finals_sample' # from irvine_24.yaml: 'finals_sample', 'finals_practice', or 'finals'
GATE_SIDE = -1 # -1 = left, 1 = right

with open('irvine_24.yaml', 'r') as file:
    data = yaml.safe_load(file)
coords = Course(**data[COURSE])

class State(IntEnum):
    # start run
    START = 0
    DIVE = 1
    GATE_DEAD_RECKON = 2

    # buoy
    BUOY_GOTO = 3
    BUOY_SURVEY = 4
    BUOY_VISION = 5
    BUOY_DEAD_RECKON = 6

    # torpedo
    TORPEDO_GOTO = 7
    TORPEDO_SURVEY = 8
    TORPEDO_VISION = 9
    TORPEDO_DEAD_RECKON = 10

    # octagon
    OCTAGON_GOTO = 11
    OCTAGON_SURFACE = 12
    OCTAGON_DIVE = 13
    SAMPLE_SURVEY = 14
    SAMPLE_PICKUP_VISION = 15
    SAMPLE_PICKUP_DEAD_RECKON = 16
    SAMPLE_DROP = 17

    # marker
    MARKER_GOTO = 18
    MARKER_SURVEY = 19
    MARKER_VISION = 20
    MARKER_DEAD_RECKON = 21

    # end run
    STYLE_GOTO = 22
    STYLE = 23
    FINAL_SURFACE = 24

class KFIrvineFinals(Mission):
    def __init__(self):
        # initializing
        self._state = State.START
        self._pinger = None # used for torpedo/octagon logic: True = torpedo, False = octagon
        self._sphincter = None # used to track open/closed state of sphincter: True = open, False = closed
        self._actuators = ActuatorClient()

        # transformation of coordinates to course frame
        self._wall_t_rear: SE3 = SE3.Rt(SO3(), (coords._wall_t_rear.x, coords._wall_t_rear.y, coords._wall_t_rear.z))
        self._rear_t_course: SE3 = SE3.Rt(SO3(), (coords._rear_t_course.x, coords._rear_t_course.y, coords._rear_t_course.z))
        self._wall_t_course: SE3 = self._wall_t_vehicle_rear * self._vehicle_rear_t_course
        self._wall_t_ref: SE3 = SE3.Rt(SO3(), (coords._wall_t_ref.x, coords._wall_t_ref.y, coords._wall_t_ref.z))
        self._ref_t_gate: SE3 = SE3.Rt(SO3(), (coords._ref_t_gate.x, coords._ref_t_gate.y, coords._ref_t_gate.z))
        self._wall_t_gate: SE3 = self._wall_t_gate * SE3.Rt(SO3.Rz(coords._ref_t_gate.deg, unit='deg'), [0, 0, 0])
        self._course_t_ref = self._wall_t_course.inv() * self._wall_t_ref
        self._course_t_gate = self._wall_t_course.inv() * self._wall_t_gate
        self._gate_offset_y = GATE_SIDE*0.75

        # buoy
        self._ref_t_bouy: SE3 = SE3.Rt(SO3(), (coords._ref_t_bouy.x, coords._ref_t_bouy.y, coords._ref_t_bouy.z))
        self._course_t_buoy = self._course_t_ref * self._ref_t_bouy
        self._course_t_buoy_approach = self._course_t_buoy * SE3.Rt(SO3.Rz(-1.57), (0, 3, 0))
        self._buoy_scan_points = [
            (-2, 0),
            (2, 0),
            (-2, 0.5),
            (2, 0.5),
        ]

        # torpedo
        self._ref_t_torpedo: SE3 = SE3.Rt(SO3.Rz(coords._ref_t_torpedo.deg, unit='deg'), (coords._ref_t_torpedo.x, coords._ref_t_torpedo.y, coords._ref_t_torpedo.z))
        self._course_t_torpedo = self._course_t_ref * self._ref_t_torpedo
        self._course_t_torpedo_approach = None  
        # this point is used as the destination of TORPEDO_GOTO, the start of 
        # the scan in TORPEDO_SURVEY, and the location torpedos are shot from  
        # in TORPEDO_DEAD_RECKON (which will likely not all be the same thing)
        self._torpedo_scan_points = [
            (-2, 0),
            (2, 0),
            (-2, 0.5),
            (2, 0.5),
        ]

        # octagon
        self._ref_t_octagon: SE3 = SE3.Rt(SO3(), (coords._ref_t_octagon.x, coords._ref_t_octagon.y, coords._ref_t_octagon.z))
        self._course_t_octagon = self._course_t_ref * self._ref_t_octagon
        self._course_t_octagon_approach = self._course_t_octagon * SE3.Rt(SO3(), (0, 0, 0.75))
        # this point is used as the destination of OCTAGON_GOTO, the start of 
        # the scan in SAMPLE_SURVEY, and the location torpedos are shot from in 
        # SAMPLE_PICKUP_DEAD_RECKON (which will likely not all be the same thing)
        # from semi run: self._course_t_octagon_approach_face_buoy = self._course_t_octagon_approach * SE3.Rt(SO3(), (-6.86,0,0))
        self._octagon_scan_points = [
            (-2, 0),
            (2, 0),
            (-2, 0.5),
            (2, 0.5),
        ]

        # marker
        self._ref_t_marker: SE3 = SE3.Rt(SO3(), (coords._ref_t_marker.x, coords._ref_t_marker.y, coords._ref_t_marker.z))
        self._course_t_marker = self._course_t_ref * self._ref_t_marker
        self._course_t_marker_approach = None  
        # this point is used as the destination of MARKER_GOTO, the start of 
        # the scan in MARKER_SURVEY, and the location torpedos are shot from in 
        # MARKER_DEAD_RECKON (which will likely not all be the same thing)
        self._marker_scan_points = [
            (-2, 0),
            (2, 0),
            (-2, 0.5),
            (2, 0.5),
        ]

        # style 
        self._course_t_style = None

    def entrypoint(self) -> Optional[Task]:
        self._state = State.DIVE
        return dive.Dive(20.0, 2.0, 0.0) 
    
    def transition(self, task: Task, task_result: TaskResult) -> Optional[Task]:
        
        ########### START RUN ############
        if self._state == State.DIVE:
            if task_result.status == dive.DiveStatus.SUCCESS:
                self._state = State.GATE_DEAD_RECKON
                return gate_dead_reckon.Gate(course_t_gate=self._course_t_gate, travel_offset_y=self._gate_offset_y)

        elif self._state == State.GATE_DEAD_RECKON:
            if task_result.status == gate_dead_reckon.GateStatus.SUCCESS:
                self._state = State.TORPEDO_GOTO
                return goto.Goto(self._course_t_torpedo_approach, in_course=True)
        
        ########### TORPEDO ############
        elif self._state == State.TORPEDO_GOTO:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.TORPEDO_SURVEY
                return scan_translate.ScanTranslate(
                    course_t_start=self._course_t_torpedo_approach, points=self._buoy_scan_points, clear=True)
        
        elif self._state == State.TORPEDO_SURVEY:
            if task_result.status == scan_translate.ScanTranslateStatus.SUCCESS:
                self._state = State.TORPEDO_VISION
                return torpedo_24.Torpedo24(timeout=None, frequency=None, dry_run=None) # VISION TORPEDO
        
        elif self._state == State.TORPEDO_VISION:
            if task_result.status == torpedo_24.Torpedo24Status.SUCCESS:
                self._state = State.OCTAGON_GOTO
                return goto.Goto(self._course_t_octagon_approach, in_course=True)
            else:
                self._state == State.TORPEDO_DEAD_RECKON
                return goto.Goto(self._course_t_torpedo_approach, in_course=True)

        elif self._state == State.TORPEDO_DEAD_RECKON:
            if task_result.status == goto.GotoStatus.SUCCESS:
                # CALL TORPEDO DEAD RECKON TASK
                self._state = State.OCTAGON_GOTO
                return goto.Goto(self._course_t_octagon_approach, in_course=True)
            
        ########### OCTAGON ############
        elif self._state == State.OCTAGON_GOTO:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.OCTAGON_SURVEY
                return scan_bottom.ScanBottom(course_t_start=self._course_t_octagon_approach, points=self._octagon_scan_points, clear=True)
            
        elif self._state == State.OCTAGON_SURVEY:
            if task_result.status == scan_bottom.ScanBottomSuccess.SUCCESS:
                self._state = State.SAMPLE_PICKUP_VISION
                return collect_sample.CollectSampleStatus(timeout=None, frequency=None, dry_run=None) # VISION SAMPLE PICKUP
        
        elif self._state == State.SAMPLE_PICKUP_VISION:
            if task_result.status == collect_sample.CollectSampleStatus.SUCCESS:
                self._sphincter = True
                self._tate = State.OCTAGON_SURFACE
                return goto_relative_with_depth.GotoRelativeWithDepth(SE2(), 0)
            else:   
                self._state = State.SAMPLE_PICKUP_DEAD_RECKON
                return goto.Goto(self._course_t_octagon_approach, in_course=True)
            
        elif self._state == State.SAMPLE_PICKUP_DEAD_RECKON:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.OCTAGON_SURFACE
                return goto_relative_with_depth.GotoRelativeWithDepth(SE2(), 0)

        elif self._state == State.OCTAGON_SURFACE:
            if task_result.status == goto_relative_with_depth.GotoRelativeWithDepthStatus.SUCCESS:
                self._state = State.OCTAGON_DIVE
                return goto.Goto(self._course_t_octagon_approach, in_course=True)

        elif self._state == State.OCTAGON_DIVE:
            if task_result.status == goto.GotoStatus.SUCCESS:
                if self._sphincter:                            # DROP
                    self._actuators.open_sphincter()
                    self._sphincter = False
                self._state = State.MARKER_GOTO
                return goto.Goto(self._course_t_marker_approach, in_course=True)
            
# can insert some way to pick up second piece if the first was retrieved with vision and if it is seen   
       
        ########### MARKER ############
        elif self._state == State.MARKER_GOTO:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.MARKER_SURVEY
                return scan_bottom.ScanBottom(course_t_start=self._course_t_marker_approach, points=self._marker_scan_points, clear=True)

        elif self._state == State.MARKER_SURVEY:
            if task_result.status == scan_bottom.ScanBottomStatus.SUCCESS:
                self._state = State.MARKER_VISION
                return marker_dropper_24.MarkerDropper24() # VISION MARKER
        
        elif self._state == State.MARKER_VISION:
            if task_result.status == marker_dropper_24.MarkerDropper24Status.SUCCESS:
                self._state = State.BUOY_GOTO
                return goto.Goto(self._course_t_buoy_approach, in_course=True)
            else:
                self._state == State.MARKER_DEAD_RECKON
                return goto.Goto(self._course_t_marker_approach, in_course=True)

        elif self._state == State.MARKER_DEAD_RECKON:
            if task_result.status == goto.GotoStatus.SUCCESS:
                # CALL MARKER DEAD RECKON TASK
                self._state = State.BUOY_GOTO
                return goto.Goto(self._course_t_buoy_approach, in_course=True)
            

        ########### BUOY ############
        elif self._state == State.BUOY_GOTO:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.BUOY_SURVEY
                return scan_translate.ScanTranslate(course_t_start=self._course_t_buoy_approach, points=self._buoy_scan_points, clear=True)

        elif self._state == State.BUOY_SURVEY:
            if task_result.status == scan_translate.ScanTranslateStatus.SUCCESS:
                self._state = State.BUOY_VISION
                return buoy_24.CircleBuoy(tag='buoy', circle_radius=1.5, circle_ccw=True,
                 waypoint_every_n_meters=0.5, stare_timeout_s=8, circle_depth=0.7, latch_buoy=False) # VISION BUOY
        
        elif self._state == State.BUOY_VISION:
            if task_result.status == buoy_24.CircleBuoyStatus.SUCCESS:
                self._state = State.STYLE_GOTO
                return goto.Goto(self._course_t_style, in_course=True)
            else:
                self._state == State.BUOY_DEAD_RECKON
                return goto.Goto(self._course_t_buoy_approach, in_course=True)

        elif self._state == State.MARKER_DEAD_RECKON:
            if task_result.status == goto.GotoStatus.SUCCESS:
                # CALL MARKER DEAD RECKON TASK
                self._state = State.STYLE_GOTO
                return goto.Goto(self._course_t_style, in_course=True)
        
        ########### STYLE ############
        elif self._state == State.STYLE_GOTO:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.STYLE
                return barrel_roll.BarrelRoll()
            
        elif self._state == State.STYLE:
            if task_result.status == barrel_roll.BarrelRollStatus.SUCCESS:
                self._state = State.FINAL_SURFACE
                return surface.Surface()

        else:
            return None


