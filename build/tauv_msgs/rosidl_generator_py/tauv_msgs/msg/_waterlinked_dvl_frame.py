# generated from rosidl_generator_py/resource/_idl.py.em
# with input from tauv_msgs:msg/WaterlinkedDvlFrame.idl
# generated code does not contain a copyright notice

# This is being done at the module level and not on the instance level to avoid looking
# for the same variable multiple times on each instance. This variable is not supposed to
# change during runtime so it makes sense to only look for it once.
from os import getenv

ros_python_check_fields = getenv('ROS_PYTHON_CHECK_FIELDS', default='')


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

# Member 'covariance'
# Member 'transducer_velocity'
# Member 'transducer_distance'
# Member 'transducer_rssi'
# Member 'transducer_nsd'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_WaterlinkedDvlFrame(type):
    """Metaclass of message 'WaterlinkedDvlFrame'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('tauv_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'tauv_msgs.msg.WaterlinkedDvlFrame')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__waterlinked_dvl_frame
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__waterlinked_dvl_frame
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__waterlinked_dvl_frame
            cls._TYPE_SUPPORT = module.type_support_msg__msg__waterlinked_dvl_frame
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__waterlinked_dvl_frame

            from std_msgs.msg import Header
            if Header.__class__._TYPE_SUPPORT is None:
                Header.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class WaterlinkedDvlFrame(metaclass=Metaclass_WaterlinkedDvlFrame):
    """Message class 'WaterlinkedDvlFrame'."""

    __slots__ = [
        '_header',
        '_time',
        '_vx',
        '_vy',
        '_vz',
        '_fom',
        '_covariance',
        '_altitude',
        '_transducer_velocity',
        '_transducer_distance',
        '_transducer_rssi',
        '_transducer_nsd',
        '_transducer_beam_valid',
        '_velocity_valid',
        '_status',
        '_time_of_validity',
        '_time_of_transmission',
        '_check_fields',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'time': 'double',
        'vx': 'double',
        'vy': 'double',
        'vz': 'double',
        'fom': 'double',
        'covariance': 'double[9]',
        'altitude': 'double',
        'transducer_velocity': 'double[4]',
        'transducer_distance': 'double[4]',
        'transducer_rssi': 'double[4]',
        'transducer_nsd': 'double[4]',
        'transducer_beam_valid': 'boolean[4]',
        'velocity_valid': 'boolean',
        'status': 'int32',
        'time_of_validity': 'int64',
        'time_of_transmission': 'int64',
    }

    # This attribute is used to store an rosidl_parser.definition variable
    # related to the data type of each of the components the message.
    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 9),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 4),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 4),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 4),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 4),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('boolean'), 4),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int64'),  # noqa: E501
        rosidl_parser.definition.BasicType('int64'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        if 'check_fields' in kwargs:
            self._check_fields = kwargs['check_fields']
        else:
            self._check_fields = ros_python_check_fields == '1'
        if self._check_fields:
            assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
                'Invalid arguments passed to constructor: %s' % \
                ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.time = kwargs.get('time', float())
        self.vx = kwargs.get('vx', float())
        self.vy = kwargs.get('vy', float())
        self.vz = kwargs.get('vz', float())
        self.fom = kwargs.get('fom', float())
        if 'covariance' not in kwargs:
            self.covariance = numpy.zeros(9, dtype=numpy.float64)
        else:
            self.covariance = numpy.array(kwargs.get('covariance'), dtype=numpy.float64)
            assert self.covariance.shape == (9, )
        self.altitude = kwargs.get('altitude', float())
        if 'transducer_velocity' not in kwargs:
            self.transducer_velocity = numpy.zeros(4, dtype=numpy.float64)
        else:
            self.transducer_velocity = numpy.array(kwargs.get('transducer_velocity'), dtype=numpy.float64)
            assert self.transducer_velocity.shape == (4, )
        if 'transducer_distance' not in kwargs:
            self.transducer_distance = numpy.zeros(4, dtype=numpy.float64)
        else:
            self.transducer_distance = numpy.array(kwargs.get('transducer_distance'), dtype=numpy.float64)
            assert self.transducer_distance.shape == (4, )
        if 'transducer_rssi' not in kwargs:
            self.transducer_rssi = numpy.zeros(4, dtype=numpy.float64)
        else:
            self.transducer_rssi = numpy.array(kwargs.get('transducer_rssi'), dtype=numpy.float64)
            assert self.transducer_rssi.shape == (4, )
        if 'transducer_nsd' not in kwargs:
            self.transducer_nsd = numpy.zeros(4, dtype=numpy.float64)
        else:
            self.transducer_nsd = numpy.array(kwargs.get('transducer_nsd'), dtype=numpy.float64)
            assert self.transducer_nsd.shape == (4, )
        self.transducer_beam_valid = kwargs.get(
            'transducer_beam_valid',
            [bool() for x in range(4)]
        )
        self.velocity_valid = kwargs.get('velocity_valid', bool())
        self.status = kwargs.get('status', int())
        self.time_of_validity = kwargs.get('time_of_validity', int())
        self.time_of_transmission = kwargs.get('time_of_transmission', int())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.get_fields_and_field_types().keys(), self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    if self._check_fields:
                        assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.header != other.header:
            return False
        if self.time != other.time:
            return False
        if self.vx != other.vx:
            return False
        if self.vy != other.vy:
            return False
        if self.vz != other.vz:
            return False
        if self.fom != other.fom:
            return False
        if any(self.covariance != other.covariance):
            return False
        if self.altitude != other.altitude:
            return False
        if any(self.transducer_velocity != other.transducer_velocity):
            return False
        if any(self.transducer_distance != other.transducer_distance):
            return False
        if any(self.transducer_rssi != other.transducer_rssi):
            return False
        if any(self.transducer_nsd != other.transducer_nsd):
            return False
        if self.transducer_beam_valid != other.transducer_beam_valid:
            return False
        if self.velocity_valid != other.velocity_valid:
            return False
        if self.status != other.status:
            return False
        if self.time_of_validity != other.time_of_validity:
            return False
        if self.time_of_transmission != other.time_of_transmission:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def header(self):
        """Message field 'header'."""
        return self._header

    @header.setter
    def header(self, value):
        if self._check_fields:
            from std_msgs.msg import Header
            assert \
                isinstance(value, Header), \
                "The 'header' field must be a sub message of type 'Header'"
        self._header = value

    @builtins.property
    def time(self):
        """Message field 'time'."""
        return self._time

    @time.setter
    def time(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'time' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'time' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._time = value

    @builtins.property
    def vx(self):
        """Message field 'vx'."""
        return self._vx

    @vx.setter
    def vx(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'vx' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'vx' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._vx = value

    @builtins.property
    def vy(self):
        """Message field 'vy'."""
        return self._vy

    @vy.setter
    def vy(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'vy' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'vy' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._vy = value

    @builtins.property
    def vz(self):
        """Message field 'vz'."""
        return self._vz

    @vz.setter
    def vz(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'vz' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'vz' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._vz = value

    @builtins.property
    def fom(self):
        """Message field 'fom'."""
        return self._fom

    @fom.setter
    def fom(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'fom' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'fom' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._fom = value

    @builtins.property
    def covariance(self):
        """Message field 'covariance'."""
        return self._covariance

    @covariance.setter
    def covariance(self, value):
        if self._check_fields:
            if isinstance(value, numpy.ndarray):
                assert value.dtype == numpy.float64, \
                    "The 'covariance' numpy.ndarray() must have the dtype of 'numpy.float64'"
                assert value.size == 9, \
                    "The 'covariance' numpy.ndarray() must have a size of 9"
                self._covariance = value
                return
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 9 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'covariance' field must be a set or sequence with length 9 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._covariance = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def altitude(self):
        """Message field 'altitude'."""
        return self._altitude

    @altitude.setter
    def altitude(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'altitude' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'altitude' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._altitude = value

    @builtins.property
    def transducer_velocity(self):
        """Message field 'transducer_velocity'."""
        return self._transducer_velocity

    @transducer_velocity.setter
    def transducer_velocity(self, value):
        if self._check_fields:
            if isinstance(value, numpy.ndarray):
                assert value.dtype == numpy.float64, \
                    "The 'transducer_velocity' numpy.ndarray() must have the dtype of 'numpy.float64'"
                assert value.size == 4, \
                    "The 'transducer_velocity' numpy.ndarray() must have a size of 4"
                self._transducer_velocity = value
                return
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 4 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'transducer_velocity' field must be a set or sequence with length 4 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._transducer_velocity = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def transducer_distance(self):
        """Message field 'transducer_distance'."""
        return self._transducer_distance

    @transducer_distance.setter
    def transducer_distance(self, value):
        if self._check_fields:
            if isinstance(value, numpy.ndarray):
                assert value.dtype == numpy.float64, \
                    "The 'transducer_distance' numpy.ndarray() must have the dtype of 'numpy.float64'"
                assert value.size == 4, \
                    "The 'transducer_distance' numpy.ndarray() must have a size of 4"
                self._transducer_distance = value
                return
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 4 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'transducer_distance' field must be a set or sequence with length 4 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._transducer_distance = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def transducer_rssi(self):
        """Message field 'transducer_rssi'."""
        return self._transducer_rssi

    @transducer_rssi.setter
    def transducer_rssi(self, value):
        if self._check_fields:
            if isinstance(value, numpy.ndarray):
                assert value.dtype == numpy.float64, \
                    "The 'transducer_rssi' numpy.ndarray() must have the dtype of 'numpy.float64'"
                assert value.size == 4, \
                    "The 'transducer_rssi' numpy.ndarray() must have a size of 4"
                self._transducer_rssi = value
                return
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 4 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'transducer_rssi' field must be a set or sequence with length 4 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._transducer_rssi = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def transducer_nsd(self):
        """Message field 'transducer_nsd'."""
        return self._transducer_nsd

    @transducer_nsd.setter
    def transducer_nsd(self, value):
        if self._check_fields:
            if isinstance(value, numpy.ndarray):
                assert value.dtype == numpy.float64, \
                    "The 'transducer_nsd' numpy.ndarray() must have the dtype of 'numpy.float64'"
                assert value.size == 4, \
                    "The 'transducer_nsd' numpy.ndarray() must have a size of 4"
                self._transducer_nsd = value
                return
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 4 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'transducer_nsd' field must be a set or sequence with length 4 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._transducer_nsd = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def transducer_beam_valid(self):
        """Message field 'transducer_beam_valid'."""
        return self._transducer_beam_valid

    @transducer_beam_valid.setter
    def transducer_beam_valid(self, value):
        if self._check_fields:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 4 and
                 all(isinstance(v, bool) for v in value) and
                 True), \
                "The 'transducer_beam_valid' field must be a set or sequence with length 4 and each value of type 'bool'"
        self._transducer_beam_valid = value

    @builtins.property
    def velocity_valid(self):
        """Message field 'velocity_valid'."""
        return self._velocity_valid

    @velocity_valid.setter
    def velocity_valid(self, value):
        if self._check_fields:
            assert \
                isinstance(value, bool), \
                "The 'velocity_valid' field must be of type 'bool'"
        self._velocity_valid = value

    @builtins.property
    def status(self):
        """Message field 'status'."""
        return self._status

    @status.setter
    def status(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'status' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'status' field must be an integer in [-2147483648, 2147483647]"
        self._status = value

    @builtins.property
    def time_of_validity(self):
        """Message field 'time_of_validity'."""
        return self._time_of_validity

    @time_of_validity.setter
    def time_of_validity(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'time_of_validity' field must be of type 'int'"
            assert value >= -9223372036854775808 and value < 9223372036854775808, \
                "The 'time_of_validity' field must be an integer in [-9223372036854775808, 9223372036854775807]"
        self._time_of_validity = value

    @builtins.property
    def time_of_transmission(self):
        """Message field 'time_of_transmission'."""
        return self._time_of_transmission

    @time_of_transmission.setter
    def time_of_transmission(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'time_of_transmission' field must be of type 'int'"
            assert value >= -9223372036854775808 and value < 9223372036854775808, \
                "The 'time_of_transmission' field must be an integer in [-9223372036854775808, 9223372036854775807]"
        self._time_of_transmission = value
