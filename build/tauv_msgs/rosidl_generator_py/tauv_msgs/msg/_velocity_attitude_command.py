# generated from rosidl_generator_py/resource/_idl.py.em
# with input from tauv_msgs:msg/VelocityAttitudeCommand.idl
# generated code does not contain a copyright notice

# This is being done at the module level and not on the instance level to avoid looking
# for the same variable multiple times on each instance. This variable is not supposed to
# change during runtime so it makes sense to only look for it once.
from os import getenv

ros_python_check_fields = getenv('ROS_PYTHON_CHECK_FIELDS', default='')


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_VelocityAttitudeCommand(type):
    """Metaclass of message 'VelocityAttitudeCommand'."""

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
                'tauv_msgs.msg.VelocityAttitudeCommand')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__velocity_attitude_command
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__velocity_attitude_command
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__velocity_attitude_command
            cls._TYPE_SUPPORT = module.type_support_msg__msg__velocity_attitude_command
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__velocity_attitude_command

            from geometry_msgs.msg import Quaternion
            if Quaternion.__class__._TYPE_SUPPORT is None:
                Quaternion.__class__.__import_type_support__()

            from geometry_msgs.msg import Vector3
            if Vector3.__class__._TYPE_SUPPORT is None:
                Vector3.__class__.__import_type_support__()

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


class VelocityAttitudeCommand(metaclass=Metaclass_VelocityAttitudeCommand):
    """Message class 'VelocityAttitudeCommand'."""

    __slots__ = [
        '_header',
        '_target_velocity',
        '_target_attitude',
        '_feedforward_acceleration',
        '_velocity_control_enabled',
        '_attitude_control_enabled',
        '_check_fields',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'target_velocity': 'geometry_msgs/Vector3',
        'target_attitude': 'geometry_msgs/Quaternion',
        'feedforward_acceleration': 'geometry_msgs/Vector3',
        'velocity_control_enabled': 'boolean',
        'attitude_control_enabled': 'boolean',
    }

    # This attribute is used to store an rosidl_parser.definition variable
    # related to the data type of each of the components the message.
    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Vector3'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Quaternion'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Vector3'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
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
        from geometry_msgs.msg import Vector3
        self.target_velocity = kwargs.get('target_velocity', Vector3())
        from geometry_msgs.msg import Quaternion
        self.target_attitude = kwargs.get('target_attitude', Quaternion())
        from geometry_msgs.msg import Vector3
        self.feedforward_acceleration = kwargs.get('feedforward_acceleration', Vector3())
        self.velocity_control_enabled = kwargs.get('velocity_control_enabled', bool())
        self.attitude_control_enabled = kwargs.get('attitude_control_enabled', bool())

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
        if self.target_velocity != other.target_velocity:
            return False
        if self.target_attitude != other.target_attitude:
            return False
        if self.feedforward_acceleration != other.feedforward_acceleration:
            return False
        if self.velocity_control_enabled != other.velocity_control_enabled:
            return False
        if self.attitude_control_enabled != other.attitude_control_enabled:
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
    def target_velocity(self):
        """Message field 'target_velocity'."""
        return self._target_velocity

    @target_velocity.setter
    def target_velocity(self, value):
        if self._check_fields:
            from geometry_msgs.msg import Vector3
            assert \
                isinstance(value, Vector3), \
                "The 'target_velocity' field must be a sub message of type 'Vector3'"
        self._target_velocity = value

    @builtins.property
    def target_attitude(self):
        """Message field 'target_attitude'."""
        return self._target_attitude

    @target_attitude.setter
    def target_attitude(self, value):
        if self._check_fields:
            from geometry_msgs.msg import Quaternion
            assert \
                isinstance(value, Quaternion), \
                "The 'target_attitude' field must be a sub message of type 'Quaternion'"
        self._target_attitude = value

    @builtins.property
    def feedforward_acceleration(self):
        """Message field 'feedforward_acceleration'."""
        return self._feedforward_acceleration

    @feedforward_acceleration.setter
    def feedforward_acceleration(self, value):
        if self._check_fields:
            from geometry_msgs.msg import Vector3
            assert \
                isinstance(value, Vector3), \
                "The 'feedforward_acceleration' field must be a sub message of type 'Vector3'"
        self._feedforward_acceleration = value

    @builtins.property
    def velocity_control_enabled(self):
        """Message field 'velocity_control_enabled'."""
        return self._velocity_control_enabled

    @velocity_control_enabled.setter
    def velocity_control_enabled(self, value):
        if self._check_fields:
            assert \
                isinstance(value, bool), \
                "The 'velocity_control_enabled' field must be of type 'bool'"
        self._velocity_control_enabled = value

    @builtins.property
    def attitude_control_enabled(self):
        """Message field 'attitude_control_enabled'."""
        return self._attitude_control_enabled

    @attitude_control_enabled.setter
    def attitude_control_enabled(self, value):
        if self._check_fields:
            assert \
                isinstance(value, bool), \
                "The 'attitude_control_enabled' field must be of type 'bool'"
        self._attitude_control_enabled = value
