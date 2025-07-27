# generated from rosidl_generator_py/resource/_idl.py.em
# with input from tauv_msgs:msg/NavigationState.idl
# generated code does not contain a copyright notice

# This is being done at the module level and not on the instance level to avoid looking
# for the same variable multiple times on each instance. This variable is not supposed to
# change during runtime so it makes sense to only look for it once.
from os import getenv

ros_python_check_fields = getenv('ROS_PYTHON_CHECK_FIELDS', default='')


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_NavigationState(type):
    """Metaclass of message 'NavigationState'."""

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
                'tauv_msgs.msg.NavigationState')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__navigation_state
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__navigation_state
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__navigation_state
            cls._TYPE_SUPPORT = module.type_support_msg__msg__navigation_state
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__navigation_state

            from geometry_msgs.msg import Pose
            if Pose.__class__._TYPE_SUPPORT is None:
                Pose.__class__.__import_type_support__()

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


class NavigationState(metaclass=Metaclass_NavigationState):
    """Message class 'NavigationState'."""

    __slots__ = [
        '_header',
        '_body_pose',
        '_v_b',
        '_a_b',
        '_omega_b',
        '_check_fields',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'body_pose': 'geometry_msgs/Pose',
        'v_b': 'geometry_msgs/Vector3',
        'a_b': 'geometry_msgs/Vector3',
        'omega_b': 'geometry_msgs/Vector3',
    }

    # This attribute is used to store an rosidl_parser.definition variable
    # related to the data type of each of the components the message.
    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Pose'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Vector3'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Vector3'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Vector3'),  # noqa: E501
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
        from geometry_msgs.msg import Pose
        self.body_pose = kwargs.get('body_pose', Pose())
        from geometry_msgs.msg import Vector3
        self.v_b = kwargs.get('v_b', Vector3())
        from geometry_msgs.msg import Vector3
        self.a_b = kwargs.get('a_b', Vector3())
        from geometry_msgs.msg import Vector3
        self.omega_b = kwargs.get('omega_b', Vector3())

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
        if self.body_pose != other.body_pose:
            return False
        if self.v_b != other.v_b:
            return False
        if self.a_b != other.a_b:
            return False
        if self.omega_b != other.omega_b:
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
    def body_pose(self):
        """Message field 'body_pose'."""
        return self._body_pose

    @body_pose.setter
    def body_pose(self, value):
        if self._check_fields:
            from geometry_msgs.msg import Pose
            assert \
                isinstance(value, Pose), \
                "The 'body_pose' field must be a sub message of type 'Pose'"
        self._body_pose = value

    @builtins.property
    def v_b(self):
        """Message field 'v_b'."""
        return self._v_b

    @v_b.setter
    def v_b(self, value):
        if self._check_fields:
            from geometry_msgs.msg import Vector3
            assert \
                isinstance(value, Vector3), \
                "The 'v_b' field must be a sub message of type 'Vector3'"
        self._v_b = value

    @builtins.property
    def a_b(self):
        """Message field 'a_b'."""
        return self._a_b

    @a_b.setter
    def a_b(self, value):
        if self._check_fields:
            from geometry_msgs.msg import Vector3
            assert \
                isinstance(value, Vector3), \
                "The 'a_b' field must be a sub message of type 'Vector3'"
        self._a_b = value

    @builtins.property
    def omega_b(self):
        """Message field 'omega_b'."""
        return self._omega_b

    @omega_b.setter
    def omega_b(self, value):
        if self._check_fields:
            from geometry_msgs.msg import Vector3
            assert \
                isinstance(value, Vector3), \
                "The 'omega_b' field must be a sub message of type 'Vector3'"
        self._omega_b = value
