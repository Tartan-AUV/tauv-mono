# generated from rosidl_generator_py/resource/_idl.py.em
# with input from tauv_msgs:msg/RpmCommand.idl
# generated code does not contain a copyright notice

# This is being done at the module level and not on the instance level to avoid looking
# for the same variable multiple times on each instance. This variable is not supposed to
# change during runtime so it makes sense to only look for it once.
from os import getenv

ros_python_check_fields = getenv('ROS_PYTHON_CHECK_FIELDS', default='')


# Import statements for member types

import builtins  # noqa: E402, I100

# Member 'rpms'
# Member 'enables'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_RpmCommand(type):
    """Metaclass of message 'RpmCommand'."""

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
                'tauv_msgs.msg.RpmCommand')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__rpm_command
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__rpm_command
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__rpm_command
            cls._TYPE_SUPPORT = module.type_support_msg__msg__rpm_command
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__rpm_command

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class RpmCommand(metaclass=Metaclass_RpmCommand):
    """Message class 'RpmCommand'."""

    __slots__ = [
        '_rpms',
        '_enables',
        '_check_fields',
    ]

    _fields_and_field_types = {
        'rpms': 'int32[8]',
        'enables': 'uint8[8]',
    }

    # This attribute is used to store an rosidl_parser.definition variable
    # related to the data type of each of the components the message.
    SLOT_TYPES = (
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('int32'), 8),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('uint8'), 8),  # noqa: E501
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
        if 'rpms' not in kwargs:
            self.rpms = numpy.zeros(8, dtype=numpy.int32)
        else:
            self.rpms = numpy.array(kwargs.get('rpms'), dtype=numpy.int32)
            assert self.rpms.shape == (8, )
        if 'enables' not in kwargs:
            self.enables = numpy.zeros(8, dtype=numpy.uint8)
        else:
            self.enables = numpy.array(kwargs.get('enables'), dtype=numpy.uint8)
            assert self.enables.shape == (8, )

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
        if any(self.rpms != other.rpms):
            return False
        if any(self.enables != other.enables):
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def rpms(self):
        """Message field 'rpms'."""
        return self._rpms

    @rpms.setter
    def rpms(self, value):
        if self._check_fields:
            if isinstance(value, numpy.ndarray):
                assert value.dtype == numpy.int32, \
                    "The 'rpms' numpy.ndarray() must have the dtype of 'numpy.int32'"
                assert value.size == 8, \
                    "The 'rpms' numpy.ndarray() must have a size of 8"
                self._rpms = value
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
                 len(value) == 8 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= -2147483648 and val < 2147483648 for val in value)), \
                "The 'rpms' field must be a set or sequence with length 8 and each value of type 'int' and each integer in [-2147483648, 2147483647]"
        self._rpms = numpy.array(value, dtype=numpy.int32)

    @builtins.property
    def enables(self):
        """Message field 'enables'."""
        return self._enables

    @enables.setter
    def enables(self, value):
        if self._check_fields:
            if isinstance(value, numpy.ndarray):
                assert value.dtype == numpy.uint8, \
                    "The 'enables' numpy.ndarray() must have the dtype of 'numpy.uint8'"
                assert value.size == 8, \
                    "The 'enables' numpy.ndarray() must have a size of 8"
                self._enables = value
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
                 len(value) == 8 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'enables' field must be a set or sequence with length 8 and each value of type 'int' and each unsigned integer in [0, 255]"
        self._enables = numpy.array(value, dtype=numpy.uint8)
