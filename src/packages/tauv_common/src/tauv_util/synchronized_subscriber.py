# Gleb Ryabtsev, 2023
from typing import Callable

import rospy
from dataclasses import dataclass


@dataclass
class MsgWrapper:
    msg: rospy.msg.AnyMsg
    seq: int  # internal sequence number


@dataclass
class SubWrapper:
    name: str  # rostopic name
    sub: rospy.Subscriber
    key_function: Callable  # key for finding messages belonging to the same frame
    q_size: int  # max number of stored messages
    msgs = dict()
    last_seq = -1

    def add_msg(self, msg):
        # remove the oldest message if the queue size limit is reached
        if len(self.msgs) >= self.q_size:
            oldest_seq = None
            oldest_key = None
            for key, msg in self.msgs.items():
                if oldest_seq is None or msg.seq < oldest_seq:
                    oldest_seq = msg.seq
                    oldest_key = key
            del self.msgs[oldest_key]

        assert (len(self.msgs) < self.q_size)
        # add current message
        self.last_seq += 1
        self.msgs[self.key_function(msg)] = MsgWrapper(msg, self.last_seq)


class SynchronizedSubscriber:
    """Class for subscribing to multiple related rostopics simultaneously and receiving
    messages from all topics synchronously. For example use this to synchronize depth and
    RGB video feeds.
    """
    def __init__(self, callback, queue_size=10):
        """Initialize SynchronizedSubscriber

        @param callback: Callback function for the complete frame. Should take a dictionary
        with topic names as keys and corresponding messages as elements.
        @param queue_size: Max number of stored messages for each subscribed topic.
        """
        self.__subscribers = []
        self.__active = False
        self.__user_callback = callback
        self.__q_size = queue_size

    def subscribe(self, name: str, data_class: str, key_function, **kwargs):
        """Subscribe to one topic

        @param name: Topic name
        @param data_class: Message type
        @param key_function: Function returning the key. Should take data_class object as
        an argument and return a hashable value that will be the same for all messages belonging
        to the same frame.
        @param kwargs: Passed to rospy.Subscriber initializer. Must not include callback_args.
        """
        index = len(self.__subscribers)
        sub = rospy.Subscriber(name, data_class, self.__message_callback,
                               callback_args=index, **kwargs)
        self.__subscribers.append(SubWrapper(name, sub, key_function, self.__q_size))

    def start(self):
        self.__active = True

    def stop(self):
        self.__active = False

    def unregister(self, name):
        """Stop receiving messages for a specific ROS topic

        @param name: topic name
        """
        i = 0
        while i < len(self.__subscribers):
            sub_data = self.__subscribers[i]
            if sub_data.name == name:
                sub_data.sub.unregister()
                self.__subscribers.pop(i)
            else:
                i += 1

    def unregister_all(self):
        """Unsubscribe from all topics.
        """
        for sub_data in self.__subscribers:
            sub_data.sub.unregister()

    def __message_callback(self, msg, sub_index):
        if not self.__active:
            return
        sub = self.__subscribers[sub_index]
        key = sub.key_function(msg)
        for other_sub in self.__subscribers:
            if other_sub is sub:
                continue
            if key not in other_sub.msgs.keys():
                sub.last_seq += 1
                sub.add_msg(msg)
                break
        else:
            frame = {sub.name: msg}
            for other_sub in self.__subscribers:
                if other_sub is sub:
                    continue
                frame[other_sub.name] = other_sub.msgs[key]
                del other_sub.msgs[key]

            self.__user_callback(frame)

