import rospy

class Reconfigable:
    def __init__(self, name, val, reloaders):
        self.name = name
        self.savedVal = val
        self.reloaders = reloaders


    def modify(self, val):
        for reloader in self.reloaders:
            reloader()

    def reset(self):
        self.modify(self.savedVal)

    def getVal(self):
        return

    def __str__(self):
        return "[{}: {}]".format(self.name, self.getVal())

class ReconfigableServer:
    def __init__(self):
        # load reconfigable configs

        # declare all the services

    #loads of these:
    def callback(self):