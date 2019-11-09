class TauvRemoteReconfigableInfo:
    def __init__(self, name, affected, current_val, reset_val, minval, maxval):
        self.name = name
        self.affected = affected
        self.current_val = current_val
        self.reset_val = reset_val
        self.minval = minval
        self.maxval = maxval

    def getName(self):
        return self.name

    def getAffected(self):
        return self.affected

    def getCurrentVal(self):
        return self.current_val

    def getResetVal(self):
        return self.reset_val

    def getMinVal(self):
        return self.minval

    def getMaxVal(self):
        return self.maxval

    def __str__(self):
        return ("Reconfigable: {}\n"
                "Current Value: {}\n"
                "Reset Value: {}\n"
                "Value Range: ({}, {})\n"
                "Affected Services: {}\n"
                ).format(
            self.name, self.current_val, self.reset_val, self.minval, self.maxval, self.affected
        )

class TauvRemoteReconfigable:
    def __init__(self, name):
        self.name = name
        # call register service: meerly verifies reconfigable existence right now

    def get_val(self):
        # call the request value service

    def set_val(self):
        # call the set value service

    def reset_val(self):
        # call the reset value service

    def get_info(self):
        # call a service to get the full TauvRemoteReconfigableInfo object:

class TauvReconfigBridge:
    def __init__(self):
        # verify service existence

    # should this be a published topic instead? or both?
    def get_reconfigables(self):
        # return a list of names of all reconfigables
