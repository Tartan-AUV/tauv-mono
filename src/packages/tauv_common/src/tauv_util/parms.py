# Gleb Ryabtsev, 2023

import yaml


class Parms(dict):
    """A dictionary-like with simple access syntax, to be used for parameters.

    For example:
    p = Parms.fromfile("my_parameters.yaml")
    print(p.category.subcategory.param_name)
    `
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes a Parms object. Passes *args and **kwargs to dict() initializer.
        """
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        for key in self:
            if type(self[key]) == dict:
                self[key] = Parms(self[key])

    @staticmethod
    def fromfile(path):
        """Constructs a Parms object from a YAML file.
        @param path: relative or absolute path to the files
        """
        with open(path, 'r') as f:
            d = yaml.safe_load(f)
        return Parms(d)
