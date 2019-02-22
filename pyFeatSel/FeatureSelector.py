import abc


class FeatureSelector(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def run(self):
        pass