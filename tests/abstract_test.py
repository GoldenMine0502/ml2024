from abc import ABC, abstractmethod


class AbstractTest(ABC):
    @abstractmethod
    def test(self, data):
        pass

    @abstractmethod
    def get_test_name(self):
        pass
