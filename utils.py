import datetime

class Utils:
    @staticmethod
    def get_kwarg(kwargs, name, default=None):
        if kwargs is None:
            kwargs = {}
        if name not in kwargs and default is None:
            raise ValueError(f"'{name}' expected and not given")
        return kwargs.get(name, default)

    @staticmethod
    def print_time(message):
        curr_time = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"{curr_time} {message}")

utils = Utils()