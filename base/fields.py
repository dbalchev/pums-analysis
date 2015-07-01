from .exceptions import InvalidCode

class ModelField:
    def __init__(self, field_name):
        self._field_name = field_name

    def __call__(self, fields_dict):
        return self._create_from_str(fields_dict[self._field_name])

    def _create_from_str(self, string_repr):
        raise NotImplementedError()


class IntegerField(ModelField):
    def _create_from_str(self, string_repr):
        if string_repr == "":
            return -1
        return int(string_repr)



class CodeBasedField(ModelField):
    def _type_name(self):
        return type(self).__name__
    def _create_from_str(self, string_repr):
        if string_repr == "":
            return ""
        try:
            return self._codes[string_repr]
        except KeyError:
            raise InvalidCode("code {} is invalid for field type {}"\
                .format(string_repr, self._type_name()))
