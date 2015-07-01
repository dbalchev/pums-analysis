from base.fields import CodeBasedField


class SexField(CodeBasedField):
    _codes = {
        "1": "Male",
        "2": "Female"
    }
