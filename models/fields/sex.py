from base.fields import CodeBasedField

class Sex(CodeBasedField):
    _codes = {
        "1": "Male",
        "2": "Female"
    }
