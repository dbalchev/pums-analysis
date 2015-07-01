class JobField(CodeBasedField):
    _codes = {"0010":"MGR-CHIEF EXECUTIVES AND LEGISLATORS",
    "0020": "MGR-GENERAL AND OPERATIONS MANAGERS"}


class Entry(Model):
    age = Integer("AGEP")
    occupation = JobField("OCCP02")
