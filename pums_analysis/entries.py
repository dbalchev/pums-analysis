from base.models import Model
from base.fields import IntegerField

from pums_analysis.fields import JobField, DegreeField, StateField


class Entry(Model):
    age = IntegerField("AGEP")
    occupation = JobField("OCCP02")
    degree = DegreeField("FOD1P")
    state = StateField("ST")
    wage = IntegerField("WAGP")