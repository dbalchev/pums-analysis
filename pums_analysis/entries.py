from base.models import Model
from base.fields import IntegerField

from pums_analysis.fields import JobField


class Entry(Model):
    age = IntegerField("AGEP")
    occupation = JobField("OCCP02")
