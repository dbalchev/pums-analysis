from csv import DictReader
from zipfile import ZipFile
from io import TextIOWrapper
FILES_IN_ARCHIVE = ["ss13pus{}.csv".format(x) for x in "abcd"]

def read_records_from_file(filename, record_type):
    with ZipFile(filename) as zip_file:
        for subfilename in FILES_IN_ARCHIVE:
            with zip_file.open(subfilename) as bin_subfile:
                subfile = TextIOWrapper(bin_subfile)
                csv_reader = DictReader(subfile)
                for record_dict in csv_reader:
                    yield record_type.from_dict(record_dict)
