import glob
import os
from os import path
from datetime import datetime
from pprint import pprint





def corrected_file_name(file_name):
    if not os.path.isfile(file_name):
        return file_name
    file_name_, file_ext = os.path.splitext(file_name)

    j = 1
    new_name = file_name
    while os.path.isfile(new_name):
        new_name = file_name_ + "." + str(j) + file_ext
        j += 1

    return new_name







file_list = [filename for filename in glob.iglob("./**/*.*", recursive=True)]
print(len(file_list))

signatures = [
    (b"\xFF\xD8\xFF", "jpg"),
    (b"\x47\x49\x46\x38", "gif"),
    (b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A", "png"),
    (b"<!DOCTYPE html>", "html"),
]


class ImgExt:
    JPG = "jpg"
    GIF = "gif"
    PNG = "png"
    KNOWN_IMAGE_TYPES = ("jpg", "gif", "png")


def get_file_type(bytes_str):
    for sig, file_type in signatures:
        size = min(len(sig), len(bytes_str))
        if sig[:size] == bytes_str[:size]:
            return file_type


from shutil import copyfile

for file_name in file_list:
    first_10 = open(file_name, "rb").read(10)
    file_type = get_file_type(first_10)
    if file_type in ImgExt.KNOWN_IMAGE_TYPES:
        pass

    else:
        print("Bad file", file_name)
        basename = path.basename(file_name)
        new_filename = os.path.join(
            "/home/art/bad",
            basename,
        )

        new_filename = corrected_file_name(new_filename)
        copyfile(file_name, new_filename)
        print(new_filename)
        os.remove(file_name)
