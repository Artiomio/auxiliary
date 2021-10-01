import os

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
