"""

"""


# Built-in
import os

# Libs

# Own modules
from mrs_utils import misc_utils


def split_by_id(file_list, id_list, filter_keys):
    assert len(file_list) == len(id_list)
    list_1 = []
    list_2 = []
    for fl, id_ in zip(file_list, id_list):
        if id_ in filter_keys:
            list_1.append(fl)
        else:
            list_2.append(fl)
    return list_1, list_2


def get_inria_city_id(file_list):
    city_id_list = []
    for fl in file_list:
        if isinstance(fl, list):
            city_name = str(os.path.basename(fl[0]).split('_')[0])
        else:
            city_name = str(os.path.basename(fl).split('_')[0])
        city_id = int(''.join(s for s in city_name if s.isdigit()))
        city_id_list.append(city_id)
    return city_id_list


if __name__ == '__main__':
    file_list_name  = r'/hdd/mrs/inria/file_list.txt'
    file_list = misc_utils.load_file(file_list_name)
    city_id_list = get_inria_city_id(file_list)
    list_1, list_2 = split_by_id(file_list, city_id_list, list(range(6)))
    for l in list_1:
        print(l)
