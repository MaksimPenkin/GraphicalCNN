# """
# @author   Maksim Penkin @MaksimPenkin
# @author   Oleg Khokhlov @okhokhlov
# """

import os
import shutil


def delete_file_folder(f):
    """Method for deleting file or folder.

    :param f: path to folder to be deleted
    :raises Exception: exception is raised if failed to delete
    """
    try:
        if os.path.isfile(f) or os.path.islink(f):
            os.unlink(f)
        elif os.path.isdir(f):
            shutil.rmtree(f)
    except Exception as e:
        print('utility.utils.py: def delete_file_folder(...): error: Failed to delete %s. Reason: %s' % (f, e))


def delete_contents_folder(folder):
    """Method for deleting all contents of folder.

    :param f: path to folder to be deleted with all contents
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        delete_file_folder(file_path)


def create_folder(folder, force=False, raise_except_if_exists=True):
    """Method for creating folder.

    :param folder: path to folder to be created
    :param force: if True, overwrite directory to be created in case of existence
    :param raise_except_if_exists: if True, raise exception in case of directory existence
    :raises Exception: exception is raised if trying to create folder which exists
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        if force:
            delete_contents_folder(folder)
        else:
            if raise_except_if_exists:
                raise Exception('utility.utils.py: def create_folder(...): error: directory {} exists.'
                                'In order to overwrite it set force=True'.format(folder))
