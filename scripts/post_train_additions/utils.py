import os
import socket
import time
from pathlib import Path


def write_in_progress_file(directory: Path, job_identifier: str):
    """
    Creates a 'in_progress.txt' file "
    """
    with open(directory / "in_progress.txt", "w") as f:
        f.write(job_identifier)
    return


def progress_file_exists(directory: Path):
    """
    Indicator if the current file exists.
    """
    return (directory / "in_progress.txt").exists()


def progress_file_id_identical_with_current(directory: Path, job_identifier: str) -> bool:
    """
    Opens file and reads if the id is identical to what the current one is working with.
    """
    read_job_id: str
    with open(directory / "in_progress.txt", "r") as f:
        read_job_id = f.read()
    if read_job_id == job_identifier:
        return True
    else:
        return False


def clean_up_after_processing(directory: Path):
    """
    Deletes the in_progress.txt file from directory.
    :param directory: Path of directory that is being processed.
    """
    os.remove(directory / "in_progress.txt")
    return


def should_process_a_file(dir_path: Path) -> bool:
    """
    Indicator if the current directory should be processed by the current process.

    :param dir_path: Path pointing to the directory with some content inside.
    """
    lsf_jobid = os.getenv("LSB_JOBID", "None")
    hostname = socket.gethostname()
    identifier = hostname + lsf_jobid

    if progress_file_exists(dir_path):
        return False
    write_in_progress_file(dir_path, identifier)
    time.sleep(15)
    if progress_file_id_identical_with_current(dir_path, identifier):
        return True
    else:
        return False
