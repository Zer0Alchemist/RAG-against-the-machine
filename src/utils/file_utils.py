def read_file(file_path: str) -> str:
    """
    This function opens the file_path given, and returns it
    as str

    args:
        file_path: str = the file path string

    returns:
        returns a string of what the file contains
    """
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError("[ERROR] File is not found!")
    except PermissionError:
        raise PermissionError("[ERROR] You do not have permission "
                              "for this file")
    except IsADirectoryError:
        raise IsADirectoryError("[ERROR]: File is directory!")
