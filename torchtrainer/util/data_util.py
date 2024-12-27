from pathlib import Path


def search_files(files: list[str], paths: list[Path]) -> list[int]:
    """Search for each file in list `paths` and return the indices of the files found
    in the paths list. Paths can then be fitlered as:

    >>> paths = [paths[idx] for idx in search_files(files, paths)] 

    Parameters
    ----------
    files
        List of file names to search   
    paths   
        List of paths.

    Returns
    -------
    List of indices of the files found in the paths list
    """

    indices = []
    for file in files:
        for idx, path in enumerate(paths):
            if file in str(path):
                indices.append(idx)

    return indices
        