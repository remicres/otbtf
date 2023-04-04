"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

for path in sorted(Path("otbtf").rglob("*.py")):  #
    module_path = path.relative_to(".").with_suffix("")  #
    doc_path = path.relative_to(".").with_suffix(".md")  #
    full_doc_path = Path("reference", doc_path)  #

    parts = list(module_path.parts)

    if parts[-1] == "__init__":  #
        parts = parts[:-1]
    elif parts[-1] == "__main__":
        continue

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:  #
        identifier = ".".join(parts)  #
        print("::: " + identifier)
        print("::: " + identifier, file=fd)  #

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# Workaround to install and execute git-lfs on Read the Docs
import os
if not os.path.exists('./git-lfs'):
    os.system('wget https://github.com/git-lfs/git-lfs/releases/download/v2.7.1/git-lfs-linux-amd64-v2.7.1.tar.gz')
    os.system('tar xvfz git-lfs-linux-amd64-v2.7.1.tar.gz')
    os.system('./git-lfs install')  # make lfs available in current repository
    os.system('./git-lfs fetch')  # download content from remote
    os.system('./git-lfs checkout')  # make local files to have the real content on them
