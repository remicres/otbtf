# -*- coding: utf-8 -*-
"""
Copyright (c) 2020-2022 INRAE

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
"""Various system operations"""
import logging
import pathlib
import os

# ---------------------------------------------------- Helpers ---------------------------------------------------------

def pathify(pth):
    """ Adds posix separator if needed """
    if not pth.endswith("/"):
        pth += "/"
    return pth


def mkdir(pth):
    """ Create a directory """
    path = pathlib.Path(pth)
    path.mkdir(parents=True, exist_ok=True)


def dirname(filename):
    """ Returns the parent directory of the file """
    return str(pathlib.Path(filename).parent)


def basic_logging_init():
    """ basic logging initialization """
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')


def logging_info(msg, verbose=True):
    """
    Prints log info only if required by `verbose`
    :param msg: message to log
    :param verbose: boolean. Whether to log msg or not. Default True
    :return:
    """
    if verbose:
        logging.info(msg)

def is_dir(filename):
    """ return True if filename is the path to a directory """
    return os.path.isdir(filename)
