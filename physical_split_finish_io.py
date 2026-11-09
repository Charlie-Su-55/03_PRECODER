"""Lossless, atomic CSV output for the independent physical-split finish tools.

Only standard-library modules are imported.  This module owns no experiment
selection, model loading, source verification or numerical transformations.
"""
from __future__ import annotations

from collections.abc import Mapping
import csv
import gzip
import io
import os
from pathlib import Path
import tempfile


def _scan_rows(rows):
    """Snapshot every mapping before opening output; later rows may add fields."""
    materialized=[]
    names=set()
    for index,row in enumerate(rows):
        if not isinstance(row,Mapping):
            raise TypeError(f'CSV row {index} must be a mapping')
        copied=dict(row)
        if any(not isinstance(name,str) for name in copied):
            raise TypeError(f'CSV row {index} field names must be strings')
        names.update(copied)
        materialized.append(copied)
    if materialized and not names:
        raise ValueError('Nonempty CSV input must contain at least one field')
    return sorted(names),materialized


def _write_rows(stream,fieldnames,rows):
    if not fieldnames:
        return
    writer=csv.DictWriter(stream,fieldnames=fieldnames,restval='',extrasaction='raise')
    writer.writeheader()
    writer.writerows(rows)


def write_union_csv(path,rows,compressed=None):
    """Write all fields without imputing missing values; return sorted columns.

    Input row order is preserved.  Absent values and None use CSV's empty field;
    real zero values remain zero.  All rows are scanned before any temporary
    output is created.  If ``compressed`` is None, a .gz suffix selects gzip.
    The destination parent must already exist.  A completed, flushed temporary
    file in that same directory replaces the destination atomically.  On error
    the destination is left intact and only this invocation's new temp is
    removed.  No existing file is removed or chmod'ed to make a write succeed.
    """
    target=Path(path)
    if compressed is None:
        compressed=target.suffix.lower()=='.gz'
    if type(compressed) is not bool:
        raise TypeError('compressed must be bool or None')
    fieldnames,materialized=_scan_rows(rows)
    if not target.parent.is_dir():
        raise FileNotFoundError('CSV output parent must already exist: '+str(target.parent))
    if target.is_symlink():
        raise ValueError('CSV destination symlink is not an owned output file: '+str(target))
    descriptor,temporary_name=tempfile.mkstemp(prefix='.'+target.name+'.',suffix='.tmp',dir=target.parent)
    temporary=Path(temporary_name)
    try:
        if compressed:
            raw=os.fdopen(descriptor,'wb');descriptor=None
            with raw:
                # Do not embed the unique temp filename or current timestamp.
                with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as binary:
                    with io.TextIOWrapper(binary,encoding='utf-8',newline='') as stream:
                        _write_rows(stream,fieldnames,materialized)
                raw.flush()
                os.fsync(raw.fileno())
        else:
            stream=os.fdopen(descriptor,'w',encoding='utf-8',newline='');descriptor=None
            with stream:
                _write_rows(stream,fieldnames,materialized)
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(temporary,target)
    except BaseException:
        # os.fdopen may itself fail before adopting the descriptor.
        if descriptor is not None:
            try:os.close(descriptor)
            except OSError:pass
        try:temporary.unlink()
        except FileNotFoundError:pass
        raise
    return fieldnames
