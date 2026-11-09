"""Standard-library CSV fixtures; no source artifacts or numerical models."""
from __future__ import annotations
import csv
import gzip
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import physical_split_finish_io as finish_io

ROOT=Path(__file__).resolve().parents[1]


class FinishIOTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='finish_io_',dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)

    def read(self,path,compressed=False):
        opener=gzip.open if compressed else open
        with opener(path,'rt',encoding='utf-8',newline='') as stream:
            reader=csv.DictReader(stream)
            rows=list(reader)
            return reader.fieldnames,rows

    def test_fdma_first_and_hybrid_first_have_same_full_sorted_union(self):
        rows=[dict(arm='fdma',rate=0.,physical_fdma=0),
              dict(arm='hybrid',rate=3.5,physical_key=.2,physical_shared=.4)]
        expected=sorted({'arm','rate','physical_fdma','physical_key','physical_shared'})
        for index,ordered in enumerate((rows,list(reversed(rows)))):
            path=self.folder/f'monitor_{index}.csv'
            self.assertEqual(finish_io.write_union_csv(path,ordered),expected)
            columns,data=self.read(path)
            self.assertEqual(columns,expected);self.assertEqual(len(data),2)
            by_arm={row['arm']:row for row in data}
            self.assertEqual(by_arm['fdma']['physical_key'],'')
            self.assertEqual(by_arm['fdma']['physical_shared'],'')
            self.assertEqual(by_arm['fdma']['physical_fdma'],'0')
            self.assertEqual(by_arm['fdma']['rate'],'0.0')
            self.assertEqual(by_arm['hybrid']['physical_fdma'],'')
            self.assertEqual(by_arm['hybrid']['physical_key'],'0.2')

    def test_missing_none_true_zero_and_quoted_unicode_round_trip_plain_and_gzip(self):
        rows=[dict(arm='FDMA',zero=0,optional=None,note='逗号, quote " and\nnewline'),
              dict(arm='Hybrid',zero=-0.,shared=.1)]
        outputs=[]
        for compressed in (False,True):
            path=self.folder/('sample.csv.gz' if compressed else 'sample.csv')
            finish_io.write_union_csv(path,iter(rows),compressed)
            outputs.append(self.read(path,compressed))
        self.assertEqual(outputs[0],outputs[1])
        data=outputs[0][1]
        self.assertEqual(data[0]['zero'],'0');self.assertEqual(data[1]['zero'],'-0.0')
        self.assertEqual(data[0]['optional'],'');self.assertEqual(data[1]['optional'],'')
        self.assertEqual(data[0]['shared'],'');self.assertEqual(data[0]['note'],rows[0]['note'])
        duplicate=self.folder/'different_filename.csv.gz'
        finish_io.write_union_csv(duplicate,rows)
        self.assertEqual(duplicate.read_bytes(),(self.folder/'sample.csv.gz').read_bytes())

    def test_prescan_failure_never_creates_temp_or_changes_destination(self):
        target=self.folder/'old.csv';target.write_bytes(b'prior user bytes')
        original={p.name:p.read_bytes() for p in self.folder.iterdir()}
        def broken_rows():
            yield dict(arm='fdma')
            raise ValueError('later row unavailable')
        with self.assertRaisesRegex(ValueError,'later row unavailable'):
            finish_io.write_union_csv(target,broken_rows())
        self.assertEqual({p.name:p.read_bytes() for p in self.folder.iterdir()},original)

    def test_midwrite_failure_preserves_old_plain_and_gzip_and_other_temp_files(self):
        class BrokenValue:
            def __str__(self):raise ValueError('value cannot be rendered')
        marker=self.folder/'.user-owned.tmp';marker.write_bytes(b'keep original user temp')
        for suffix in ('.csv','.csv.gz'):
            target=self.folder/('old'+suffix);target.write_bytes(b'old immutable bytes')
            before={p.name:p.read_bytes() for p in self.folder.iterdir()}
            with self.assertRaisesRegex(ValueError,'cannot be rendered'):
                finish_io.write_union_csv(target,[dict(first=1),dict(later=BrokenValue())])
            self.assertEqual({p.name:p.read_bytes() for p in self.folder.iterdir()},before)

    def test_replace_failure_keeps_old_bytes_and_strict_writer_does_not_ignore_fields(self):
        target=self.folder/'old.csv';target.write_bytes(b'old output')
        actual_writer=csv.DictWriter
        with patch.object(finish_io.csv,'DictWriter',wraps=actual_writer) as writer,patch.object(
                finish_io.os,'replace',side_effect=OSError('atomic replacement unavailable')):
            with self.assertRaisesRegex(OSError,'atomic replacement unavailable'):
                finish_io.write_union_csv(target,[{'fdma':1},{'hybrid':2}])
        self.assertEqual(target.read_bytes(),b'old output')
        self.assertEqual(list(self.folder.iterdir()),[target])
        self.assertEqual(writer.call_args.kwargs['extrasaction'],'raise')
        self.assertEqual(writer.call_args.kwargs['fieldnames'],['fdma','hybrid'])

    def test_empty_input_is_empty_and_invalid_schema_refuses_before_output(self):
        empty=self.folder/'empty.csv'
        self.assertEqual(finish_io.write_union_csv(empty,[]),[]);self.assertEqual(empty.read_bytes(),b'')
        target=self.folder/'old.csv';target.write_bytes(b'prior bytes')
        for rows in ([{}],[{1:'not a string column'}],[['not a mapping']]):
            with self.assertRaises((TypeError,ValueError)):finish_io.write_union_csv(target,rows)
            self.assertEqual(target.read_bytes(),b'prior bytes')
        with self.assertRaises(TypeError):finish_io.write_union_csv(target,[{'value':1}],compressed='yes')
        self.assertEqual(target.read_bytes(),b'prior bytes')


if __name__=='__main__':unittest.main()
