import json
from pathlib import Path
import tempfile
import unittest

from paper_repro.stream_cache import cached_source


class SourceCacheTests(unittest.TestCase):
    def source(self, n_classes=3):
        return [(c, f'encoded-{c}-{i}'.encode(), c*50+i)
                for c in range(n_classes) for i in range(50)]

    def test_reuses_verified_cache_across_runs(self):
        records = self.source()
        calls = []
        def factory():
            calls.append(1)
            return iter(records)
        with tempfile.TemporaryDirectory() as tmp:
            key = dict(revision='a'*40, corruption='gaussian_noise', severity=5)
            self.assertEqual(list(cached_source(factory,tmp,key,3)),records)
            self.assertEqual(list(cached_source(factory,tmp,key,3)),records)
            self.assertEqual(calls,[1])
            other_order = dict(severity=5, corruption='gaussian_noise', revision='a'*40)
            self.assertEqual(list(cached_source(factory,tmp,other_order,3)),records)
            self.assertEqual(calls,[1])

    def test_corruption_is_rebuilt_before_yielding_any_records(self):
        records = self.source()
        calls = []
        def factory():
            calls.append(1)
            return iter(records)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(list(cached_source(factory,tmp,'source-v1',3)),records)
            data = next(Path(tmp).glob('*/records.bin'))
            with data.open('r+b') as stream:
                stream.seek(-1,2)
                stream.write(b'!')
            self.assertEqual(list(cached_source(factory,tmp,'source-v1',3)),records)
            self.assertEqual(calls,[1,1])

    def test_changed_revision_does_not_reuse_source_bytes(self):
        calls = []
        def factory():
            calls.append(1)
            return iter(self.source())
        with tempfile.TemporaryDirectory() as tmp:
            list(cached_source(factory,tmp,{'revision':'a'*40},3))
            list(cached_source(factory,tmp,{'revision':'b'*40},3))
            self.assertEqual(len(calls),2)

    def test_incomplete_source_and_failed_download_never_complete_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'exactly 50'):
                list(cached_source(lambda: iter(self.source()[:-1]),tmp,'key',3))
            self.assertEqual(list(Path(tmp).glob('*/manifest.json')),[])
            def interrupted():
                yield self.source()[0]
                raise OSError('download interrupted')
            with self.assertRaises(OSError):
                list(cached_source(interrupted,tmp,'key',3))
            self.assertEqual(list(Path(tmp).glob('*/manifest.json')),[])
            self.assertEqual(list(cached_source(lambda: iter(self.source()),tmp,'key',3)),self.source())

    def test_duplicate_ids_invalid_labels_and_unsorted_blocks_are_rejected(self):
        bad_cases = [self.source()+[self.source()[0]],
                     [(1,b'image',0)]+self.source()[1:],
                     self.source()[50:]+self.source()[:50]]
        for i,records in enumerate(bad_cases):
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                list(cached_source(lambda: iter(records),tmp,f'bad-{i}',3))

    def test_manifest_mismatch_and_truncation_rebuild(self):
        calls = []
        def factory():
            calls.append(1)
            return iter(self.source())
        with tempfile.TemporaryDirectory() as tmp:
            list(cached_source(factory,tmp,'source',3))
            manifest = next(Path(tmp).glob('*/manifest.json'))
            saved = json.loads(manifest.read_text())
            saved['records'] = 1
            manifest.write_text(json.dumps(saved))
            list(cached_source(factory,tmp,'source',3))
            next(Path(tmp).glob('*/records.bin')).write_bytes(b'broken')
            list(cached_source(factory,tmp,'source',3))
            self.assertEqual(len(calls),3)


if __name__ == '__main__':
    unittest.main()
