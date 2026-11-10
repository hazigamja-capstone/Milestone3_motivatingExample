import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from paper_repro import evaluate_detector as detector
from paper_repro import protocol
from paper_repro.run_plan import commands


class StreamTests(unittest.TestCase):
    def test_class_partition_is_disjoint_complete_and_seed_independent(self):
        cal = set(protocol.role_classes('calibration'))
        confirm = set(protocol.role_classes('confirmation'))
        self.assertEqual(len(cal),400)
        self.assertEqual(len(confirm),600)
        self.assertFalse(cal & confirm)
        self.assertEqual(cal | confirm,set(range(1000)))

    def test_disk_reordering_preserves_records_and_changes_blocks(self):
        source = [(c,bytes([i]),c*50+i) for c in range(10) for i in range(50)]
        a = list(protocol.reorder_stream(iter(source),'confirmation',201,0,n_classes=10))
        b = list(protocol.reorder_stream(iter(source),'confirmation',202,0,n_classes=10))
        self.assertEqual(set(a),set(b))
        self.assertEqual(len(a),300)
        self.assertNotEqual(a,b)
        self.assertEqual(a,list(protocol.reorder_stream(iter(source),'confirmation',201,0,n_classes=10)))
        self.assertTrue(all(len({v[0] for v in a[i:i+50]})==1 for i in range(0,len(a),50)))

    def test_paper_order_and_domain_randomization(self):
        source = [(c,b'x',c*50+i) for c in range(10) for i in range(50)]
        paper = list(protocol.reorder_stream(iter(source),'paper',1,0,n_classes=10))
        self.assertEqual([x[0] for x in paper],sorted(x[0] for x in paper))
        domains = list(range(15))
        a = protocol.domain_order(domains,'confirmation',201)
        b = protocol.domain_order(domains,'confirmation',202)
        self.assertEqual(set(a),set(domains)); self.assertNotEqual(a,b)
        self.assertEqual(protocol.domain_order(domains,'paper',1),domains)


class DetectorTests(unittest.TestCase):
    def rows(self,n=110):
        return [dict(disagree=i/n,ent_t=i/n,ent_z=1-i/n,progress=i/n,
                     kl=i/n,topk=1-i/n,delta=(-90 if i%2 else 20)) for i in range(n)]

    def test_calibration_does_not_use_outcomes_and_does_not_fallback(self):
        rows = self.rows()
        before = detector.unlabeled_thresholds([rows])
        changed = copy.deepcopy(rows)
        for row in changed:
            row['delta'] = 100
        self.assertEqual(before,detector.unlabeled_thresholds([changed]))
        with self.assertRaisesRegex(ValueError,'100'):
            detector.unlabeled_thresholds([rows[:99]])

    def test_calibration_prefix_cannot_see_later_values(self):
        rows = self.rows()
        before = detector.unlabeled_thresholds([rows])
        for row in rows[100:]:
            row['disagree'] = 100
        self.assertEqual(before,detector.unlabeled_thresholds([rows]))

    def test_end_of_window_timing_and_bounded_onsets(self):
        event = dict(start=5,end=8)
        on = [True]*9
        self.assertIsNone(detector.event_detection(on,event))
        on = [False]*4+[True]*5
        detected = detector.event_detection(on,event)
        self.assertEqual(detected['available_sample'],5*512)
        self.assertEqual(detected['delay_from_start_samples'],0)
        on = [False]*5+[True]*4
        self.assertEqual(detector.event_detection(on,event)['delay_from_start_samples'],512)
        self.assertIsNone(detector.event_detection([False]*9,event))

    def test_miss_is_retained_and_false_alarms_counted(self):
        rows = [dict(delta=0)]*3+[dict(delta=-10)]*3
        result = detector.signal_metrics(rows,[dict(start=3,end=5)],[True]*6)
        self.assertEqual(result['missed'],1)
        self.assertTrue(result['median_delay_is_infinite'])
        self.assertEqual(result['false_alarm_segment_rate'],1.)
        self.assertEqual(result['false_alarm_clusters'],1)

    def test_switch_only_after_available_alarm_without_labels(self):
        n = 1536
        data = dict(y=np.zeros(n,dtype=int), pred_t=np.ones(n,dtype=int),pred_z=np.zeros(n,dtype=int))
        result = detector.utility(data,[True,False,False],64)
        self.assertEqual(result['switch_sample'],512)
        self.assertAlmostEqual(result['switch_policy']['accuracy'],100*1024/1536)
        self.assertEqual(result['zero_shot']['accuracy'],100)
        # Different ground truth changes evaluation, never policy time.
        data['y'][:] = 1
        self.assertEqual(detector.utility(data,[True,False,False],64)['switch_sample'],512)
        self.assertFalse(detector.utility(data,[False]*3,64)['switched'])

    def test_plan_commands_cover_all_seeds_and_preserve_role(self):
        out = Path('/tmp/result')
        cal = commands('B1','calibration',out,'a'*40)
        test = commands('B1','confirmation',out,'a'*40)
        self.assertEqual([int(c[-1]) for c in cal],[101,102])
        self.assertEqual([int(c[-1]) for c in test],list(range(201,211)))
        self.assertTrue(all('--data_revision' in c and 'confirmation' in c for c in test))

    def test_plan_blocks_unfrozen_confirmation_and_revision_changes(self):
        from paper_repro import run_plan
        with tempfile.TemporaryDirectory() as tmp:
            base = [sys.executable,run_plan.__file__,'--combo','B1','--phase','confirmation',
                    '--out',tmp,'--data_revision','a'*40]
            blocked = subprocess.run(base,capture_output=True,text=True)
            self.assertNotEqual(blocked.returncode,0)
            self.assertIn('Freeze calibration',blocked.stderr)
            self.assertFalse(list(Path(tmp).glob('eta_*')))
            changed = subprocess.run(base[:-1]+['b'*40],capture_output=True,text=True)
            self.assertNotEqual(changed.returncode,0)
            self.assertIn('share the study dataset SHA',changed.stderr)



class EndToEndTests(unittest.TestCase):
    def fixture(self,root,role,seed):
        root.mkdir()
        classes = protocol.role_classes(role)
        domains = protocol.domain_order(['gaussian_noise','shot_noise','impulse_noise'],role,seed)
        for i,dom in enumerate(domains):
            order = classes.copy(); random.Random(seed*1000+i).shuffle(order)
            ids = np.array([c*50+j for c in order for j in range(50)],dtype=np.int64)
            y = ids//50; n = len(y)
            pred = y.copy()
            # Deliberate harm after an otherwise safe prefix. All arrays are synthetic.
            if role == 'confirmation':
                pred[8*512:12*512] = (pred[8*512:12*512]+1)%1000
            np.savez(root/f'dom{i:02d}_{dom}.npz',y=y,pred_t=pred,pred_z=y,
                     ent_t=np.ones(n),ent_z=np.ones(n),maxp_t=np.full(n,.5),
                     kl_tz=(pred!=y).astype(float),rank_z_at_t=(pred!=y).astype(int)*10,sample_id=ids)
        meta = dict(seed=seed,method='eta',arch='ViT-B-16',setting='correlated',domains=domains,
                    severity=5,bs=64,lr=.00025,paper_commit='abc',torch='fixture',
                    cfg=f'RNG_SEED: {seed}\nTEST: 1',role=role,protocol_version=3,classes=classes,
                    data_revision='a'*40,grad_ckpt=False,
                    identity={'source_sha256':{'fixture':'same'},'runner_version':3})
        (root/'meta.json').write_text(json.dumps(meta))
        return str(root)

    def test_frozen_calibration_then_ten_confirmation_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cal = [self.fixture(root/f'cal{s}','calibration',s) for s in (101,102)]
            tests = [self.fixture(root/f'test{s}','confirmation',s) for s in range(201,211)]
            artifact_path = root/'frozen.json'; result_path = root/'evaluation.json'
            subprocess.run([sys.executable,detector.__file__,'calibrate',*cal,'--output',str(artifact_path)],check=True,capture_output=True)
            artifact_bytes = artifact_path.read_bytes()
            duplicate = subprocess.run([sys.executable,detector.__file__,'calibrate',*cal,'--output',str(artifact_path)],capture_output=True)
            self.assertNotEqual(duplicate.returncode,0)
            self.assertEqual(artifact_path.read_bytes(),artifact_bytes)
            subprocess.run([sys.executable,detector.__file__,'evaluate',*tests,'--calibration',str(artifact_path),'--output',str(result_path)],check=True,capture_output=True)
            result = json.loads(result_path.read_text())
            self.assertEqual(len(result['runs']),10)
            self.assertTrue(all(r['signals']['disagree']['events']>0 for r in result['runs']))
            self.assertTrue(all(r['signals']['disagree']['detected']>0 for r in result['runs']))
            self.assertTrue(all(r['utility']['switch_policy']['accuracy']>=r['utility']['always_tta']['accuracy'] for r in result['runs']))
            self.assertEqual(artifact_path.read_bytes(),artifact_bytes)
            artifact = json.loads(artifact_bytes); artifact['alpha'] = .1
            with self.assertRaises(ValueError):
                detector.check_artifact(artifact)
            with self.assertRaisesRegex(ValueError,'exactly'):
                detector.checked(tests[:3],'confirmation',protocol.CONFIRMATION_SEEDS)


if __name__ == '__main__':
    unittest.main()
