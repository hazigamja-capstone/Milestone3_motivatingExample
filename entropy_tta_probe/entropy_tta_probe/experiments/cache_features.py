"""Pre-compute center-crop CLIP image features for every split (used by E1, E4, E5, E6)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from etta.common import base_args, get_features, load_bundle  # noqa: E402
from etta.data import expand_names  # noqa: E402

if __name__ == "__main__":
    args = base_args(__doc__).parse_args()
    bundle = load_bundle(args)
    for name in expand_names(args.datasets):
        d = get_features(args, bundle, name)
        print(f"{name}: {len(d['labels'])} samples, {len(d['classnames'])} classes")
