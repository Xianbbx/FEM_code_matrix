"""
main.py
Master orchestrator — runs all simulations in sequence.
Usage:
    python main.py            # run all
    python main.py beam       # run only beam_bending
    python main.py modal      # run only modal_analysis
    python main.py plate      # run only plate_2d
    python main.py plastic    # run only plastic_yielding
    python main.py bimetal    # run only bimetal
    python main.py sinusoidal # run only sinusoidal_actuation
    python main.py bc         # run only boundary_effects
    python main.py opt        # run only optimization
"""
import sys
import os
import time

# Ensure project root is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config

# ---- Create output directories -----
os.makedirs(config.PLOTS_DIR,   exist_ok=True)
os.makedirs(config.ANIM_DIR,    exist_ok=True)


SIMULATIONS = {
    'beam':       ('simulations.beam_bending',         'run'),
    'bimetal':    ('simulations.bimetal',               'run'),
    'sinusoidal': ('simulations.sinusoidal_actuation',  'run'),
    'bc':         ('simulations.boundary_effects',      'run'),
    'modal':      ('simulations.modal_analysis',        'run'),
    'plastic':    ('simulations.plastic_yielding',      'run'),
    'plate':      ('simulations.plate_2d',              'run'),
    'opt':        ('simulations.optimization',          'run'),
}


def run_all(keys=None):
    if keys is None:
        keys = list(SIMULATIONS.keys())

    total_start = time.time()
    results = {}

    for key in keys:
        if key not in SIMULATIONS:
            print(f'[WARNING] Unknown simulation key: {key}  (skip)')
            continue
        module_path, fn_name = SIMULATIONS[key]
        print(f'\n{"="*60}')
        print(f'  Running: {key}  ({module_path})')
        print(f'{"="*60}')
        t0 = time.time()
        try:
            import importlib
            mod = importlib.import_module(module_path)
            fn  = getattr(mod, fn_name)
            fn()
            elapsed = time.time() - t0
            results[key] = ('OK', elapsed)
            print(f'  Completed in {elapsed:.1f}s')
        except Exception as exc:
            elapsed = time.time() - t0
            results[key] = ('FAILED', str(exc))
            print(f'  [FAILED] {exc}')

    print(f'\n{"="*60}')
    print('  SUMMARY')
    print(f'{"="*60}')
    for k, v in results.items():
        status, info = v
        if status == 'OK':
            print(f'  {k:<15} OK  ({info:.1f}s)')
        else:
            print(f'  {k:<15} FAILED — {info}')
    print(f'\n  Total time: {time.time()-total_start:.1f}s')
    print(f'  Plots saved to: {os.path.abspath(config.PLOTS_DIR)}')
    print(f'  Animations saved to: {os.path.abspath(config.ANIM_DIR)}')


if __name__ == '__main__':
    args = sys.argv[1:]
    if args:
        run_all(args)
    else:
        run_all()
