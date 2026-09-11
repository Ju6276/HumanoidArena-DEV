"""Launch sim_main without its legacy process-name based global cleanup."""
import sys
from pathlib import Path
SIMULATOR = Path(__file__).resolve().parents[1]
REPOSITORY = SIMULATOR.parent
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(SIMULATOR))
import sim_main
failure = False
try:
    sim_main.main()
except BaseException:
    failure = True
    import traceback
    traceback.print_exc()
finally:
    # Kit can hang during close, including after a startup exception. Bound the
    # child only so the suite can report infrastructure failure and continue.
    import os,threading
    timer=threading.Timer(15,lambda:os._exit(1 if failure else 0));timer.daemon=True;timer.start()
    sim_main.simulation_app.close()
    timer.cancel()
if failure:
    raise SystemExit(1)
