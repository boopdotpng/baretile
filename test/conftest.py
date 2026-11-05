# Shared chip fixture and tensor helpers. test_device.py covers the device and runtime; test_kernels.py covers
# code generation and tensors; test_models.py covers models, serving and adapters. They run on the simulator that
# python3 -m baretile.chip ttsim downloads, or the libttsim.so that TTSIM names. On a machine with a card and no TTSIM,
# they open the first Blackhole through tt-kmd, or as root through sysfs without it (see docs/bring-up.md).
import functools, gc, json, os, pathlib, random, statistics, pytest
import numpy as np
from baretile import pci
from baretile.chip import Chip
from baretile.runtime import DATA, compile_kernel

def bf16(a, packer=False):  # rounds to bf16 and back: to nearest even like our uploads, or half up like the packer
  b = np.asarray(a, np.float32).view(np.uint32)
  u = np.uint32  # constants stay uint32: NumPy 1.x widens a 0-d uint32 plus a Python int to int64
  return ((b + (u(0x8000) if packer else u(0x7FFF) + (b >> u(16) & u(1)))) >> u(16) << u(16)).view(np.float32).astype(np.float64)

def value(t): return np.array(t.tolist()).reshape(t.shape)  # t as a numpy array of its shape

HELD = 'void kmain(volatile unsigned *a) { while (*(volatile unsigned *)L1_DATA != a[2]) asm volatile("fence" ::: "memory"); }'
@functools.cache
def _held(): return compile_kernel(HELD)
def hold(rt):
  """Queue a launch on rt's first worker that waits for the host, so that the launches queued after it wait too, on
  a card as on ttsim. Call the function this returns to let them go. Don't upload tensor.DMA_UPLOAD bytes or more
  in between, since such an upload waits for the queue."""
  rt.wait()  # nothing queued earlier may write the word it waits on
  x, y = rt.cores[0]
  rt.chip.write32(x, y, DATA, 0)
  rt.launch(_held(), (1,), units=1, name="hold")
  return lambda: rt.chip.write32(x, y, DATA, 1)

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):  # with SKIPS=fail, a test that skips fails, for a CI job that must run all it selects
  report = (yield).get_result()
  if report.skipped and os.environ.get("SKIPS") == "fail":
    report.outcome, report.longrepr = "failed", f"it skipped, with SKIPS=fail: {report.longrepr}"

# With PART=i/n, part i of n of the tests, for one of n processes or CI jobs. The parts take about as long, by the
# seconds each test took in test/durations.json: longest first, each goes to the part with the least so far. A test the
# file doesn't list counts as its median. DURATIONS=write rewrites the file from the tests this run runs.
DURATIONS, times = pathlib.Path(__file__).with_name("durations.json"), {}
def pytest_collection_modifyitems(config, items):
  if not (part := os.environ.get("PART")): return
  i, n = map(int, part.split("/"))
  known = json.loads(DURATIONS.read_text())
  seconds = lambda t: known.get(t.nodeid, statistics.median(known.values()))
  load, mine = [0.0] * n, set()
  for t in sorted(items, key=seconds, reverse=True):
    k = load.index(min(load))
    load[k] += seconds(t)
    if k == i - 1: mine.add(t.nodeid)
  config.hook.pytest_deselected(items=[t for t in items if t.nodeid not in mine])
  items[:] = [t for t in items if t.nodeid in mine]  # in the suite's order

def pytest_runtest_logreport(report):  # setup, call and teardown add up to a test's time
  if os.environ.get("DURATIONS") == "write": times[report.nodeid] = times.get(report.nodeid, 0) + report.duration

def pytest_sessionfinish(session):  # in the main process, which gets every worker's reports
  if os.environ.get("DURATIONS") == "write" and not hasattr(session.config, "workerinput"):
    DURATIONS.write_text(json.dumps({k: round(v, 2) for k, v in sorted(times.items())}, indent=0) + "\n")

@pytest.fixture(autouse=True)
def seeded(request):  # the random module's numbers depend on the test alone, so a failure repeats in any order or worker
  random.seed(request.node.nodeid)

@pytest.fixture
def no_gc():  # Python collects only where the test calls gc.collect()
  collecting = gc.isenabled()
  gc.disable()
  yield
  if collecting: gc.enable()

@pytest.fixture(scope="session")
def chip(): return Chip(pci.open_device())  # session-wide, since libttsim can only be initialized once a process
