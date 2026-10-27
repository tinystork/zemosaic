"""Top-level spawn targets for the M1 stop-channel test.

Kept outside the test module so ``multiprocessing`` spawn can import the target
by module name in the child process.
"""

from __future__ import annotations


def child_request_ack(channel, timeout=15.0):
    seen = channel.wait_request(timeout)
    if not seen:
        raise RuntimeError("child did not observe the stop request")
    channel.acknowledge("STOPPED_IN_MEMORY", "child saw request")


def child_no_ack(channel, timeout=0.4):
    # Observe (or time out) but deliberately never acknowledge.
    channel.wait_request(timeout)
