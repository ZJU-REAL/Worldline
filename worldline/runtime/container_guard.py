"""Independent Docker cleanup after the runner dies, including SIGKILL."""

import os
import select
import sys
import time

from worldline.runtime.containment import remove_container


def process_identity(pid):
    try:
        with open(f"/proc/{pid}/stat") as stream:
            return stream.read().rsplit(")", 1)[1].split()[19]
    except OSError:
        return None


def main():
    parent, identity, container = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    while process_identity(parent) == identity:
        if select.select([sys.stdin], [], [], 0.2)[0]:
            if not os.read(sys.stdin.fileno(), 1):
                break
        time.sleep(0.05)
    remove_container(container)


if __name__ == "__main__":
    main()
