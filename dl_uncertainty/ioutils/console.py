import select
import sys
import platform


def read_line(impatient: bool = False, discard_non_last: bool = False):
    """
    Returns a line from standard input.
    :param impatient: If set to True and no input is available, returns None.
    :param discard_non_last: If set to True and multiple lines are available, 
    consumes all lines and returns the last line.
    """

    def input_available():
        if platform.system() == 'Windows':
            import msvcrt
            return msvcrt.kbhit()
        else:
            return select.select([sys.stdin], [], [], 0)[0]

    if impatient and not input_available(): return None
    cmd = input()
    if discard_non_last:
        while input_available():
            cmd = input()
    return cmd


def main():
    while True:
        print(read_line(impatient=False, discard_non_last=True))


if __name__ == "__main__":
    main()