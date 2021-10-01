from __future__ import print_function
import sys
from ctypes import *

STD_OUTPUT_HANDLE = -11
class COORD(Structure):
    pass



if "win" not in sys.platform or "cygwin" in sys.platform:
    print("POSIX detected")
    def gotoXY(x, y):
        print("\033[" + str(int(y))+";" + str(int(x))+"H", end="")

    console_print = print
else:

    COORD._fields_ = [("X", c_short), ("Y", c_short)]
 
    def gotoXY(c, r):
        r = int(r)
        c = int(c)
        h = windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        windll.kernel32.SetConsoleCursorPosition(h, COORD(int(c), int(r)))

    def console_print_(s):
        h = windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        c = s.encode("windows-1252")
        windll.kernel32.WriteConsoleA(h, c_char_p(c), len(c), None, None)        

    def console_print(*args):
        s = " ".join([str(x) for x in args])
        console_print_(s)


if __name__ == "__main__":
    import math
    import time


    for alpha in range(0, 2000):
        gotoXY(round(20 + 20.0 *  math.cos(alpha*0.01)), round(16 + 10.0 * math.sin(alpha*0.01))     )
        console_print("*")
        time.sleep(0.01)