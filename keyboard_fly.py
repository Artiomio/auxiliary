from __future__ import print_function
try:
    import msvcrt

    def key_pressed():
        return msvcrt.kbhit()

    def read_key():
        return msvcrt.getch()

except:

    try:
        import sys
        import select
        import tty
        import termios
        import atexit

        def key_pressed():
            return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

        def read_key():
            return sys.stdin.read(1)

        def restore_settings():
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


        atexit.register(restore_settings)
        old_settings = termios.tcgetattr(sys.stdin)

        tty.setcbreak(sys.stdin.fileno())
    except:
        print("Can't deal with your keyboard!")
        

if __name__ == "__main__":
    print("Press any key")
    c = read_key()
    print("You pressed", c)