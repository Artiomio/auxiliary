import time
from gotoXY import gotoXY, console_print
from keyboard_fly import key_pressed, read_key

gotoXY(1, 1)
console_print("Press 'w', 'a', 's', 'd' to move")
key = ''

x = 10
y = 10

vx = 0
vy = 1

velocities = {b'a': (-1, 0), b'd': (1, 0), b'w': (0, -1), b's':(0, 1)}

while key != b'q':
    if key_pressed():

        key_data = read_key()
        try:
            key = bytes(key_data)
        except:
            key = bytes(key_data, encoding="ascii")

        if key in velocities:
            vx, vy = velocities[key]

    x = x + vx
    y = y + vy

    gotoXY(x, y)
    print("*")
    time.sleep(0.3)

    gotoXY(x, y)
    print(" ")



gotoXY(1, 1)
console_print("Bye!")



    
