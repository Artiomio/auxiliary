import shutil

columns, lines = shutil.get_terminal_size((80, 20))
print(f"{columns}x{lines}")

while True:
    current_col, current_lines = shutil.get_terminal_size((80, 20))
    
    if (current_col, current_lines) != (columns, lines):
        columns, lines = shutil.get_terminal_size((80, 20))    
        print(f"Changed to {columns}x{lines}")


    