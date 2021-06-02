with open('./kernel.h', 'r') as f:
    lines = f.readlines()
with open('./kernel.h', 'w') as f:
    #i = 0
    for i, line in enumerate(lines):
        #i = i + 1
        if i ==18:
            if len(line) == 29:
                f.write(line[2:])
            else:
                f.write(line)
        else:
            f.write(line)