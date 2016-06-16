for line in open('test','r'):
    line = line.split(',')[0] + ',0' + line.split(line.split(',')[0])[1]
