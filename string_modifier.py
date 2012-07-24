import sys
file_in = sys.argv[1]
# file_out = sys.argv[2]
f_in = open(file_in, 'r')
# f_out = open(file_out, 'w')

for line in f_in.readlines():
    # print line
    while True:
        pow_location = line.find('^')
        if pow_location == -1:
            break

        end_pos = pow_location - 1
        start_pos = end_pos
        if line[end_pos] == ')':
            match_found = False
            brackets = 1
            while match_found == False:
                start_pos -= 1
                if line[start_pos] == '(':
                    brackets -= 1
                    if brackets == 0:
                        match_found = True
                elif line[start_pos] == ')':
                    brackets += 1

        while (line[start_pos - 1] != '/' and
               line[start_pos - 1] != '*' and
               line[start_pos - 1] != ' ' and
               line[start_pos - 1] != '('):
            start_pos = start_pos - 1

        line = (line[:start_pos] + 
                'pow(' + line[start_pos:end_pos + 1] + 
                ', ' +
                line[pow_location + 1] +
                ')' +
                line[pow_location + 2:])

    print line
    # f_out.write(line)
