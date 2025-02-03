with open(r"data\wiki\train", "a") as output:
    file = open(r"data\raw\train", "r")
    for i, line in enumerate(file.readlines()):
        try:
            s, p, o, _, since, _, until = line.strip().split('\t')
            try:
                if since != "None" and until != "None":
                    start = int(since)
                    end = int(until)
                    if o[0] == "Q":
                        while start <= end:
                            new_line = "{0}\t{1}\t{2}\t{3}\n".format(s, p, o, start)
                            output.write(new_line) 
                            start += 1
                elif since == "None" and until != "None":
                    if o[0] == "Q":
                        new_line = "{0}\t{1}\t{2}\t{3}\n".format(s, p, o, until)
                        output.write(new_line)
                elif since != "None" and until == "None":
                    if o[0] == "Q":
                        new_line = "{0}\t{1}\t{2}\t{3}\n".format(s, p, o, since)
                        output.write(new_line)
            except IndexError as e:
                print(f"IndexError: {e}, line: {line}")
        except ValueError:
            s, p, o, timestamp = line.strip().split('\t')
            try:
                if o[0] == "Q":
                    new_line = "{0}\t{1}\t{2}\t{3}\n".format(s, p, o, timestamp)
                    output.write(new_line)
            except IndexError as e:
                print(f"IndexError: {e}, line: {line}")
    file.close()

with open(r"data\wiki\test", "a") as output:
    file = open(r"data\raw\test", "r")
    for i, line in enumerate(file.readlines()):
        try:
            s, p, o, _, since, _, until = line.strip().split('\t')
            try:
                if since != "None" and until != "None":
                    start = int(since)
                    end = int(until)
                    if o[0] == "Q":
                        while start <= end:
                            new_line = "{0}\t{1}\t{2}\t{3}\n".format(s, p, o, start)
                            output.write(new_line) 
                            start += 1
                elif since == "None" and until != "None":
                    if o[0] == "Q":
                        new_line = "{0}\t{1}\t{2}\t{3}\n".format(s, p, o, until)
                        output.write(new_line)
                elif since != "None" and until == "None":
                    if o[0] == "Q":
                        new_line = "{0}\t{1}\t{2}\t{3}\n".format(s, p, o, since)
                        output.write(new_line)
            except IndexError as e:
                print(f"IndexError: {e}, line: {line}")
        except ValueError:
            s, p, o, timestamp = line.strip().split('\t')
            try:
                if o[0] == "Q":
                    new_line = "{0}\t{1}\t{2}\t{3}\n".format(s, p, o, timestamp)
                    output.write(new_line)
            except IndexError as e:
                print(f"IndexError: {e}, line: {line}")
    file.close()

with open(r"data\wiki\valid", "a") as output:
    file = open(r"data\raw\valid", "r")
    for i, line in enumerate(file.readlines()):
        try:
            s, p, o, _, since, _, until = line.strip().split('\t')
            try:
                if since != "None" and until != "None":
                    start = int(since)
                    end = int(until)
                    if o[0] == "Q":
                        while start <= end:
                            new_line = "{0}\t{1}\t{2}\t{3}\n".format(s, p, o, start)
                            output.write(new_line) 
                            start += 1
                elif since == "None" and until != "None":
                    if o[0] == "Q":
                        new_line = "{0}\t{1}\t{2}\t{3}\n".format(s, p, o, until)
                        output.write(new_line)
                elif since != "None" and until == "None":
                    if o[0] == "Q":
                        new_line = "{0}\t{1}\t{2}\t{3}\n".format(s, p, o, since)
                        output.write(new_line)
            except IndexError as e:
                print(f"IndexError: {e}, line: {line}")
        except ValueError:
            s, p, o, timestamp = line.strip().split('\t')
            try:
                if o[0] == "Q":
                    new_line = "{0}\t{1}\t{2}\t{3}\n".format(s, p, o, timestamp)
                    output.write(new_line)
            except IndexError as e:
                print(f"IndexError: {e}, line: {line}")
    file.close()