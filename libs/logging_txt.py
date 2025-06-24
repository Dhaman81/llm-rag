def write_log(logfile,title,message):
    with open(logfile,"a") as f:
        f.write("================================================================================================================\n")
        f.write(str(title) + "\n")
        f.write("================================================================================================================\n")
        f.write(str(message) + "\n \n")

def open_log(logfile):
    with open(logfile) as f:
        print(f.read())