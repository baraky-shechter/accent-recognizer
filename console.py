import datetime

def log(string):
    string = "[ " + str(datetime.datetime.now().strftime("%H:%M:%S")) + " ] " + string
    print(string)