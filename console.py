import datetime

def log(*string):
    string = "[ " + str(datetime.datetime.now().strftime("%H:%M:%S")) + " ] " + str(string)
    print(string)