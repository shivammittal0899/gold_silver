from strategy import *
# def log2(msg):
#     with open("static/logs.txt", "a") as f:
#         f.write(f"{datetime.now().strftime('%H:%M:%S')}  {msg}\n")
TRAILLING_STOP = False
def run_trailling_strategy():
    global kite, TRAILLING_STOP
    TRAILLING_STOP = False     # reset when starting

def stop_trailling_strategy():
    global TRAILLING_STOP
    TRAILLING_STOP = True
    log("⛔ Strategy STOP requested.")
