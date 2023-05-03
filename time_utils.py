import time

def sleep_interval(last_time: float, interval: float):
    if (sleep_time := interval-(time.time()-last_time)) > 0:
        time.sleep(sleep_time)