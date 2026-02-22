# src/fertility_window.py

def compute_fertile_window(ovulation_day, variability=0):
    start = ovulation_day - 5 - variability
    end = ovulation_day + variability
    return start, end
