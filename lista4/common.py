def interpret_corr(r):
    r_abs = abs(r)
    if r_abs < 0.2:
        return "bardzo słaba"
    elif r_abs < 0.4:
        return "słaba"
    elif r_abs < 0.6:
        return "umiarkowana"
    elif r_abs < 0.8:
        return "silna"
    else:
        return "bardzo silna"