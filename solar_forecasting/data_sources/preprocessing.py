import numpy as np
import datetime
import os

def date_index(actual_day):

    '''Takes a date and returns the list of relevant indices in the data.'''

    data = np.load(os.path.join(os.environ.get("LOCAL_DATA_PATH"), "raw_data", os.environ.get("FINE_NAME_X_TRAIN")), allow_pickle = True)
    riv = data['datetime']
    date_format = []

    for i in range(riv.shape[0]):
        date_format.append(riv[i].date())

    dx = datetime.datetime.strptime(actual_day, "%d-%m-%Y")
    dx = dx.date()

    index = []

    for i in range(riv.shape[0]):
        if dx == date_format[i]:
            index.append(i)

    if not index:
        return ("The date out of range.")
    else:
        return index

def number_of_observations(dx):

    '''Takes a date and returns the number of observations taken that day.'''

    #dx = datetime.datetime.strptime(actual_day, "%d-%m-%Y")
    #dx = dx.date()

    d1 = datetime.date(2012,1,1)
    d2 = datetime.date(2012,2,20)
    d3 = datetime.date(2012,4,12)
    d4 = datetime.date(2012,8,28)
    d5 = datetime.date(2012,10,20)
    d6 = datetime.date(2012,12,31)

    obs = 0

    if dx>=d1 and dx<=d2:
        obs=4
    elif dx>d2 and dx<=d3:
        obs=5
    elif dx>d3 and dx<=d4:
        obs=6
    elif dx>d4 and dx<=d5:
        obs=5
    elif dx>d5 and dx<=d6:
        obs=4

    if obs!=0:
        return obs
    else:
        return ("The date out of range.")

if __name__ == "__main__":
    date_index(actual_day='01-01-2012')
    print('Index retrieved!')
    number_of_observations(actual_day='01-01-2012')
    print('Number of observations retrieved!')
