from __future__ import annotations
# from typing import Union

class TimeVar:
    def __init__(self, hours:int, minutes:int):
        while minutes > 60:
            minutes -= 60
            hours += 1
        self.hours = hours
        self.minutes = minutes
        
        self.time_str = f'{hours}:{minutes}'
    
    def __str__(self):
        return self.time_str
    
    def __add__(self, added_time:TimeVar):
        hours = self.hours + added_time.hours
        minutes = self.minutes + added_time.minutes
        return TimeVar(self.hours + added_time.hours, self.minutes + added_time.minutes)
    
    @classmethod
    def by_string(cls, time:str):
        time_split_hour_min = time.split(":")
        hours = int(time_split_hour_min[0])
        minutes = int(time_split_hour_min[1])
        return cls(hours, minutes)



# t = TimeVar.by_string("8:30")
# r = TimeVar(2, 40)
# e = t + r
# print(f'{t} + {r} = {e}')

print(list("ABCD"))

print([(i , j) for i in range(3) for j in range(2)])