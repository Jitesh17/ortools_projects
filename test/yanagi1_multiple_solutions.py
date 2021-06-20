from __future__ import annotations

import os
import sys

import numpy as np
import openpyxl
import pandas as pd
import printj
from ortools.sat.python import cp_model
from printj import ColorText as ct

# from typing import Union


class TimeVar:
    def __init__(self, hours: int, minutes: int):
        while minutes > 60:
            minutes -= 60
            hours += 1
        self.hours = hours
        self.minutes = minutes
        self.time_str = f'{hours}:{minutes}'

    def __str__(self):
        return self.time_str

    def __add__(self, added_time: TimeVar):
        return TimeVar(self.hours + added_time.hours, self.minutes + added_time.minutes)

    @classmethod
    def by_string(cls, time: str):
        time_split_hour_min = time.split(":")
        hours = int(time_split_hour_min[0])
        minutes = int(time_split_hour_min[1])
        return cls(hours, minutes)
# # function to get unique values
# def unique(list1):

#     # insert the list to the set
#     list_set = set(list1)
#     # convert the set to the list
#     unique_list = (list(list_set))
#     # for x in unique_list:
#     #     print x,
#     return unique_list


class WorkersPartialSolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, shifts, num_workers, num_packages, num_shifts, num_vehicles, sols):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._shifts = shifts
        self._num_workers = num_workers
        self._num_packages = num_packages
        self._num_shifts = num_shifts
        self._num_vehicles = num_vehicles
        self._solutions = set(range(sols))
        self._solution_count = 0
        self.__solution_limit = sols  # limit 
        self.time_shifts = [TimeVar(6, 30) + TimeVar(0, 20*i) for i in range(num_shifts)]

    def on_solution_callback(self):
        alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if self._solution_count in self._solutions:
            print('Solution %i' % self._solution_count)
            data = []
            for p in range(self._num_packages):
                # print('Package %i' % p)
                data_i = []
                for s in range(self._num_shifts):
                    s_val = ct.white('0 ')
                    for w in range(self._num_workers):
                        is_working = False
                        for v in range(self._num_vehicles):
                            if self.Value(self._shifts[(w, p, v, s)]):
                                is_working = True
                                # print('  Worker %i works shift %i' % (w, s))
                                text_worker = ct.green(
                                    f'Worker {alphabets[w]}')
                                # text_shift = ct.purple(f'shift {["9:00", "10:00", "11:00", "12:00", ][s]}')
                                text_shift = ct.purple(f'shift {self.time_shifts[s]}')
                                # text_shift = ct.purple(f'shift {s}')
                                text_package = ct.cyan(f'package-{p}')
                                text_vehicle = ct.yellow(
                                    f'vehicle {v+1}')
                                # text_keiro = ct.yellow(
                                #     f'keiro {["Main2", "Main1", "SUB", ][v]}')
                                # if p in [2, 4]:
                                # print(
                                #     f'  {text_worker} at {text_shift} moves {text_package} using {text_vehicle}')
                                s_val = ct.green(f'{alphabets[w]}{v+1} ')
                    data_i.append(s_val)
                data.append(data_i)
            # data = pd.DataFrame(data, columns=self.time_shifts)
            data = pd.DataFrame(data, columns=[ct.yellow(f'  {s}') for s in self.time_shifts])
            data.index = [f'Package-{p}' for p in range(self._num_packages)]
            print()
            print(data)
                                
                    # if not is_working:
                    #     print('  Worker {} does not work'.format(w))
            print()
        self._solution_count += 1

        if self._solution_count >= self.__solution_limit:
            print('Stop search after %i solutions' % self.__solution_limit)
            self.StopSearch()

    def solution_count(self):
        return self._solution_count


def schedule(
    input_data_package, input_data_worker, input_data_location,
    a_few_solutions: int=1, 
    num_vehicles: int= 4,
    num_shifts: int= 7,
    ):
    # Data.
    # package_to_table = [
    #     [1, 0, 0, 0, 0, 0],
    #     [1, 1, 0, 0, 0, 0],
    #     [0, 0, 1, 1, 0, 0],
    #     [0, 0, 0, 0, 1, 0],
    #     [0, 0, 0, 1, 0, 1],
    #     [1, 1, 1, 1, 1, 1],
    # ]
    # workers_to_table = [
    #     [1, 1, 1, 1, 0, 1],
    #     [1, 1, 1, 1, 1, 0],
    #     [1, 1, 1, 1, 0, 1],
    #     [1, 1, 1, 1, 1, 0],
    #     ]
    printj.yellow('::::::::::::::::::: preprocess :::::::::::::::::::')
    num_locations = len(input_data_location.location)
    package_to_location = pd.crosstab(index= input_data_package['package'], columns =  input_data_package['location']).to_numpy()
    package_to_vehicle = pd.DataFrame({p: [1 if (v+1) in vehicles_list else 0 for v in range(num_vehicles)] for p, vehicles_list in enumerate(input_data_package.vehicle)}).T.to_numpy()   # num_vehicle = 4
    worker_to_vehicle = pd.DataFrame({p: [1 if (v+1) in vehicles_list else 0 for v in range(num_vehicles)] for p, vehicles_list in enumerate(input_data_worker.vehicle)}).T.to_numpy()   # num_vehicle = 4
    location_to_worker = pd.DataFrame({p: [1 if v in worker_list else 0 for v in range(num_locations)] for p, worker_list in enumerate(input_data_worker.location)}).to_numpy()   # num_keiro = 6
    package_orders = [[i,int(next_i)]  for (i, next_i) in zip(input_data_package.package, input_data_package.next) if pd.notna(next_i)]
    print("package_to_vehicle\n", package_to_vehicle)
    print("worker_to_vehicle\n", worker_to_vehicle)
    print("package_to_location\n", package_to_location)
    print("location_to_worker\n", location_to_worker)
    print("package_orders\n", package_orders)
    print()
    print()
    # print(package_to_location.to_numpy())
    # sys.exit()

    # package_orders = [[0, 1], [1, 2], ]
    # main2, main1, sub
    # package_to_vehicle = np.array([
    #     [1, 1, 1, 1],
    #     [1, 0, 0, 0],
    #     [1, 0, 0, 0],
    #     [0, 1, 1, 0],
    #     [0, 0, 1, 1],
    #     [0, 0, 1, 1],
    # ])
    # package_to_location = np.array([
    #     [1, 0, 0],
    #     [1, 0, 0],
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [1, 0, 0],
    #     [0, 0, 1],
    # ])
    # workers_to_keiro = np.array([
    #     [1, 0, 1],
    #     [1, 1, 0],
    #     [1, 0, 1],
    #     [1, 1, 0],
    # ])
    num_workers = len(input_data_worker.worker)  # 4
    num_packages = len(input_data_package.package)  # 5
    # num_tables = 6
    all_workers = range(num_workers)
    all_packages = range(num_packages)
    all_shifts = range(num_shifts)
    all_vehicles = range(num_vehicles)
    all_locations = range(num_locations)

    # print(all_vehicles)
    print(
        f'\nNo. of package  {num_packages}, No. of workers  {num_workers}')
    alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    """
    available_workers_per_package = []
    for i, item in enumerate(package_to_vehicle):
        available_workers_list = []
        for j, table in enumerate(item):
            if table == 1:
                available_workers_list += [k for k in range(len(workers_to_keiro)) if workers_to_keiro[k][j]==1]
        available_workers_list = unique(available_workers_list)
        print(f'Package-{i} can be moved by workers {"".join(alphabets[l] for l in available_workers_list)}')
        available_workers_per_package.append(available_workers_list)

    print(available_workers_per_package)
    print(np.array(available_workers_per_package))
    """
    # package_to_worker = np.matmul(package_to_location, workers_to_keiro.T)
    # print(package_to_location.shape, location_to_worker.shape)
    package_to_worker = np.matmul(package_to_location, location_to_worker)
    available_workers_per_package = [[i for i, ll in enumerate(l) if ll == 1] for l in package_to_worker]
    available_vehicles_per_package = [[i for i, ll in enumerate(l) if ll == 1] for l in package_to_vehicle]
    available_packages_per_location = [[i for i, ll in enumerate(l) if ll == 1] for l in package_to_location.T]
    available_vehicles_per_worker = [[i for i, ll in enumerate(l) if ll == 1] for l in worker_to_vehicle]

    # print()
    # for p, item in enumerate(available_workers_per_package):
    #     text_worker = ct.green(
    #         f'workers {"".join(alphabets[l] for l in item)}')
    #     text_package = ct.cyan(f'Package-{p}')
    #     print(f'{text_package} can be moved by {text_worker}')
    print()
    for w, item in enumerate(available_vehicles_per_worker):
        text_vehicle = ct.green(
            f'vehicle {", ".join(f"{l+1}" for l in item)}')
        text_worker = ct.cyan(f'worker {alphabets[w]}')
        print(f'{text_worker} can use {text_vehicle}')
    print()
    # for p, item in enumerate(available_vehicles_per_package):
    #     text_vehicle = ct.yellow(
    #         f'vehicle {" ".join(["Main2", "Main1", "SUB", ][l] for l in item)}')
    #     text_package = ct.cyan(f'Package-{p}')
    #     print(f'{text_package} can be moved to {text_vehicle}')
    # print()
    for p, (workers, vehicles) in enumerate(zip(available_workers_per_package, available_vehicles_per_package)):
        text_worker = ct.green(
            f'workers {", ".join(alphabets[l] for l in workers)}')
        text_vehicle = ct.yellow(
            f'vehicle {", ".join(str(v) for v in vehicles)}')
        text_package = ct.cyan(f'Package-{p}')
        print(f'{text_package} can be moved by \t{text_worker}\tusing {text_vehicle}')
    print()
    for l, item in enumerate(available_packages_per_location):
        text_package = ct.cyan(f'package {", ".join(f"{i}" for i in item)}')
        text_location = ct.green(
            f'location {l}')
        print(f'{text_location} carries {text_package}')
    print()

    # vehicle_to_worker = np.matmul(package_to_vehicle.T, package_to_worker)
    # sys.exit()
    # Creates the model.
    model = cp_model.CpModel()

    # Creates shift variables.
    # shifts[(w, p, v, s)]: nurse 'n' works shift 's' on package 'd'.
    shifts = {}
    for w in all_workers:
        for p in all_packages:
            for v in all_vehicles:
                for s in all_shifts:
                    shifts[(w, p, v, s)] = model.NewBoolVar(
                        'shift_w%ip%iv%is%i' % (w, p, v, s))
    package_quantity = 1
    for pi, p in enumerate(all_packages):
        package_quantity = input_data_package.quantity[pi]

        # package_quantity = min(package_quantity, )
        # 1 worker needed per package
        
        model.Add(sum(sum(sum(shifts[(w, p, v, s)] for v in all_vehicles)
                  for s in all_shifts) for w in all_workers) <= package_quantity)
        # 1 available worker per package
        model.Add(sum(sum(sum(shifts[(w, p, v, s)] for v in all_vehicles)
                  for s in all_shifts) for w in available_workers_per_package[p]) <= package_quantity)
        # 1 available vehicle per package
        model.Add(sum(sum(sum(shifts[(w, p, v, s)] for w in all_workers)
                  for s in all_shifts) for v in available_vehicles_per_package[p]) <= package_quantity)
        
        for s in all_shifts:
            model.Add(sum(sum(shifts[(w, p, v, s)] for v in all_vehicles) for w in all_workers) <= 1)

    # Capacity constraints
    # location_filled = dict.fromkeys(input_data_location.location, 0)
    for l in all_locations:
        # total_quantity = sum(input_data_package.quantity[p] for p in available_packages_per_location[l])
        # print(total_quantity)
        # location_filled[l] += sum(sum(sum(sum(shifts[(w, p, v, s)]for v in all_vehicles) for s in all_shifts) for w in all_workers) for p in available_packages_per_location[l])
        capacity = input_data_location.capacity[l]
        # decay = input_data_location.decay[l]
        # current_empty_space = capacity #  10 = 3 nimotsu + 7 empty_space = 2 nimotsu + 8 empty_space = 10 empty_space
        # empty_space = max(current_empty_space + decay*1, capacity)  # using max: empty space can't be more than the capacity of the shelf/location
        # empty_space = min(total_quantity, empty_space)  # using min: 
        # model.Add(location_filled[l]==empty_space)
        for si in all_shifts:
            for p in available_packages_per_location[l]:
                decay = input_data_package.decay[p]
                # sum_package = sum(sum(sum(sum(shifts[(w, p, v, s)]for v in all_vehicles) for w in all_workers) for s in range(si+1)))
                sum_package = sum(sum(sum(shifts[(w, p, v, s)]for v in all_vehicles) for w in all_workers) for s in range(si+1))
                sum_package += input_data_package.yesterday[p] 
                model.Add(2*sum_package-decay*(si+1) <= 2*capacity)
                model.Add(2*sum_package-decay*(si+1) >= 0)
    
    # a = sum(sum(sum(sum(sum(shifts[(w, p, v, s)] for v in all_vehicles) for w in all_workers) for s in range(si+1)) for p in all_packages) for s in all_shifts)
    # model.Maximize(a)
    # 1 W, V, S for 1 package
    for s in all_shifts:
        for w in all_workers:
            for v in all_vehicles:
                model.Add(sum(shifts[(w, p, v, s)] for p in all_packages) <= 1)

    printj.red(f'all_vehicles: {list(all_vehicles)}')
    printj.red(f'available_vehicles_per_worker: {available_vehicles_per_worker}')
    for w in all_workers:
        for v in all_vehicles:
        # 1 available vehicle per worker
            if v in available_vehicles_per_worker[w]:
                model.Add(sum(sum(shifts[(w, p, v, s)] for p in all_packages)
                        for s in all_shifts) >= 0)
            else:
                model.Add(sum(sum(shifts[(w, p, v, s)] for p in all_packages)
                        for s in all_shifts) == 0)

    #  package_order   # s(p=2) < s(p=4)
    for package_order in package_orders:
        shift_before = 0
        for s in all_shifts:
            for w in all_workers:
                for v in all_vehicles:
                    # s = {0, 1, 2, 3}
                    shift_before += shifts[(w, package_order[0], v, s)]
                    shift_after = 0
                    # for s2 in range(s, num_shifts):
                    for s2 in range(s+2):
                        if s2 < num_shifts:
                            for w2 in all_workers:
                                for v2 in all_vehicles:
                                    # (4 - {0, 1, 2, 3})
                                    shift_after += shifts[(w2, package_order[1], v2, s2)]
                    # model.Add(shift_before <= shift_after)
                    model.Add(shift_before == shift_after).OnlyEnforceIf(shifts[(w, package_order[0], v, s)])
                    model.Add(shift_before == shift_after).OnlyEnforceIf(shifts[(w, package_order[1], v, s)])

    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    solver.parameters.linearization_level = 0
    # Display the first five solutions.
    # a_few_solutions = 100
    solution_printer = WorkersPartialSolutionPrinter(shifts, num_workers,
                                                     num_packages, num_shifts,
                                                     num_vehicles,
                                                     a_few_solutions)
    printj.yellow('::::::::::::::::::::: Output :::::::::::::::::::::')
    solver.SearchForAllSolutions(model, solution_printer)
    # solver.Solve(model)

    # Statistics.
    print()
    print('Statistics')
    print('  - conflicts       : %i' % solver.NumConflicts())
    print('  - branches        : %i' % solver.NumBranches())
    print('  - wall time       : %f s' % solver.WallTime())
    print('  - solutions found : %i' % solution_printer.solution_count())
    # assert solution_printer.solution_count() == 5


def bool2int(x):
    y = 0
    for i, j in enumerate(x):
        y += j << i
    return y


if __name__ == '__main__':

    printj.yellow('::::::::::::::::::::: Input :::::::::::::::::::::')
    path = "test/xl.xlsx"
    input_data_package = pd.read_excel(open(path, 'rb'),
              sheet_name='Sheet_package') 
    input_data_worker = pd.read_excel(open(path, 'rb'),
              sheet_name='Sheet_worker') 
    input_data_location = pd.read_excel(open(path, 'rb'),
              sheet_name='Sheet_location') 
    input_data_package.vehicle = [[int(i) for i in v.split(",")] for v in input_data_package.vehicle]
    input_data_package.next = [v if isinstance(v, int) else None for v in input_data_package.next]
    input_data_worker.location = [[int(i) for i in v.split(",")] for v in input_data_worker.location]
    input_data_worker.vehicle = [[int(i) for i in v.split(",")] for v in input_data_worker.vehicle]
    num_vehicles = 4
    """
    input_data_package = pd.DataFrame({
        "package": [0, 1],
        "quantity": [2, 2],  
        "location": [0, 0],
        "vehicle": [[1, 2, 3, 4], [1]],
        # "next": [None, 2, 3, 4, 5, None],
        "next": [None, None], # Only work if the quantity is same,
        "yesterday": [1, 2], 
    })
    input_data_worker = pd.DataFrame({
        "worker": list("ABCD"),
        "location": [[0, 2], [0, 1], [0, 2], [0, 1]],
        "vehicle": [[1 ], [1, 2, 3, 4], [1], [1, 2, 3, 4]],
    })
    input_data_location = pd.DataFrame({
        "location": list(range(3)),
        # "decay": [1, 1, 1],  # per shift
        "capacity": [4, 3, 3],
    }) # 4 - num_pack_loc0_shift + f(decay*(0, shift))
    """
    """
    input_data_package = pd.DataFrame({
        "package": [0, 1, 2, 3, 4, 5],
        "quantity": [20000, 20000, 20000, 20000, 20000, 200000],  
        "decay": [1, 1, 1, 1, 1, 1],
        "location": [0, 0, 0, 1, 0, 2],
        "vehicle": [[1, 2, 3, 4], [1], [1], [2, 3], [3, 4], [1]],
        # "next": [None, 2, 3, 4, 5, None],
        "next": [None, None, None, None, None, None], # Only work if the quantity is same,
        "yesterday": [0, 0, 0, 0, 0, 0], 
    })
    input_data_worker = pd.DataFrame({
        "worker": list("ABCD"),
        "location": [[0, 2], [0, 1], [0, 2], [0, 1]],
        "vehicle": [[1 ], [1, 2, 3, 4], [1], [1, 2, 3, 4]],
    })
    input_data_location = pd.DataFrame({
        "location": list(range(3)),
        # "decay": [1, 1, 1],  # per shift
        "capacity": [10, 10, 10],
    }) # 4 - num_pack_loc0_shift + f(decay*(0, shift))
    """
    """
    input_data_package = pd.DataFrame({
        "package": [0],
        "quantity": [6],  
        "location": [0],
        "vehicle": [[1, 2, 3, 4]],
        # "next": [None, 2, 3, 4, 5, None],
        "next": [None], # Only work if the quantity is same
        "yesterday": [1], 
    })
    input_data_worker = pd.DataFrame({
        "worker": list("ABCD"),
        "location": [[0], [0], [0], [0]],
        "vehicle": [[1 ], [1, 2, 3, 4], [1], [1, 2, 3, 4]],
    })
    input_data_location = pd.DataFrame({
        "location": list(range(1)),
        # "decay_rate": [1, 1, 1],  # per shift
        "capacity": [1],
    }) # 4 - num_pack_loc0_shift + f(decay*(0, shift))
    """
    # """
    print(input_data_package)
    print(input_data_worker)
    print(input_data_location)
    print()
    schedule(input_data_package, input_data_worker, input_data_location,
    a_few_solutions =1, 
    num_vehicles = 4,
    num_shifts = 7,)
    # """
    """  
    wb = openpyxl.Workbook()
    sheet = wb.active

    sheet_title = sheet.title
    wb.save(path)

    print("active sheet title: " + sheet_title)
    """