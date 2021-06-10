from __future__ import annotations

import os
import sys

import numpy as np
import printj
from ortools.sat.python import cp_model
from printj import ColorText as ct

# from typing import Union


class TimeVar:
    def __init__(self, hours: int, minutes: int):
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

    def __init__(self, shifts, num_workers, num_packages, num_shifts, num_keiros, sols):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._shifts = shifts
        self._num_workers = num_workers
        self._num_packages = num_packages
        self._num_shifts = num_shifts
        self._num_keiros = num_keiros
        self._solutions = set(range(sols))
        self._solution_count = 0
        self.__solution_limit = sols  # limit

    def on_solution_callback(self):
        alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if self._solution_count in self._solutions:
            print('Solution %i' % self._solution_count)
            for p in range(self._num_packages):
                # print('Package %i' % p)
                for w in range(self._num_workers):
                    is_working = False
                    for s in range(self._num_shifts):
                        for k in range(self._num_keiros):
                            if self.Value(self._shifts[(w, p, k, s)]):
                                is_working = True
                                # print('  Worker %i works shift %i' % (w, s))
                                text_worker = ct.green(
                                    f'Worker {alphabets[w]}')
                                # text_shift = ct.purple(f'shift {["9:00", "10:00", "11:00", "12:00", ][s]}')
                                text_shift = ct.purple(f'shift {["9:00", "10:00", "11:00", "12:00", ][s]}')
                                # text_shift = ct.purple(f'shift {s}')
                                text_package = ct.cyan(f'package-{p}')
                                text_keiro = ct.yellow(
                                    f'keiro {["Main2", "Main1", "SUB", ][k]}')
                                # if p in [2, 4]:
                                print(
                                    f'  {text_worker} at {text_shift} moves {text_package} through {text_keiro}')
                    # if not is_working:
                    #     print('  Worker {} does not work'.format(w))
            print()
        self._solution_count += 1

        if self._solution_count >= self.__solution_limit:
            print('Stop search after %i solutions' % self.__solution_limit)
            self.StopSearch()

    def solution_count(self):
        return self._solution_count


def main():
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
    package_orders = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
    # main2, main1, sub
    package_to_vehicle = np.array([
        [1, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ])
    package_to_keiro = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ])
    workers_to_keiro = np.array([
        [1, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 0],
    ])
    num_workers = len(workers_to_keiro)  # 4
    num_packages = len(package_to_keiro)  # 5
    num_shifts = 40
    num_tables = 6
    num_keiros = 3
    all_workers = range(num_workers)
    all_packages = range(num_packages)
    all_shifts = range(num_shifts)
    all_keiros = range(num_keiros)

    print(
        f'\nNo. of package  {len(package_to_keiro)}, No. of workers  {len(workers_to_keiro)}')
    alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    """
    available_workers_per_package = []
    for i, item in enumerate(package_to_keiro):
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
    package_to_worker = np.matmul(package_to_keiro, workers_to_keiro.T)
    available_workers_per_package = [
        [i for i, ll in enumerate(l) if ll == 1] for l in package_to_worker]
    available_keiros_per_package = [
        [i for i, ll in enumerate(l) if ll == 1] for l in package_to_keiro]

    print()
    for p, item in enumerate(available_workers_per_package):
        text_worker = ct.green(
            f'workers {"".join(alphabets[l] for l in item)}')
        text_package = ct.cyan(f'Package-{p}')
        print(f'{text_package} can be moved by {text_worker}')
    print()
    for p, item in enumerate(available_keiros_per_package):
        text_keiro = ct.yellow(
            f'keiro {" ".join(["Main2", "Main1", "SUB", ][l] for l in item)}')
        text_package = ct.cyan(f'Package-{p}')
        print(f'{text_package} can be moved to {text_keiro}')
    print()

    vehicle_to_worker = np.matmul(package_to_vehicle.T, package_to_worker)
    # sys.exit()
    # Creates the model.
    model = cp_model.CpModel()

    # Creates shift variables.
    # shifts[(w, p, k, s)]: nurse 'n' works shift 's' on package 'd'.
    shifts = {}
    for w in all_workers:
        for p in all_packages:
            for k in all_keiros:
                for s in all_shifts:
                    shifts[(w, p, k, s)] = model.NewBoolVar(
                        'shift_w%ip%ik%is%i' % (w, p, k, s))
    # for w in all_workers:
    #     for p in all_packages:
    #         for s in all_shifts:
    #             shifts[(w, p, k, s)] = model.NewBoolVar('shift_w%ip%is%i' % (w, p, k, s))

    # # # Each shift is assigned to exactly one nurse in the schedule period.
    # for p in all_packages:
    #     for s in all_shifts:
    #         model.Add(sum(shifts[(w, p, k, s)] for w in all_workers) == 1)

    # 1 worker needed per package
    # for w in all_workers:
    for p in all_packages:
        model.Add(sum(sum(sum(shifts[(w, p, k, s)] for k in all_keiros)
                  for s in all_shifts) for w in all_workers) == 1)
        model.Add(sum(sum(sum(shifts[(w, p, k, s)] for k in all_keiros)
                  for s in all_shifts) for w in available_workers_per_package[p]) == 1)

        model.Add(sum(sum(sum(shifts[(w, p, k, s)] for w in all_workers)
                  for s in all_shifts) for k in available_keiros_per_package[p]) == 1)

        # for s in all_shifts:
        #     # print(shifts[(w, p, k, s)] for k in all_keiros for w in all_workers)
        #     aa = [workers_to_keiro[w][k]*shifts[(w, p, k ,s)] for k in all_keiros for w in all_workers]
        #     print(aa)
        # model.Add(sum(aa))

    # for w in all_workers:
    #     for p in all_packages:
    #         x = [shifts[(w, p, k, s)] for s in all_shifts]
    #         # sum_all_shift_val_dec = int(''.join(map(lambda x: str(int(x)), x)), 2)bool2int(x[::-1])
    #         # model.Minimize(int(''.join(map(lambda x: str(x), x)), 2))
    #         model.Minimize(bool2int(x[::-1]))

    # for w in all_workers:
    #     for p in all_packages:
    # diag1 = []
    # for w in all_workers:
    #     for p in package_order:
    #         p1 = model.NewIntVar(0, num_packages, 'pack1_%i' % p)
    #         diag1.append(p1)
    #         model.Add(p1 == p)
        # for s in all_shifts:

    for package_order in package_orders:
        shift_before = 0
        for s in all_shifts:
            for w in all_workers:
                for k in all_keiros:
                    # s = {0, 1, 2, 3}
                    shift_before += shifts[(w, package_order[0], k, s)]
                    shift_after = 0
                    for s_inner in range(s, num_shifts):
                        for w2 in all_workers:
                            for k2 in all_keiros:
                                # (4 - {0, 1, 2, 3})
                                shift_after += shifts[(w2,
                                                       package_order[1], k2, s_inner)]
                    model.Add(shift_before <= shift_after)
                    # print(ct.yellow(f'{package_order} ')+ct.cyan(f'shift_before: ')+f'{shift_before}, '+ct.green('shift_after: ')+f'{shift_after}')

# 0, 1, 2, 3, 4, 5
# s(p=2) < s(p=4)

# abc
# d
        # model.Add(sum(sum(shifts[(w, p, k, s)] for s in all_shifts) for w in available_workers_per_package[p]) == 1)
    # # Each nurse works at most one shift per package.
    # for n in all_workers:
    #     for d in all_packages:
    #         model.Add(sum(shifts[(w, p, k, s)] for s in all_shifts) <= 1)

    # # Try to distribute the shifts evenly, so that each nurse works
    # # min_shifts_per_nurse shifts. If this is not possible, because the total
    # # number of shifts is not divisible by the number of workers, some workers will
    # # be assigned one more shift.
    # min_shifts_per_nurse = (num_shifts * num_packages) // num_workers
    # if num_shifts * num_packages % num_workers == 0:
    #     max_shifts_per_nurse = min_shifts_per_nurse
    # else:
    #     max_shifts_per_nurse = min_shifts_per_nurse + 1
    # for n in all_workers:
    #     num_shifts_worked = 0
    #     for d in all_packages:
    #         for s in all_shifts:
    #             num_shifts_worked += shifts[(w, p, k, s)]
    #     model.Add(min_shifts_per_nurse <= num_shifts_worked)
    #     model.Add(num_shifts_worked <= max_shifts_per_nurse)

    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    solver.parameters.linearization_level = 0
    # Display the first five solutions.
    a_few_solutions = 10000
    solution_printer = WorkersPartialSolutionPrinter(shifts, num_workers,
                                                     num_packages, num_shifts,
                                                     num_keiros,
                                                     a_few_solutions)
    solver.SearchForAllSolutions(model, solution_printer)

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
    main()
