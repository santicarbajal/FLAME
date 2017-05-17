__author__ = 'Santi'
# ###
# CLOSED. May, 15, 2017
# ###
# Field.py
# ###
"""Field.py
Provide a simple class for a fast implementation of Lee's Algorithm"""

from Queue import *


class Field(object):
    # ----------------------------------------- 
    def __init__(self, len, start, finish, barriers):
        """
            .
        """
        self._len = len
        self.__start = start
        self.finish = finish
        self.__finish = finish
        self._barriers = barriers
        self.__field = None
        self._build()
    # ----------------------------------------- 
    

    # ----------------------------------------- 
    def emit(self):
        """
            Cleaning issues. Done. Problem when restarting from path to path. Fixed.
        """
        q = Queue()
        q.put(self.__start)
        while not q.empty():
            index = q.get()
            l = (index[0]-1, index[1])
            r = (index[0]+1, index[1])
            u = (index[0], index[1]-1)
            d = (index[0], index[1]+1)

            if l[0] >= 0 and self[l[0]][l[1]] == 0:
                self[l[0]][l[1]] += self[index[0]][index[1]] + 1
                q.put(l)
            if r[0] < self._len and self[r[0]][r[1]] == 0:
                self[r[0]][r[1]] += self[index[0]][index[1]] + 1
                q.put(r)
            if u[1] >= 0 and self[u[0]][u[1]] == 0:
                self[u[0]][u[1]] += self[index[0]][index[1]] + 1
                q.put(u)
            if d[1] < self._len and self[d[0]][d[1]] == 0:
                self[d[0]][d[1]] += self[index[0]][index[1]] + 1
                q.put(d)
    # ----------------------------------------- 


    # ----------------------------------------- 
    def get_path(self):
        """
            Returns the current path between two points in the grid.
        """

        if self[self.finish[0]][self.finish[1]] == 0 or \
                self[self.finish[0]][self.finish[1]] == -1:
            raise

        path = []
        item = self.finish
        while not path.append(item) and item != self.__start:
            l = (item[0]-1, item[1])
            if l[0] >= 0 and self[l[0]][l[1]] == self[item[0]][item[1]] - 1:
                item = l
                continue
            r = (item[0]+1, item[1])
            if r[0] < self._len and self[r[0]][r[1]] == self[item[0]][item[1]] - 1:
                item = r
                continue
            u = (item[0], item[1]-1)
            if u[1] >= 0 and self[u[0]][u[1]] == self[item[0]][item[1]] - 1:
                item = u
                continue
            d = (item[0], item[1]+1)
            if d[1] < self._len and self[d[0]][d[1]] == self[item[0]][item[1]] - 1:
                item = d
                continue        

        return reversed(path)
    # ----------------------------------------- 


    # ----------------------------------------- 
    def update(self):
        """
            Cleaning stuff. Done.
        """
        self.__field = [[0 for i in range(self._len)] for i in range(self._len)]
    # ----------------------------------------- 


    # ----------------------------------------- 
    def _show(self):
        """
            Shows the current path.
        """
        for i in self:
            for j in i:
                print j,
            print
    # ----------------------------------------- 


    # ----------------------------------------- 
    def _clean(self):
        """
            Cleaning stuff. Auxiliary.
        """
        self[self.__finish[0]][self.__finish[1]] = 1
    # -----------------------------------------     


    # ----------------------------------------- 
    def __call__(self, *args, **kwargs):
        """
            Debug purposes. Muted.
        """
        pass
        #self._show()
    # ----------------------------------------- 


    # ----------------------------------------- 
    def __getitem__(self, item):
        """
            Accesor.
        """
        return self.__field[item]
    # ----------------------------------------- 


    # ----------------------------------------- 
    def _build(self):
        """
            Rebuilds. Marks barriers as non-accesible points in the grid.
        """
        self.__field = [[0 for i in range(self._len)] for i in range(self._len)]
        for b in self._barriers:
            self[b[0]][b[1]] = -1
        self[self.__start[0]][self.__start[1]] = 1
    # ----------------------------------------- 


