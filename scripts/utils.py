# -*- coding: utf-8 -*-
"""
/***************************************************************************
 Sensum
                                 A QGIS plugin
 Sensum QGIS Plugin
                              -------------------
        begin                : 2014-05-27
        copyright            : (C) 2014 by Eucentre
        email                : dgaleazzo@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

import sys

class Bar(object):

    def __init__(self, max, status=""):

        self.status = status
        self.max = max
        self.dimension = 100
        self._current = 0
        self(self._current)

    def __call__(self, value, status=None):

        value = float(value)
        _current = int(value / self.max * self.dimension)
        if status:
            self.status = status
        if _current > self._current:
            wildcards = spaces = ""
            for _ in range(_current): wildcards = wildcards + "*"
            for _ in range(self.dimension-_current): wildcards = wildcards + " "
            bar = "\rSTATUS: " + self.status + " [" + wildcards + spaces + "]"
            sys.stdout.write(bar)
            sys.stdout.flush()
            self._current = _current
