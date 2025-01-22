
class TimeSeries(dict):
    """
    TimeSeries is a standard Python Dictionary with some helper
    functions useful for data processing collections of time series.
    """

    def __init__(self, indict={}):
        dict.__init__(self)
        # not "self.__keydict" because I want it to be easily accessible by subclasses!
        self._keydict = {}

        for entry in indict:
            self[entry] = indict[entry]

    def append(self, key, name, units, data):
        """
        Append a new time series to the TimeSeries object.

        Args:
            key (str): Unique identifier of the variable.
            name (str): More descriptive name of the variable.
            units (str): Units of the variable (e.g., grams).
            data (list): Time series data array.

        Raises:
            KeyError: If the TimeSeries object already has the specified key.

        Example:
            >>> ts = TimeSeries()
            >>> ts.append('v', 'Velocity', 'm/s', [1, 2, 3])
        """
        if key in self:
            raise KeyError('Error: Dictionary already has key \"%s\".' % key)
        self[key] = {'name': name, 'data': data, 'units': units}

    def getUnits(self, key):
        """
        Get the units of a variable in the TimeSeries object.

        Args:
            key (str): Unique identifier of the variable.

        Returns:
            str: Units of the variable.

        Example:
            >>> ts = TimeSeries({'v': {'name': 'Velocity', 'data': [1, 2, 3], 'units': 'm/s'}})
            >>> ts.getUnits('v')
            'm/s'
        """
        return self[key]['units']

    def setUnits(self, key, units):
        """
        Set the units of a variable in the TimeSeries object.

        Args:
            key (str): Unique identifier of the variable.
            units (str): Units of the variable.

        Example:
            >>> ts = TimeSeries({'v': {'name': 'Velocity', 'data': [1, 2, 3], 'units': 'm/s'}})
            >>> ts.setUnits('v', 'km/h')
        """
        self[key]['units'] = units

    def getName(self, key):
        """
        Get the name of a variable in the TimeSeries object.

        Args:
            key (str): Unique identifier of the variable.

        Returns:
            str: Name of the variable.

        Example:
            >>> ts = TimeSeries({'v': {'name': 'Velocity', 'data': [1, 2, 3], 'units': 'm/s'}})
            >>> ts.getName('v')
            'Velocity'
        """
        return self[key]['name']

    def setName(self, key, name):
        """
        Set the name of a variable in the TimeSeries object.

        Args:
            key (str): Unique identifier of the variable.
            name (str): Name of the variable.

        Example:
            >>> ts = TimeSeries({'v': {'name': 'Velocity', 'data': [1, 2, 3], 'units': 'm/s'}})
            >>> ts.setName('v', 'Speed')
        """
        self[key]['name'] = name

    def getData(self, key):
        """
        Get the data of a variable in the TimeSeries object.

        Args:
            key (str): Unique identifier of the variable.

        Returns:
            list: Time series data of the variable.

        Example:
            >>> ts = TimeSeries({'v': {'name': 'Velocity', 'data': [1, 2, 3], 'units': 'm/s'}})
            >>> ts.getData('v')
            [1, 2, 3]
        """
        return self[key]['data']

    def setData(self, key, data, index=None):
        """
        Set the data of a variable in the TimeSeries object.

        Args:
            key (str): Unique identifier of the variable.
            data (list): Time series data of the variable.
            index (int, optional): Index of the data to set. If not provided, the entire data array will be replaced.

        Example:
            >>> ts = TimeSeries({'v': {'name': 'Velocity', 'data': [1, 2, 3], 'units': 'm/s'}})
            >>> ts.setData('v', [4, 5, 6])
        """
        try:
            self[key]['data'][index] = data
        except TypeError:
            self[key]['data'] = data

def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()