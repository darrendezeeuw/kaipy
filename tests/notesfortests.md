# Hints and notes for running tests #

Kaipy uses the pytest package for doing unit tests.  This file provides a few hints and tips for running the tests with this package

## Running tests ##

The `pytest` command will run all the tests execpt those in the `test_satcomp_cdasws.py` which are marked as slow since the pull data down from CDAWeb.  To run those tests use `pytest --runslow`.  

To run tests in a specific file use `pyteset tests/test_kaijson.py`.  To run only a single test use `pytest tests/test_kaijson.py -k test_dumps`.  

To get the coverage of the unit tests run `pytest --cov=kaipy`

## To dos ##
Kaijson is not handling datetime objects correctly at the moment so those tests are disable until a fix is developed.
