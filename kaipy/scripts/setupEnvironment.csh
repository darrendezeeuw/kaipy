#!/usr/bin/env csh
# usage: source setupEnvironment.csh
#
# This script will automatically set the KAIPYHOME environment variable to the root of this kaipy installation,
#   and also add the appropriate scripts folders to your PATH and PYTHONPATH environment variables
 
# borrowed this one-liner to get the directory containing the script:
# https://stackoverflow.com/questions/59895/how-can-i-get-the-source-directory-of-a-bash-script-from-within-the-script-itsel
 
set rootdir = `dirname $0`
set SCRIPT_DIR = `cd $rootdir && pwd`
echo $SCRIPT_DIR 

# strip off the "/kaipy/scripts" folder to get to the root of the repository
set ROOT_DIR = `cd $rootdir/../.. && pwd`
echo $ROOT_DIR 
setenv KAIPYHOME $ROOT_DIR
echo $KAIPYHOME

setenv PYTHONPATH ${KAIPYHOME}:${PYTHONPATH}
echo $PYTHONPATH
setenv PATH ${PATH}:${SCRIPT_DIR}:${SCRIPT_DIR}/datamodel:${SCRIPT_DIR}/helio:${SCRIPT_DIR}/preproc:${SCRIPT_DIR}/postproc:${SCRIPT_DIR}/quicklook
echo $PATH
