#!/usr/bin/env bash
# usage: setupEnvironment.sh
#
# This script will automatically set the KAIPYHOME environment variable to the root of this kaiju installation,
#   and also add the appropriate scripts folders to your PATH and PYTHONPATH environment variables

# borrowed this one-liner to get the directory containing the script:
# https://stackoverflow.com/questions/59895/how-can-i-get-the-source-directory-of-a-bash-script-from-within-the-script-itsel
# modified to also work with zsh
# https://stackoverflow.com/questions/9901210/bash-source0-equivalent-in-zsh
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-${(%):-%x}}" )" &> /dev/null && pwd )"

# strip off the "/kaipy/scripts" folder to get to the root of the repository
ROOT_DIR="$(echo "$SCRIPT_DIR" | sed 's:/kaipy/scripts$::')"

# using only export= and not export+= to work with bash and zsh
export KAIPYHOME="$ROOT_DIR"

export PYTHONPATH="$KAIPYHOME:$PYTHONPATH"

export PATH="$PATH:$KAIPYHOME/scripts:$KAIPYHOME/scripts/datamodel:$KAIPYHOME/scripts/helio:$KAIPYHOME/scripts/legacy:$KAIPYHOME/scripts/preproc:$KAIPYHOME/scripts/postproc:$KAIPYHOME/scripts/quicklook"
