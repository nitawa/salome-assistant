#!/bin/bash
if [ "${SALOME_ASSISTANT_ROOT_DIR}" == "" ]; then
    export SALOME_ASSISTANT_ROOT_DIR=$(cd $(dirname ${BASH_SOURCE[0]});pwd)
fi
source $SALOME_ASSISTANT_ROOT_DIR/bin/activate
salome-assistant
