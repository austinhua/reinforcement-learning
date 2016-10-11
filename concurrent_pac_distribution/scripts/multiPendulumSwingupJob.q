#!/bin/bash
#
#$ -S /bin/bash -cwd
#$ -o stdoutstderr.out -j y
#$ -M jpazis@cs.duke.edu -m b
#$ -M jpazis@cs.duke.edu -m e
#$ -l mem_free=4G
#$ -t 1-100

words=( $( uname -a ) )
export SYSTEM="${words[0]}"
export HOST="${words[1]}"
export VERSION="${words[2]}"
if [[ $SYSTEM == 'IRIX' || $SYSTEM == 'IRIX64' ]]; then
	export HOSTTYPE="${words[5]}"
elif [[ "$SYSTEM" == 'Linux' ]]; then
	export HOSTTYPE="${words[12]}"
	export PLATFORM="${words[13]}"
else
	export HOSTTYPE="${words[4]}"
fi
unset words

if [[ ! ${DOMAINNAME+1} ]]; then
	export DOMAINNAME=$( domainname )
fi

# remove the domainname from the hostname.
if [[ $HOST == *".$DOMAINNAME" ]]; then
	#HOST=$(expr "$HOST" ":" "\(.*\)\.$DOMAINNAME" )
	HOST=${HOST%.$DOMAINNAME}
fi

export dKnown=0.1
export epsilonA=0.001
export epsilonB=0.1
export regularizer=1.0
export maxNumberOfEpisodes=256
export numberOfDomains=1

hostname >> ROSfreePendulumSwingupExperiment$dKnown.$epsilonA.$epsilonB.$regularizer.$maxNumberOfEpisodes.$SGE_TASK_ID.out
./scripts/ROSfreeMultiPendulumSwingupExperiment.py $dKnown $epsilonA $epsilonB $regularizer $maxNumberOfEpisodes $numberOfDomains >> ROSfreePendulumSwingupExperiment$dKnown.$epsilonA.$epsilonB.$regularizer.$maxNumberOfEpisodes.$SGE_TASK_ID.out

