display_usage() { 
	echo "This script assumes you have set up flatiron in your ssh config\n\
and must be given a worker and a port." 
	echo "\nUsage: $0 workerXXXX:PPPP \n" 
	} 
# if not exactly one argument is supplied, display usage 
	if [  $# != 1 ] 
	then 
		display_usage
		exit 1
	fi 
 
# check whether user had supplied -h or --help . If yes display usage 
	if [[ ( $@ == "--help") ||  $@ == "-h" ]] 
	then 
		display_usage
		exit 0
	fi

ssh -L127.0.0.1:8888:$1 flatiron
