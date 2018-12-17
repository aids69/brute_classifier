import sys
from subprocess import call

call("node addPersonById.js " + str(sys.argv[1]), cwd="/home/ftlka/Documents/diploma/fetcher", shell=True)