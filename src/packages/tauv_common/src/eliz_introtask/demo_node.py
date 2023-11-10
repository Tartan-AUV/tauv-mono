##STEP #1: THIS SHOULD NEVER BE EXECUTED DIRECTLY - SRC CODE GOES ELSEWHERE
##STEP #2: corresponding script will have same name - give it shebang #!/usr/bin/env python3 to indicate that it should be run w
##python treats folders as packages and files as modules
##STEP #3: need to use update setup.py so that it will build packages and scripts correctly
##rospack list | grep tauv
##STEP #4a: chmod +x /path/to/scripts/file
##STEp #4: rosrun [package] [script], ex rosun tauv_common demo_node
##STEP #5: Launch file. --> coordinate running pacakages. WRitten in XML 
#--> include (like C) nexted hierarhcy of launch files. other launch files we want to start
#--> pkg = pkg, type = name of executable 
##---> name is the name of what is returne

##source --> tells your bash shell to run your setup file. Bash is a wrapper file, it contians a bash script that 
##ends in .sh that actually do stuff

def main():
    print("HELLO.")
if __name__ == "__main__":
    main()